from collections import defaultdict

import torch
from torch import nn
from torch_geometric.nn import MessagePassing


class BaseBlock(MessagePassing):
    def __init__(self, edge_processor_dict, node_processor_dict):
        # We don't use the Inspector API at all – keep this simple.
        super().__init__(aggr='add')

        edge_processor_dict = {k: v() for k, v in edge_processor_dict.items()}
        self.edge_processor_dict = nn.ModuleDict(edge_processor_dict)

        node_processor_dict = {k: v() for k, v in node_processor_dict.items()}
        self.node_processor_dict = nn.ModuleDict(node_processor_dict)

    def forward(self, sample):
        sample = self.propagate(sample)
        return sample

    def message(self, edge_processor_key, target_features_i=None, source_features_j=None, edge_features=None):
        """
        All three inputs here are already per-edge:
        - source_features: [E, F_src]
        - target_features: [E, F_tgt]
        - edge_features:   [E, F_edge]
        """
        in_features = []
        for features in [target_features_i, source_features_j, edge_features]:
            if features is not None:
                in_features.append(features)

        assert (len(in_features) > 0)

        in_features = torch.cat(in_features, dim=-1)
        out_features = self.edge_processor_dict[edge_processor_key](in_features)
        return out_features

    def aggregate_nodes(self, edge_features, edge_index, size, **kwargs):
        """
        Simple manual aggregation: aggregate messages to target nodes.
        edge_features: [E, F]
        edge_index: [2, E] (source, target)
        size: (N_source, N_target)
        """
        _, num_target = size
        device = edge_features.device
        E, F = edge_features.shape

        # Aggregate into target nodes (standard message-passing semantics).
        tgt_idx = edge_index[1]

        out = torch.zeros(num_target, F, device=device, dtype=edge_features.dtype)
        out.index_add_(0, tgt_idx, edge_features)

        return out

    def update_edge_features(self, sample, edge_processor_key, source_key, edge_key, target_key):
        """
        Convert node features -> per-edge features using edge_index,
        then call message().
        """

        mesh_edges = sample[source_key, edge_key, target_key]
        source = sample[source_key]
        target = sample[target_key]

        edge_index = mesh_edges.edge_index  # [2, E]
        # PyG convention: edge_index[0] = source, edge_index[1] = target
        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        # Make them per-edge by indexing with edge_index
        source_features = source.node_features[src_idx]   # [E, F_src]
        target_features = target.node_features[dst_idx]   # [E, F_tgt]
        edge_features = mesh_edges.features               # [E, F_edge]

        return self.message(
            edge_processor_key=edge_processor_key,
            target_features_i=target_features,
            source_features_j=source_features,
            edge_features=edge_features,
        )

    def update(self, features_list, node_processor_key):
        input_features = torch.cat(features_list, dim=1)
        out_features = self.node_processor_dict[node_processor_key](input_features)
        # print(f'update\tfrom {input_features.shape} to {out_features.shape}')
        return out_features


def make_edgesets_dict(n_coarse_levels, body=True, selfcoll=False, new_body=False, separate_attraction_edges=False):
    edge_sets_full = defaultdict(dict)
    edge_sets_full['mesh']['source'] = 'cloth'
    edge_sets_full['mesh']['edge_key'] = 'mesh_edge'
    edge_sets_full['mesh']['target'] = 'cloth'

    for i in range(n_coarse_levels):
        edge_sets_full[f'coarse{i}']['source'] = 'cloth'
        edge_sets_full[f'coarse{i}']['edge_key'] = f'coarse_edge{i}'
        edge_sets_full[f'coarse{i}']['target'] = 'cloth'

    if body:
        if new_body:
            direct_key = 'body_direct'
            inverse_key = 'body_inverse'
            edge_key = 'body_edge'
        else:
            direct_key = 'world_direct'
            inverse_key = 'world_inverse'
            edge_key = 'world_edge'

        edge_sets_full[direct_key]['source'] = 'obstacle'
        edge_sets_full[direct_key]['edge_key'] = edge_key
        edge_sets_full[direct_key]['target'] = 'cloth'

        edge_sets_full[inverse_key]['source'] = 'cloth'
        edge_sets_full[inverse_key]['edge_key'] = edge_key
        edge_sets_full[inverse_key]['target'] = 'obstacle'

    if selfcoll:
        if separate_attraction_edges:
            for k in ['repulsion', 'attraction']:
                edge_sets_full[k]['source'] = 'cloth'
                edge_sets_full[k]['edge_key'] = f'{k}_edge'
                edge_sets_full[k]['target'] = 'cloth'
        else:
            edge_sets_full['world_cloth']['source'] = 'cloth'
            edge_sets_full['world_cloth']['edge_key'] = 'world_edge'
            edge_sets_full['world_cloth']['target'] = 'cloth'

    return edge_sets_full