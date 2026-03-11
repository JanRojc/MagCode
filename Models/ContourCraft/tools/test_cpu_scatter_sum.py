import argparse

import numpy as np
import torch

from cpu_scatter_sum import aggregate_edges


def main():
    parser = argparse.ArgumentParser(description="Validate CPU CSR scatter-sum vs torch index_add.")
    parser.add_argument("--nodes", type=int, default=128)
    parser.add_argument("--edges", type=int, default=512)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    num_nodes = args.nodes
    num_edges = args.edges
    feat = args.features

    # Random edge list: (source, target)
    src = rng.integers(0, num_nodes, size=(num_edges,), dtype=np.int64)
    tgt = rng.integers(0, num_nodes, size=(num_edges,), dtype=np.int64)
    edge_index = np.stack([src, tgt], axis=0)

    edge_features = rng.standard_normal(size=(num_edges, feat)).astype(np.float32)

    # Torch reference (aggregate to target)
    edge_features_t = torch.from_numpy(edge_features)
    tgt_idx = torch.from_numpy(tgt)
    out_t = torch.zeros(num_nodes, feat, dtype=edge_features_t.dtype)
    out_t.index_add_(0, tgt_idx, edge_features_t)

    # CPU CSR aggregation (receiver=target, sender=source)
    edges_rcv_snd = np.stack([tgt, src], axis=1)
    out_cpu, _, _, _ = aggregate_edges(edges_rcv_snd, edge_features, num_nodes)

    max_diff = np.max(np.abs(out_t.numpy() - out_cpu))
    


if __name__ == "__main__":
    main()
