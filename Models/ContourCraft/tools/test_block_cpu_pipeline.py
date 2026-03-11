import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.cpu_scatter_sum import aggregate_edges
from tools.hood_minimal import HoodMinimal, DefaultsOverride
from tools.ort_runtime import create_session


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def run_edge_mlp_onnx(session, target, source, edge_feat):
    inp = np.concatenate([target, source, edge_feat], axis=1).astype(np.float32)
    out = session.run(None, {session.get_inputs()[0].name: inp})[0]
    return out


def run_edge_mlp_torch(edge_mlp, target, source, edge_feat):
    inp = torch.cat([target, source, edge_feat], dim=1)
    with torch.no_grad():
        out = edge_mlp(inp)
    return out


def aggregate(edge_index_src_tgt, edge_values, num_target):
    # edge_index_src_tgt: [2,E] (source, target)
    src = edge_index_src_tgt[0]
    tgt = edge_index_src_tgt[1]
    edges_rcv_snd = np.stack([tgt, src], axis=1)
    out, _, _, _ = aggregate_edges(edges_rcv_snd, edge_values, num_target)
    return out


def main():
    parser = argparse.ArgumentParser(description="Validate ONNX block pipeline vs PyTorch using CPU scatter-sum.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-root", type=str, default="Data/ccraft_data")
    parser.add_argument("--onnx-dir", type=str,
                        default="Models/ContourCraft/tools/onnx_out_opset18_lnprim_embedded")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    os.environ.setdefault("TMPDIR", str(Path("/tmp").resolve()))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load PyTorch model and first block
    defaults = DefaultsOverride(
        data_root=args.data_root,
        project_dir=str(Path("Models/ContourCraft").resolve()),
        config_dir=str(Path("Models/ContourCraft/configs").resolve()),
    )
    model = HoodMinimal(checkpoint_path=args.checkpoint, defaults=defaults).model
    model.eval()
    learned = model._learned_model
    block = learned.levels[0][0]

    edge_mlp_mesh = block.edge_processor_dict["mesh_edge"]
    edge_mlp_coarse0 = block.edge_processor_dict["coarse_edge0"]
    edge_mlp_world = block.edge_processor_dict["world_edge"]
    node_mlp = block.node_processor_dict["node"]
    edge_keys = block.edge_keys  # sorted list

    # Synthetic node features (latent space)
    N_cloth = 8
    N_obstacle = 6
    F = learned._latent_size

    cloth_nodes = torch.randn(N_cloth, F)
    obstacle_nodes = torch.randn(N_obstacle, F)

    # Synthetic edges
    def rand_edges(num_edges, n_src, n_tgt):
        src = np.random.randint(0, n_src, size=(num_edges,), dtype=np.int64)
        tgt = np.random.randint(0, n_tgt, size=(num_edges,), dtype=np.int64)
        return np.stack([src, tgt], axis=0)

    E_mesh = 8
    E_coarse0 = 8
    E_world = 8

    edge_index_mesh = rand_edges(E_mesh, N_cloth, N_cloth)
    edge_index_coarse0 = rand_edges(E_coarse0, N_cloth, N_cloth)
    edge_index_world_direct = rand_edges(E_world, N_obstacle, N_cloth)  # source obstacle -> target cloth
    edge_index_world_inverse = np.stack([edge_index_world_direct[1], edge_index_world_direct[0]], axis=0)

    # Edge latent features
    edge_feat_mesh = torch.randn(E_mesh, F)
    edge_feat_coarse0 = torch.randn(E_coarse0, F)
    edge_feat_world_direct = torch.randn(E_world, F)
    edge_feat_world_inverse = torch.randn(E_world, F)

    # Prepare ONNX sessions
    onnx_dir = Path(args.onnx_dir).resolve()
    preferred = ["CPUExecutionProvider"]
    sess_edge_mesh = create_session(str((onnx_dir / "blocks/block_0_0_edge_mesh_edge.onnx").resolve()), preferred)
    sess_edge_coarse0 = create_session(str((onnx_dir / "blocks/block_0_0_edge_coarse_edge0.onnx").resolve()), preferred)
    sess_edge_world = create_session(str((onnx_dir / "blocks/block_0_0_edge_world_edge.onnx").resolve()), preferred)
    sess_node = create_session(str((onnx_dir / "blocks/block_0_0_node.onnx").resolve()), preferred)

    # Edge MLPs (PyTorch)
    t_mesh = run_edge_mlp_torch(
        edge_mlp_mesh,
        cloth_nodes[edge_index_mesh[1]],
        cloth_nodes[edge_index_mesh[0]],
        edge_feat_mesh,
    )
    t_coarse0 = run_edge_mlp_torch(
        edge_mlp_coarse0,
        cloth_nodes[edge_index_coarse0[1]],
        cloth_nodes[edge_index_coarse0[0]],
        edge_feat_coarse0,
    )
    t_world_direct = run_edge_mlp_torch(
        edge_mlp_world,
        cloth_nodes[edge_index_world_direct[1]],
        obstacle_nodes[edge_index_world_direct[0]],
        edge_feat_world_direct,
    )
    t_world_inverse = run_edge_mlp_torch(
        edge_mlp_world,
        obstacle_nodes[edge_index_world_inverse[1]],
        cloth_nodes[edge_index_world_inverse[0]],
        edge_feat_world_inverse,
    )

    # Edge MLPs (ONNX)
    o_mesh = run_edge_mlp_onnx(
        sess_edge_mesh,
        to_numpy(cloth_nodes[edge_index_mesh[1]]),
        to_numpy(cloth_nodes[edge_index_mesh[0]]),
        to_numpy(edge_feat_mesh),
    )
    o_coarse0 = run_edge_mlp_onnx(
        sess_edge_coarse0,
        to_numpy(cloth_nodes[edge_index_coarse0[1]]),
        to_numpy(cloth_nodes[edge_index_coarse0[0]]),
        to_numpy(edge_feat_coarse0),
    )
    o_world_direct = run_edge_mlp_onnx(
        sess_edge_world,
        to_numpy(cloth_nodes[edge_index_world_direct[1]]),
        to_numpy(obstacle_nodes[edge_index_world_direct[0]]),
        to_numpy(edge_feat_world_direct),
    )
    o_world_inverse = run_edge_mlp_onnx(
        sess_edge_world,
        to_numpy(obstacle_nodes[edge_index_world_inverse[1]]),
        to_numpy(cloth_nodes[edge_index_world_inverse[0]]),
        to_numpy(edge_feat_world_inverse),
    )

    # Edge MLP diffs
    print("edge_mesh max_abs_diff", float(np.max(np.abs(to_numpy(t_mesh) - o_mesh))))
    print("edge_coarse0 max_abs_diff", float(np.max(np.abs(to_numpy(t_coarse0) - o_coarse0))))
    print("edge_world_direct max_abs_diff", float(np.max(np.abs(to_numpy(t_world_direct) - o_world_direct))))
    print("edge_world_inverse max_abs_diff", float(np.max(np.abs(to_numpy(t_world_inverse) - o_world_inverse))))

    # Aggregate for cloth nodes (PyTorch uses updated edge features)
    agg_mesh = aggregate(edge_index_mesh, o_mesh, N_cloth)
    agg_coarse0 = aggregate(edge_index_coarse0, o_coarse0, N_cloth)
    agg_world_cloth = aggregate(edge_index_world_direct, o_world_direct, N_cloth)

    # Build node input in edge_key order
    agg_map = {
        "mesh_edge": agg_mesh,
        "coarse_edge0": agg_coarse0,
        "world_edge": agg_world_cloth,
    }

    node_inputs = [to_numpy(cloth_nodes)]
    for key in edge_keys:
        node_inputs.append(agg_map.get(key, np.zeros((N_cloth, F), dtype=np.float32)))
    node_in = np.concatenate(node_inputs, axis=1).astype(np.float32)

    # Node MLP ONNX
    o_node = sess_node.run(None, {sess_node.get_inputs()[0].name: node_in})[0]

    # Node MLP PyTorch
    with torch.no_grad():
        t_node = node_mlp(torch.from_numpy(node_in))

    print("node_mlp max_abs_diff", float(np.max(np.abs(to_numpy(t_node) - o_node))))


if __name__ == "__main__":
    main()
