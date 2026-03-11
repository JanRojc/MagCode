import argparse
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


def run_mlp_onnx(session, x):
    return session.run(None, {session.get_inputs()[0].name: x})[0]


def edge_mlp_input(target, source, edge_feat):
    return np.concatenate([target, source, edge_feat], axis=1).astype(np.float32)


def aggregate(edge_index_src_tgt, edge_values, num_target):
    # edge_index: [2,E] (source, target)
    src = edge_index_src_tgt[0]
    tgt = edge_index_src_tgt[1]
    edges_rcv_snd = np.stack([tgt, src], axis=1)
    out, _, _, _ = aggregate_edges(edges_rcv_snd, edge_values, num_target)
    return out


def main():
    parser = argparse.ArgumentParser(description="Full CPU pipeline: ONNX MLPs + CPU scatter-sum vs PyTorch.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-root", type=str, default="Data/ccraft_data")
    parser.add_argument("--onnx-dir", type=str,
                        default="Models/ContourCraft/tools/onnx_out_opset18_lnprim_embedded")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    defaults = DefaultsOverride(
        data_root=args.data_root,
        project_dir=str(Path("Models/ContourCraft").resolve()),
        config_dir=str(Path("Models/ContourCraft/configs").resolve()),
    )
    model = HoodMinimal(checkpoint_path=args.checkpoint, defaults=defaults).model
    model.eval()
    learned = model._learned_model

    # Sizes
    F = learned._latent_size
    n_node = learned.n_nodefeatures
    n_edge_mesh = learned.n_edgefeatures_mesh
    n_edge_world = learned.n_edgefeatures_world
    n_edge_coarse = learned.n_edgefeatures_coarse

    # Fixed shapes to match exported ONNX
    # Edge encoders were exported with batch=16, edge MLPs with batch=8.
    N_cloth = 8
    N_obstacle = 8
    E_BLOCK = 8
    E_ENCODER = 16

    # Random raw features
    cloth_raw = np.random.randn(N_cloth, n_node).astype(np.float32)
    obstacle_raw = np.random.randn(N_obstacle, n_node).astype(np.float32)

    mesh_raw = np.random.randn(E_ENCODER, n_edge_mesh).astype(np.float32)
    coarse0_raw = np.random.randn(E_ENCODER, n_edge_coarse).astype(np.float32)
    coarse1_raw = np.random.randn(E_ENCODER, n_edge_coarse).astype(np.float32)
    coarse2_raw = np.random.randn(E_ENCODER, n_edge_coarse).astype(np.float32)
    world_direct_raw = np.random.randn(E_BLOCK, n_edge_world).astype(np.float32)
    world_inverse_raw = np.random.randn(E_BLOCK, n_edge_world).astype(np.float32)

    # Edge indices (source, target)
    def rand_edges(num_edges, n_src, n_tgt):
        src = np.random.randint(0, n_src, size=(num_edges,), dtype=np.int64)
        tgt = np.random.randint(0, n_tgt, size=(num_edges,), dtype=np.int64)
        return np.stack([src, tgt], axis=0)

    edge_index_mesh = rand_edges(E_BLOCK, N_cloth, N_cloth)
    edge_index_coarse0 = rand_edges(E_BLOCK, N_cloth, N_cloth)
    edge_index_coarse1 = rand_edges(E_BLOCK, N_cloth, N_cloth)
    edge_index_coarse2 = rand_edges(E_BLOCK, N_cloth, N_cloth)
    edge_index_world_direct = rand_edges(E_BLOCK, N_obstacle, N_cloth)  # obstacle -> cloth
    edge_index_world_inverse = np.stack([edge_index_world_direct[1], edge_index_world_direct[0]], axis=0)

    # ONNX sessions
    onnx_dir = Path(args.onnx_dir).resolve()
    preferred = ["CPUExecutionProvider"]

    sess_node_encoder = create_session(str((onnx_dir / "node_encoder.onnx").resolve()), preferred)
    sess_edge_mesh = create_session(str((onnx_dir / "edge_encoder_mesh.onnx").resolve()), preferred)
    sess_edge_world = create_session(str((onnx_dir / "edge_encoder_world.onnx").resolve()), preferred)
    sess_edge_coarse0 = create_session(str((onnx_dir / "edge_encoder_coarse0.onnx").resolve()), preferred)
    sess_edge_coarse1 = create_session(str((onnx_dir / "edge_encoder_coarse1.onnx").resolve()), preferred)
    sess_edge_coarse2 = create_session(str((onnx_dir / "edge_encoder_coarse2.onnx").resolve()), preferred)
    sess_decoder = create_session(str((onnx_dir / "node_decoder.onnx").resolve()), preferred)

    # Node encoder (PyTorch)
    with torch.no_grad():
        t_node_in = torch.from_numpy(cloth_raw)
        t_node_latent = learned.node_encoder(t_node_in)
    # Obstacle inactive -> zeros
    t_obstacle_latent = torch.zeros(N_obstacle, F)

    # Node encoder (ONNX)
    o_node_latent = run_mlp_onnx(sess_node_encoder, cloth_raw)
    o_obstacle_latent = np.zeros((N_obstacle, F), dtype=np.float32)

    # Edge encoders (PyTorch)
    with torch.no_grad():
        t_mesh_latent_full = learned.edgeset_encoders["mesh"](torch.from_numpy(mesh_raw))
        t_coarse0_latent_full = learned.edgeset_encoders["coarse0"](torch.from_numpy(coarse0_raw))
        t_coarse1_latent_full = learned.edgeset_encoders["coarse1"](torch.from_numpy(coarse1_raw))
        t_coarse2_latent_full = learned.edgeset_encoders["coarse2"](torch.from_numpy(coarse2_raw))
        t_world_cat = torch.cat(
            [torch.from_numpy(world_direct_raw), torch.from_numpy(world_inverse_raw)], dim=0
        )
        t_world_cat_latent = learned.edgeset_encoders["world"](t_world_cat)
        t_world_direct_latent = t_world_cat_latent[:E_BLOCK]
        t_world_inverse_latent = t_world_cat_latent[E_BLOCK:]
        t_mesh_latent = t_mesh_latent_full[:E_BLOCK]
        t_coarse0_latent = t_coarse0_latent_full[:E_BLOCK]
        t_coarse1_latent = t_coarse1_latent_full[:E_BLOCK]
        t_coarse2_latent = t_coarse2_latent_full[:E_BLOCK]

    # Edge encoders (ONNX)
    o_mesh_latent_full = run_mlp_onnx(sess_edge_mesh, mesh_raw)
    o_coarse0_latent_full = run_mlp_onnx(sess_edge_coarse0, coarse0_raw)
    o_coarse1_latent_full = run_mlp_onnx(sess_edge_coarse1, coarse1_raw)
    o_coarse2_latent_full = run_mlp_onnx(sess_edge_coarse2, coarse2_raw)
    o_world_cat = run_mlp_onnx(sess_edge_world, np.concatenate([world_direct_raw, world_inverse_raw], axis=0))
    o_world_direct_latent = o_world_cat[:E_BLOCK]
    o_world_inverse_latent = o_world_cat[E_BLOCK:]
    o_mesh_latent = o_mesh_latent_full[:E_BLOCK]
    o_coarse0_latent = o_coarse0_latent_full[:E_BLOCK]
    o_coarse1_latent = o_coarse1_latent_full[:E_BLOCK]
    o_coarse2_latent = o_coarse2_latent_full[:E_BLOCK]

    # Initialize features dicts
    t_nodes = {"cloth": t_node_latent, "obstacle": t_obstacle_latent}
    o_nodes = {"cloth": o_node_latent, "obstacle": o_obstacle_latent}

    t_edges = {
        "mesh_edge": t_mesh_latent,
        "coarse_edge0": t_coarse0_latent,
        "coarse_edge1": t_coarse1_latent,
        "coarse_edge2": t_coarse2_latent,
        "world_direct": t_world_direct_latent,
        "world_inverse": t_world_inverse_latent,
    }
    o_edges = {
        "mesh_edge": o_mesh_latent,
        "coarse_edge0": o_coarse0_latent,
        "coarse_edge1": o_coarse1_latent,
        "coarse_edge2": o_coarse2_latent,
        "world_direct": o_world_direct_latent,
        "world_inverse": o_world_inverse_latent,
    }

    edge_indices = {
        "mesh_edge": edge_index_mesh,
        "coarse_edge0": edge_index_coarse0,
        "coarse_edge1": edge_index_coarse1,
        "coarse_edge2": edge_index_coarse2,
        "world_direct": edge_index_world_direct,
        "world_inverse": edge_index_world_inverse,
    }

    # Iterate over exported blocks in model order
    for level_idx, level in enumerate(learned.levels):
        for block_idx, block in enumerate(level):
            # ONNX sessions for this block
            def block_path(kind):
                return onnx_dir / "blocks" / f"block_{level_idx}_{block_idx}_{kind}.onnx"

            sess_edge = {
                "mesh_edge": create_session(str(block_path("edge_mesh_edge")), preferred)
                if block_path("edge_mesh_edge").exists() else None,
                "coarse_edge0": create_session(str(block_path("edge_coarse_edge0")), preferred)
                if block_path("edge_coarse_edge0").exists() else None,
                "coarse_edge1": create_session(str(block_path("edge_coarse_edge1")), preferred)
                if block_path("edge_coarse_edge1").exists() else None,
                "coarse_edge2": create_session(str(block_path("edge_coarse_edge2")), preferred)
                if block_path("edge_coarse_edge2").exists() else None,
                "world_edge": create_session(str(block_path("edge_world_edge")), preferred)
                if block_path("edge_world_edge").exists() else None,
            }
            sess_node = create_session(str(block_path("node")), preferred)

            # PyTorch MLPs
            t_edge_mlps = block.edge_processor_dict
            t_node_mlp = block.node_processor_dict["node"]
            edge_keys = block.edge_keys

            # Edge update per edgeset (PyTorch)
            def torch_edge_update(edge_key, src_key, tgt_key):
                src_idx, tgt_idx = edge_indices[edge_key]
                src = t_nodes[src_key][src_idx]
                tgt = t_nodes[tgt_key][tgt_idx]
                feat = t_edges[edge_key if edge_key != "world_edge" else src_key]
                out = t_edge_mlps["world_edge" if "world" in edge_key else edge_key](
                    torch.cat([tgt, src, feat], dim=1)
                )
                return out

            # Edge update per edgeset (ONNX)
            def onnx_edge_update(edge_key, src_key, tgt_key, sess):
                src_idx, tgt_idx = edge_indices[edge_key]
                src = o_nodes[src_key][src_idx]
                tgt = o_nodes[tgt_key][tgt_idx]
                feat = o_edges[edge_key if edge_key != "world_edge" else src_key]
                inp = edge_mlp_input(tgt, src, feat)
                return run_mlp_onnx(sess, inp)

            # World edges (shared MLP)
            updated_world_direct_t = torch_edge_update("world_direct", "obstacle", "cloth")
            updated_world_inverse_t = torch_edge_update("world_inverse", "cloth", "obstacle")
            updated_world_direct_o = onnx_edge_update("world_direct", "obstacle", "cloth", sess_edge["world_edge"])
            updated_world_inverse_o = onnx_edge_update("world_inverse", "cloth", "obstacle", sess_edge["world_edge"])

            # Mesh/coarse edges
            updated_mesh_t = torch_edge_update("mesh_edge", "cloth", "cloth") if sess_edge["mesh_edge"] else None
            updated_coarse0_t = torch_edge_update("coarse_edge0", "cloth", "cloth") if sess_edge["coarse_edge0"] else None
            updated_coarse1_t = torch_edge_update("coarse_edge1", "cloth", "cloth") if sess_edge["coarse_edge1"] else None
            updated_coarse2_t = torch_edge_update("coarse_edge2", "cloth", "cloth") if sess_edge["coarse_edge2"] else None

            updated_mesh_o = onnx_edge_update("mesh_edge", "cloth", "cloth", sess_edge["mesh_edge"]) if sess_edge["mesh_edge"] else None
            updated_coarse0_o = onnx_edge_update("coarse_edge0", "cloth", "cloth", sess_edge["coarse_edge0"]) if sess_edge["coarse_edge0"] else None
            updated_coarse1_o = onnx_edge_update("coarse_edge1", "cloth", "cloth", sess_edge["coarse_edge1"]) if sess_edge["coarse_edge1"] else None
            updated_coarse2_o = onnx_edge_update("coarse_edge2", "cloth", "cloth", sess_edge["coarse_edge2"]) if sess_edge["coarse_edge2"] else None

            # Residual edge update
            t_edges["world_direct"] = t_edges["world_direct"] + updated_world_direct_t
            t_edges["world_inverse"] = t_edges["world_inverse"] + updated_world_inverse_t
            o_edges["world_direct"] = o_edges["world_direct"] + updated_world_direct_o
            o_edges["world_inverse"] = o_edges["world_inverse"] + updated_world_inverse_o

            if updated_mesh_t is not None:
                t_edges["mesh_edge"] = t_edges["mesh_edge"] + updated_mesh_t
                o_edges["mesh_edge"] = o_edges["mesh_edge"] + updated_mesh_o
            if updated_coarse0_t is not None:
                t_edges["coarse_edge0"] = t_edges["coarse_edge0"] + updated_coarse0_t
                o_edges["coarse_edge0"] = o_edges["coarse_edge0"] + updated_coarse0_o
            if updated_coarse1_t is not None:
                t_edges["coarse_edge1"] = t_edges["coarse_edge1"] + updated_coarse1_t
                o_edges["coarse_edge1"] = o_edges["coarse_edge1"] + updated_coarse1_o
            if updated_coarse2_t is not None:
                t_edges["coarse_edge2"] = t_edges["coarse_edge2"] + updated_coarse2_t
                o_edges["coarse_edge2"] = o_edges["coarse_edge2"] + updated_coarse2_o

            # Aggregate for cloth nodes
            def agg_for_cloth(updated, edge_key):
                return aggregate(edge_indices[edge_key], to_numpy(updated), N_cloth)

            agg_world_cloth_t = agg_for_cloth(updated_world_direct_t, "world_direct")
            agg_world_cloth_o = agg_for_cloth(updated_world_direct_o, "world_direct")

            agg_map_t = {"world_edge": agg_world_cloth_t}
            agg_map_o = {"world_edge": agg_world_cloth_o}

            if updated_mesh_t is not None:
                agg_map_t["mesh_edge"] = agg_for_cloth(updated_mesh_t, "mesh_edge")
                agg_map_o["mesh_edge"] = agg_for_cloth(updated_mesh_o, "mesh_edge")
            if updated_coarse0_t is not None:
                agg_map_t["coarse_edge0"] = agg_for_cloth(updated_coarse0_t, "coarse_edge0")
                agg_map_o["coarse_edge0"] = agg_for_cloth(updated_coarse0_o, "coarse_edge0")
            if updated_coarse1_t is not None:
                agg_map_t["coarse_edge1"] = agg_for_cloth(updated_coarse1_t, "coarse_edge1")
                agg_map_o["coarse_edge1"] = agg_for_cloth(updated_coarse1_o, "coarse_edge1")
            if updated_coarse2_t is not None:
                agg_map_t["coarse_edge2"] = agg_for_cloth(updated_coarse2_t, "coarse_edge2")
                agg_map_o["coarse_edge2"] = agg_for_cloth(updated_coarse2_o, "coarse_edge2")

            # Node input for cloth
            node_in_t = [to_numpy(t_nodes["cloth"])]
            node_in_o = [o_nodes["cloth"]]
            for k in edge_keys:
                node_in_t.append(agg_map_t.get(k, np.zeros((N_cloth, F), dtype=np.float32)))
                node_in_o.append(agg_map_o.get(k, np.zeros((N_cloth, F), dtype=np.float32)))

            node_in_t = np.concatenate(node_in_t, axis=1).astype(np.float32)
            node_in_o = np.concatenate(node_in_o, axis=1).astype(np.float32)

            with torch.no_grad():
                updated_nodes_t = t_node_mlp(torch.from_numpy(node_in_t))
            updated_nodes_o = run_mlp_onnx(sess_node, node_in_o)

            # Residual node update (cloth only)
            t_nodes["cloth"] = t_nodes["cloth"] + updated_nodes_t
            o_nodes["cloth"] = o_nodes["cloth"] + updated_nodes_o

            max_diff_nodes = float(np.max(np.abs(to_numpy(t_nodes["cloth"]) - o_nodes["cloth"])))
            print(f"block_{level_idx}_{block_idx} cloth_node max_abs_diff={max_diff_nodes:.6e}")

    # Decoder (cloth only)
    with torch.no_grad():
        t_out = learned.decoder(t_nodes["cloth"])
    o_out = run_mlp_onnx(sess_decoder, o_nodes["cloth"].astype(np.float32))

    print("final decoder max_abs_diff", float(np.max(np.abs(to_numpy(t_out) - o_out))))


if __name__ == "__main__":
    main()
