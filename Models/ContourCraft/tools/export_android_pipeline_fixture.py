import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.hood_minimal import HoodMinimal, DefaultsOverride
from tools.cpu_scatter_sum import aggregate_edges


def write_float_bin(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.astype(np.float32).tofile(str(path))


def write_int_bin(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.astype(np.int32).tofile(str(path))


def write_shape(path: Path, shape) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(shape)))


def edge_mlp_input(tgt: torch.Tensor, src: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
    return torch.cat([tgt, src, feat], dim=1)


def aggregate(edge_index_src_tgt: np.ndarray, edge_values: np.ndarray, num_target: int) -> np.ndarray:
    src = edge_index_src_tgt[0]
    tgt = edge_index_src_tgt[1]
    edges_rcv_snd = np.stack([tgt, src], axis=1)
    out, _, _, _ = aggregate_edges(edges_rcv_snd, edge_values, num_target)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-root", type=str, default="Data/ccraft_data")
    parser.add_argument("--out-dir", type=str,
                        default="Android/HoodOnnxTest/app/src/main/assets/pipeline")
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

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sizes (match exported ONNX)
    N_cloth = 8
    N_obstacle = 8
    E_BLOCK = 8
    E_ENCODER = 16

    F = learned._latent_size
    n_node = learned.n_nodefeatures
    n_edge_mesh = learned.n_edgefeatures_mesh
    n_edge_world = learned.n_edgefeatures_world
    n_edge_coarse = learned.n_edgefeatures_coarse

    # Random raw features
    cloth_raw = np.random.randn(N_cloth, n_node).astype(np.float32)
    obstacle_raw = np.random.randn(N_obstacle, n_node).astype(np.float32)

    mesh_raw = np.random.randn(E_ENCODER, n_edge_mesh).astype(np.float32)
    coarse0_raw = np.random.randn(E_ENCODER, n_edge_coarse).astype(np.float32)
    coarse1_raw = np.random.randn(E_ENCODER, n_edge_coarse).astype(np.float32)
    coarse2_raw = np.random.randn(E_ENCODER, n_edge_coarse).astype(np.float32)
    world_direct_raw = np.random.randn(E_BLOCK, n_edge_world).astype(np.float32)
    world_inverse_raw = np.random.randn(E_BLOCK, n_edge_world).astype(np.float32)

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

    # Encode nodes (note: obstacle nodes are kept zero in the pipeline)
    with torch.no_grad():
        t_cloth = learned.node_encoder(torch.from_numpy(cloth_raw))
    t_obstacle = torch.zeros(N_obstacle, F)

    # Save encoder outputs for debugging
    write_float_bin(out_dir / "expected_node_encoder_cloth.bin", t_cloth.detach().cpu().numpy())
    write_shape(out_dir / "expected_node_encoder_cloth_shape.json", t_cloth.shape)
    write_float_bin(out_dir / "expected_node_encoder_obstacle.bin", t_obstacle.detach().cpu().numpy())
    write_shape(out_dir / "expected_node_encoder_obstacle_shape.json", t_obstacle.shape)

    # Encode edges
    with torch.no_grad():
        t_mesh_latent_full = learned.edgeset_encoders["mesh"](torch.from_numpy(mesh_raw))
        t_coarse0_latent_full = learned.edgeset_encoders["coarse0"](torch.from_numpy(coarse0_raw))
        t_coarse1_latent_full = learned.edgeset_encoders["coarse1"](torch.from_numpy(coarse1_raw))
        t_coarse2_latent_full = learned.edgeset_encoders["coarse2"](torch.from_numpy(coarse2_raw))
        t_world_cat = torch.cat(
            [torch.from_numpy(world_direct_raw), torch.from_numpy(world_inverse_raw)], dim=0
        )
        t_world_cat_latent = learned.edgeset_encoders["world"](t_world_cat)

        t_mesh_latent = t_mesh_latent_full[:E_BLOCK]
        t_coarse0_latent = t_coarse0_latent_full[:E_BLOCK]
        t_coarse1_latent = t_coarse1_latent_full[:E_BLOCK]
        t_coarse2_latent = t_coarse2_latent_full[:E_BLOCK]
        t_world_direct_latent = t_world_cat_latent[:E_BLOCK]
        t_world_inverse_latent = t_world_cat_latent[E_BLOCK:]

    # Save edge encoder outputs for debugging
    write_float_bin(out_dir / "expected_edge_encoder_mesh.bin", t_mesh_latent.detach().cpu().numpy())
    write_shape(out_dir / "expected_edge_encoder_mesh_shape.json", t_mesh_latent.shape)
    write_float_bin(out_dir / "expected_edge_encoder_coarse0.bin", t_coarse0_latent.detach().cpu().numpy())
    write_shape(out_dir / "expected_edge_encoder_coarse0_shape.json", t_coarse0_latent.shape)
    write_float_bin(out_dir / "expected_edge_encoder_coarse1.bin", t_coarse1_latent.detach().cpu().numpy())
    write_shape(out_dir / "expected_edge_encoder_coarse1_shape.json", t_coarse1_latent.shape)
    write_float_bin(out_dir / "expected_edge_encoder_coarse2.bin", t_coarse2_latent.detach().cpu().numpy())
    write_shape(out_dir / "expected_edge_encoder_coarse2_shape.json", t_coarse2_latent.shape)
    write_float_bin(out_dir / "expected_edge_encoder_world_direct.bin", t_world_direct_latent.detach().cpu().numpy())
    write_shape(out_dir / "expected_edge_encoder_world_direct_shape.json", t_world_direct_latent.shape)
    write_float_bin(out_dir / "expected_edge_encoder_world_inverse.bin", t_world_inverse_latent.detach().cpu().numpy())
    write_shape(out_dir / "expected_edge_encoder_world_inverse_shape.json", t_world_inverse_latent.shape)

    nodes = {"cloth": t_cloth, "obstacle": t_obstacle}
    edges = {
        "mesh_edge": t_mesh_latent,
        "coarse_edge0": t_coarse0_latent,
        "coarse_edge1": t_coarse1_latent,
        "coarse_edge2": t_coarse2_latent,
        "world_direct": t_world_direct_latent,
        "world_inverse": t_world_inverse_latent,
    }

    edge_indices = {
        "mesh_edge": edge_index_mesh,
        "coarse_edge0": edge_index_coarse0,
        "coarse_edge1": edge_index_coarse1,
        "coarse_edge2": edge_index_coarse2,
        "world_direct": edge_index_world_direct,
        "world_inverse": edge_index_world_inverse,
    }

    blocks_cfg = []

    # Forward all blocks
    for level_idx, level in enumerate(learned.levels):
        for block_idx, block in enumerate(level):
            edge_keys = block.edge_keys
            blocks_cfg.append({
                "level": level_idx,
                "block": block_idx,
                "edge_keys": edge_keys,
            })

            t_edge_mlps = block.edge_processor_dict
            t_node_mlp = block.node_processor_dict["node"]

            def torch_edge_update(edge_key, src_key, tgt_key):
                src_idx, tgt_idx = edge_indices[edge_key]
                src = nodes[src_key][src_idx]
                tgt = nodes[tgt_key][tgt_idx]
                feat = edges[edge_key if edge_key != "world_edge" else src_key]
                out = t_edge_mlps["world_edge" if "world" in edge_key else edge_key](
                    edge_mlp_input(tgt, src, feat)
                )
                return out

            # Update edges
            updated_world_direct = torch_edge_update("world_direct", "obstacle", "cloth")
            updated_world_inverse = torch_edge_update("world_inverse", "cloth", "obstacle")

            updated_mesh = torch_edge_update("mesh_edge", "cloth", "cloth") if "mesh_edge" in edge_keys else None
            updated_coarse0 = torch_edge_update("coarse_edge0", "cloth", "cloth") if "coarse_edge0" in edge_keys else None
            updated_coarse1 = torch_edge_update("coarse_edge1", "cloth", "cloth") if "coarse_edge1" in edge_keys else None
            updated_coarse2 = torch_edge_update("coarse_edge2", "cloth", "cloth") if "coarse_edge2" in edge_keys else None

            edges["world_direct"] = edges["world_direct"] + updated_world_direct
            edges["world_inverse"] = edges["world_inverse"] + updated_world_inverse
            if updated_mesh is not None:
                edges["mesh_edge"] = edges["mesh_edge"] + updated_mesh
            if updated_coarse0 is not None:
                edges["coarse_edge0"] = edges["coarse_edge0"] + updated_coarse0
            if updated_coarse1 is not None:
                edges["coarse_edge1"] = edges["coarse_edge1"] + updated_coarse1
            if updated_coarse2 is not None:
                edges["coarse_edge2"] = edges["coarse_edge2"] + updated_coarse2

            # Aggregation (cloth/obstacle)
            def agg(edge_key, num_target):
                return aggregate(edge_indices[edge_key], edges[edge_key].detach().cpu().numpy(), num_target)

            agg_world_cloth = agg("world_direct", N_cloth)
            agg_world_obs = agg("world_inverse", N_obstacle)

            agg_map_cloth = {"world_edge": agg_world_cloth}
            agg_map_obs = {"world_edge": agg_world_obs}
            if updated_mesh is not None:
                agg_map_cloth["mesh_edge"] = agg("mesh_edge", N_cloth)
            if updated_coarse0 is not None:
                agg_map_cloth["coarse_edge0"] = agg("coarse_edge0", N_cloth)
            if updated_coarse1 is not None:
                agg_map_cloth["coarse_edge1"] = agg("coarse_edge1", N_cloth)
            if updated_coarse2 is not None:
                agg_map_cloth["coarse_edge2"] = agg("coarse_edge2", N_cloth)

            # Node inputs in edge_keys order
            node_in_cloth = [nodes["cloth"].detach().cpu().numpy()]
            node_in_obs = [nodes["obstacle"].detach().cpu().numpy()]
            for k in edge_keys:
                node_in_cloth.append(agg_map_cloth.get(k, np.zeros((N_cloth, F), dtype=np.float32)))
                # obstacle branch only receives world edge messages
                if k == "world_edge":
                    node_in_obs.append(agg_map_obs.get(k, np.zeros((N_obstacle, F), dtype=np.float32)))
                else:
                    node_in_obs.append(np.zeros((N_obstacle, F), dtype=np.float32))

            node_in_cloth = np.concatenate(node_in_cloth, axis=1).astype(np.float32)
            node_in_obs = np.concatenate(node_in_obs, axis=1).astype(np.float32)

            with torch.no_grad():
                updated_nodes_cloth = t_node_mlp(torch.from_numpy(node_in_cloth))
                updated_nodes_obs = t_node_mlp(torch.from_numpy(node_in_obs))

            # Extra debug for first block
            if level_idx == 0 and block_idx == 0:
                dbg_dir = out_dir / "blocks"
                dbg_dir.mkdir(parents=True, exist_ok=True)
                write_float_bin(dbg_dir / "block_0_0_agg_world_cloth.bin", agg_world_cloth)
                write_float_bin(dbg_dir / "block_0_0_agg_mesh.bin", agg_map_cloth.get("mesh_edge", np.zeros_like(agg_world_cloth)))
                write_float_bin(dbg_dir / "block_0_0_agg_coarse0.bin", agg_map_cloth.get("coarse_edge0", np.zeros_like(agg_world_cloth)))
                write_float_bin(dbg_dir / "block_0_0_node_in_cloth.bin", node_in_cloth)
                write_shape(dbg_dir / "block_0_0_node_in_cloth_shape.json", node_in_cloth.shape)
                write_float_bin(dbg_dir / "block_0_0_node_out_cloth.bin", updated_nodes_cloth.detach().cpu().numpy())
                write_shape(dbg_dir / "block_0_0_node_out_cloth_shape.json", updated_nodes_cloth.shape)
                write_float_bin(dbg_dir / "block_0_0_updated_world_direct.bin", updated_world_direct.detach().cpu().numpy())
                write_float_bin(dbg_dir / "block_0_0_updated_world_inverse.bin", updated_world_inverse.detach().cpu().numpy())
                if updated_mesh is not None:
                    write_float_bin(dbg_dir / "block_0_0_updated_mesh.bin", updated_mesh.detach().cpu().numpy())
                if updated_coarse0 is not None:
                    write_float_bin(dbg_dir / "block_0_0_updated_coarse0.bin", updated_coarse0.detach().cpu().numpy())

            nodes["cloth"] = nodes["cloth"] + updated_nodes_cloth
            nodes["obstacle"] = nodes["obstacle"] + updated_nodes_obs

            # Save per-block cloth node state for debugging
            blk_dir = out_dir / "blocks"
            blk_dir.mkdir(parents=True, exist_ok=True)
            blk_name = f"block_{level_idx}_{block_idx}_cloth_nodes"
            write_float_bin(blk_dir / f"{blk_name}.bin", nodes["cloth"].detach().cpu().numpy())
            write_shape(blk_dir / f"{blk_name}_shape.json", nodes["cloth"].shape)

    # Decoder (cloth)
    with torch.no_grad():
        output = learned.decoder(nodes["cloth"]).detach().cpu().numpy().astype(np.float32)

    # Write config
    config = {
        "N_cloth": N_cloth,
        "N_obstacle": N_obstacle,
        "E_block": E_BLOCK,
        "E_encoder": E_ENCODER,
        "latent_size": F,
        "output_size": learned._output_size,
        "blocks": blocks_cfg,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Write inputs
    write_float_bin(out_dir / "cloth_raw.bin", cloth_raw)
    write_shape(out_dir / "cloth_raw_shape.json", cloth_raw.shape)
    write_float_bin(out_dir / "obstacle_raw.bin", obstacle_raw)
    write_shape(out_dir / "obstacle_raw_shape.json", obstacle_raw.shape)

    write_float_bin(out_dir / "mesh_raw.bin", mesh_raw)
    write_shape(out_dir / "mesh_raw_shape.json", mesh_raw.shape)
    write_float_bin(out_dir / "coarse0_raw.bin", coarse0_raw)
    write_shape(out_dir / "coarse0_raw_shape.json", coarse0_raw.shape)
    write_float_bin(out_dir / "coarse1_raw.bin", coarse1_raw)
    write_shape(out_dir / "coarse1_raw_shape.json", coarse1_raw.shape)
    write_float_bin(out_dir / "coarse2_raw.bin", coarse2_raw)
    write_shape(out_dir / "coarse2_raw_shape.json", coarse2_raw.shape)
    write_float_bin(out_dir / "world_direct_raw.bin", world_direct_raw)
    write_shape(out_dir / "world_direct_raw_shape.json", world_direct_raw.shape)
    write_float_bin(out_dir / "world_inverse_raw.bin", world_inverse_raw)
    write_shape(out_dir / "world_inverse_raw_shape.json", world_inverse_raw.shape)

    write_int_bin(out_dir / "edge_index_mesh.bin", edge_index_mesh)
    write_shape(out_dir / "edge_index_mesh_shape.json", edge_index_mesh.shape)
    write_int_bin(out_dir / "edge_index_coarse0.bin", edge_index_coarse0)
    write_shape(out_dir / "edge_index_coarse0_shape.json", edge_index_coarse0.shape)
    write_int_bin(out_dir / "edge_index_coarse1.bin", edge_index_coarse1)
    write_shape(out_dir / "edge_index_coarse1_shape.json", edge_index_coarse1.shape)
    write_int_bin(out_dir / "edge_index_coarse2.bin", edge_index_coarse2)
    write_shape(out_dir / "edge_index_coarse2_shape.json", edge_index_coarse2.shape)
    write_int_bin(out_dir / "edge_index_world_direct.bin", edge_index_world_direct)
    write_shape(out_dir / "edge_index_world_direct_shape.json", edge_index_world_direct.shape)
    write_int_bin(out_dir / "edge_index_world_inverse.bin", edge_index_world_inverse)
    write_shape(out_dir / "edge_index_world_inverse_shape.json", edge_index_world_inverse.shape)

    write_float_bin(out_dir / "expected_output.bin", output)
    write_shape(out_dir / "expected_output_shape.json", output.shape)

    print(f"Wrote pipeline fixture to {out_dir}")


if __name__ == "__main__":
    main()
