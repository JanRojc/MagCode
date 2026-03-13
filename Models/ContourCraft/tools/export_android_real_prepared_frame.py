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
from utils.common import copy_pyg_batch
from utils.validation import create_one_sequence_dataloader


def write_float_bin(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.astype(np.float32).tofile(str(path))


def write_int_bin(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.astype(np.int32).tofile(str(path))


def write_shape(path: Path, shape) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([int(x) for x in shape]))


def normalizer_stats(normalizer) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        mean = normalizer._mean().detach().cpu().numpy().astype(np.float32).reshape(-1)
        std = normalizer._std_with_epsilon().detach().cpu().numpy().astype(np.float32).reshape(-1)
    return mean, std


def edge_mlp_input(tgt: torch.Tensor, src: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
    return torch.cat([tgt, src, feat], dim=1)


def aggregate(edge_index_src_tgt: np.ndarray, edge_values: np.ndarray, num_target: int) -> np.ndarray:
    src = edge_index_src_tgt[0]
    tgt = edge_index_src_tgt[1]
    edges_rcv_snd = np.stack([tgt, src], axis=1)
    out, _, _, _ = aggregate_edges(edges_rcv_snd, edge_values, num_target)
    return out.astype(np.float32)


def build_process_steps(learned) -> list[dict]:
    process_steps: list[dict] = []
    for level_idx, level in enumerate(learned.levels):
        for block_idx, block in enumerate(level):
            process_steps.append({
                "type": "block",
                "level": level_idx,
                "block": block_idx,
                "edge_keys": list(block.edge_keys),
            })
        if level_idx < len(learned.level_changes):
            lchange = learned.level_changes[level_idx]
            class_name = type(lchange).__name__
            if class_name == "DownSample":
                process_steps.append({
                    "type": "downsample",
                    "target_edge_keys": sorted({v["edge_key"] for v in lchange.target_edgesets.values()}),
                    "garment_nodes_label": lchange.garment_nodes_label,
                    "filter_edge_labels": ["world_direct", "world_inverse"],
                })
            elif class_name == "UpSample":
                process_steps.append({
                    "type": "upsample",
                    "filter_edge_labels": ["world_direct", "world_inverse"],
                })
            else:
                raise ValueError(f"Unsupported level change block: {class_name}")
    return process_steps


def resolve_garment_inputs(garment_template_path: str, aux_data_root: Path) -> tuple[str, str]:
    garment_path = Path(garment_template_path).resolve()
    try:
        garment_rel = garment_path.relative_to(aux_data_root)
    except ValueError as exc:
        raise ValueError(
            f"Garment path {garment_path} is not under aux_data root {aux_data_root}"
        ) from exc

    garment_dicts_dir = str(garment_rel.parent)
    garment_name = garment_rel.stem
    return garment_dicts_dir, garment_name


def build_sequence(
    sequence_path: str,
    garment_template_path: str,
    gender: str,
    n_coarse_levels: int,
    separate_arms: bool,
    fps: int,
    aux_data_root: Path,
):
    sequence_path = Path(sequence_path).resolve()
    garment_dicts_dir, garment_name = resolve_garment_inputs(garment_template_path, aux_data_root)
    data_root = str(sequence_path.parent)
    single_sequence_file = sequence_path.stem

    dataloader = create_one_sequence_dataloader(
        use_config="contourcraft",
        data_root=data_root,
        single_sequence_file=single_sequence_file,
        single_sequence_garment=garment_name,
        gender=gender,
        sequence_loader="cmu_npz_smpl",
        obstacle_dict_file="smpl_aux.pkl",
        garment_dicts_dir=garment_dicts_dir,
        wholeseq=True,
        pinned_verts=True,
        separate_arms=separate_arms,
        noise_scale=0,
        fps=fps,
        n_coarse_levels=n_coarse_levels,
    )
    return next(iter(dataloader))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one real HOOD-prepared frame for Android.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default="hood_final")
    parser.add_argument("--data-root", type=str, default="Data/ccraft_data")
    parser.add_argument("--project-dir", type=str, default="Models/ContourCraft")
    parser.add_argument("--config-dir", type=str, default="Models/ContourCraft/configs")
    parser.add_argument("--sequence-path", type=str, required=True)
    parser.add_argument("--garment-template-path", type=str, required=True)
    parser.add_argument("--gender", type=str, default="male")
    parser.add_argument("--frame-idx", type=int, default=0, help="Rollout step index. 0 is the first inference step.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--separate-arms", action="store_true", default=True)
    parser.add_argument("--out-dir", type=str, default="Android/HoodOnnxTest/app/src/main/assets/pipeline_real")
    args = parser.parse_args()

    defaults = DefaultsOverride(
        data_root=args.data_root,
        project_dir=str(Path(args.project_dir).resolve()),
        config_dir=str(Path(args.config_dir).resolve()),
    )

    hood = HoodMinimal(
        config_name=args.config,
        checkpoint_path=args.checkpoint,
        device="cpu",
        defaults=defaults,
    )
    runner = hood.runner
    model = hood.model
    learned = model._learned_model

    sequence = build_sequence(
        sequence_path=args.sequence_path,
        garment_template_path=args.garment_template_path,
        gender=args.gender,
        n_coarse_levels=learned._n_coarse_levels,
        separate_arms=args.separate_arms,
        fps=args.fps,
        aux_data_root=Path(defaults.data_root).resolve() / "aux_data",
    )

    sequence = runner.add_cloth_obj(sequence)
    sample_step = runner.collect_sample_wholeseq(sequence, args.frame_idx, prev_out_dict=None)
    prepared = model.prepare_inputs(copy_pyg_batch(sample_step))

    with torch.no_grad():
        decoded = learned(copy_pyg_batch(prepared))
        expected_output = decoded["cloth"].node_features.detach().cpu().numpy().astype(np.float32)
        positioned = model.get_position(copy_pyg_batch(decoded))
        expected_pred_pos = positioned["cloth"].pred_pos.detach().cpu().numpy().astype(np.float32)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    blocks_cfg = []
    for level_idx, level in enumerate(learned.levels):
        for block_idx, block in enumerate(level):
            blocks_cfg.append({
                "level": level_idx,
                "block": block_idx,
                "edge_keys": list(block.edge_keys),
            })
    process_steps = build_process_steps(learned)

    config = {
        "mode": "prepared_real_frame",
        "sequence_path": str(Path(args.sequence_path).resolve()),
        "garment_template_path": str(Path(args.garment_template_path).resolve()),
        "frame_idx": args.frame_idx,
        "gender": args.gender,
        "fps": args.fps,
        "N_cloth": int(prepared["cloth"].node_features.shape[0]),
        "N_obstacle": int(prepared["obstacle"].node_features.shape[0]),
        "latent_size": int(learned._latent_size),
        "output_size": int(learned._output_size),
        "collision_radius": float(model.collision_radius),
        "k_world_edges": None if model.k_world_edges is None else int(model.k_world_edges),
        "blocks": blocks_cfg,
        "process_steps": process_steps,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    cloth_raw = prepared["cloth"].node_features.detach().cpu().numpy().astype(np.float32)
    obstacle_raw = prepared["obstacle"].node_features.detach().cpu().numpy().astype(np.float32)
    obstacle_active_mask = prepared["obstacle"].active_mask.detach().cpu().numpy().astype(np.int32)

    mesh_raw = prepared["cloth", "mesh_edge", "cloth"].features.detach().cpu().numpy().astype(np.float32)
    coarse0_raw = prepared["cloth", "coarse_edge0", "cloth"].features.detach().cpu().numpy().astype(np.float32)
    coarse1_raw = prepared["cloth", "coarse_edge1", "cloth"].features.detach().cpu().numpy().astype(np.float32)
    coarse2_raw = prepared["cloth", "coarse_edge2", "cloth"].features.detach().cpu().numpy().astype(np.float32)

    # Match block semantics from models.core.base.make_edgesets_dict:
    # world_direct is obstacle -> cloth, world_inverse is cloth -> obstacle.
    world_direct_raw = prepared["obstacle", "world_edge", "cloth"].features.detach().cpu().numpy().astype(np.float32)
    world_inverse_raw = prepared["cloth", "world_edge", "obstacle"].features.detach().cpu().numpy().astype(np.float32)

    edge_index_mesh = prepared["cloth", "mesh_edge", "cloth"].edge_index.detach().cpu().numpy().astype(np.int32)
    edge_index_coarse0 = prepared["cloth", "coarse_edge0", "cloth"].edge_index.detach().cpu().numpy().astype(np.int32)
    edge_index_coarse1 = prepared["cloth", "coarse_edge1", "cloth"].edge_index.detach().cpu().numpy().astype(np.int32)
    edge_index_coarse2 = prepared["cloth", "coarse_edge2", "cloth"].edge_index.detach().cpu().numpy().astype(np.int32)
    edge_index_world_direct = prepared["obstacle", "world_edge", "cloth"].edge_index.detach().cpu().numpy().astype(np.int32)
    edge_index_world_inverse = prepared["cloth", "world_edge", "obstacle"].edge_index.detach().cpu().numpy().astype(np.int32)

    cloth_pos = sample_step["cloth"].pos.detach().cpu().numpy().astype(np.float32)
    cloth_prev_pos = sample_step["cloth"].prev_pos.detach().cpu().numpy().astype(np.float32)
    cloth_target_pos = sample_step["cloth"].target_pos.detach().cpu().numpy().astype(np.float32)
    cloth_rest_pos = sample_step["cloth"].rest_pos.detach().cpu().numpy().astype(np.float32)
    cloth_vertex_type = sample_step["cloth"].vertex_type.detach().cpu().numpy().astype(np.int32)
    cloth_vertex_level = sample_step["cloth"].vertex_level.detach().cpu().numpy().astype(np.int32)
    cloth_faces = sample_step["cloth"].faces_batch.T.detach().cpu().numpy().astype(np.int32)
    cloth_log_v_mass = torch.log(sample_step["cloth"].v_mass).detach().cpu().numpy().astype(np.float32)
    cloth_bending_coeff_input = sample_step["cloth"].bending_coeff_input.detach().cpu().numpy().astype(np.float32)
    cloth_lame_mu_input = sample_step["cloth"].lame_mu_input.detach().cpu().numpy().astype(np.float32)
    cloth_lame_lambda_input = sample_step["cloth"].lame_lambda_input.detach().cpu().numpy().astype(np.float32)
    timestep = sample_step["cloth"].timestep.detach().cpu().numpy().astype(np.float32)

    obstacle_pos = sample_step["obstacle"].pos.detach().cpu().numpy().astype(np.float32)
    obstacle_prev_pos = sample_step["obstacle"].prev_pos.detach().cpu().numpy().astype(np.float32)
    obstacle_target_pos = sample_step["obstacle"].target_pos.detach().cpu().numpy().astype(np.float32)
    obstacle_vertex_type = sample_step["obstacle"].vertex_type.detach().cpu().numpy().astype(np.int32)
    obstacle_vertex_level = sample_step["obstacle"].vertex_level.detach().cpu().numpy().astype(np.int32)
    obstacle_faces = sample_step["obstacle"].faces_batch.T.detach().cpu().numpy().astype(np.int32)

    node_type_embedding = model.nodetype_embedding.weight.detach().cpu().numpy().astype(np.float32)
    vertex_level_embedding = model.vertexlevel_embedding.weight.detach().cpu().numpy().astype(np.float32)
    node_norm_mean, node_norm_std = normalizer_stats(model._node_normalizer)
    mesh_norm_mean, mesh_norm_std = normalizer_stats(model._mesh_edge_normalizer)
    world_norm_mean, world_norm_std = normalizer_stats(model._world_edge_normalizer)

    n_cloth = cloth_raw.shape[0]
    n_obstacle = obstacle_raw.shape[0]
    latent = learned._latent_size

    obstacle_mask_bool = prepared["obstacle"].active_mask[:, 0].detach().cpu().bool()
    cloth_features_t = prepared["cloth"].node_features.detach().cpu()
    obstacle_features_t = prepared["obstacle"].node_features.detach().cpu()
    obstacle_active_features_t = obstacle_features_t[obstacle_mask_bool]
    combined_features_t = torch.cat([cloth_features_t, obstacle_active_features_t], dim=0)

    with torch.no_grad():
        combined_latents_t = learned.node_encoder(combined_features_t)
    cloth_latents_t = combined_latents_t[:n_cloth]
    obstacle_active_latents_t = combined_latents_t[n_cloth:]
    obstacle_latents_t = torch.zeros(n_obstacle, latent)
    obstacle_latents_t[obstacle_mask_bool] = obstacle_active_latents_t

    write_float_bin(out_dir / "expected_node_encoder_cloth.bin", cloth_latents_t.detach().cpu().numpy())
    write_shape(out_dir / "expected_node_encoder_cloth_shape.json", cloth_latents_t.shape)
    write_float_bin(out_dir / "expected_node_encoder_obstacle.bin", obstacle_latents_t.detach().cpu().numpy())
    write_shape(out_dir / "expected_node_encoder_obstacle_shape.json", obstacle_latents_t.shape)

    with torch.no_grad():
        mesh_latents_t = learned.edgeset_encoders["mesh"](torch.from_numpy(mesh_raw))
        coarse0_latents_t = learned.edgeset_encoders["coarse0"](torch.from_numpy(coarse0_raw))
        coarse1_latents_t = learned.edgeset_encoders["coarse1"](torch.from_numpy(coarse1_raw))
        coarse2_latents_t = learned.edgeset_encoders["coarse2"](torch.from_numpy(coarse2_raw))
        world_cat_t = torch.cat([torch.from_numpy(world_direct_raw), torch.from_numpy(world_inverse_raw)], dim=0)
        world_cat_latents_t = learned.edgeset_encoders["world"](world_cat_t)
        world_direct_latents_t = world_cat_latents_t[:world_direct_raw.shape[0]]
        world_inverse_latents_t = world_cat_latents_t[world_direct_raw.shape[0]:]

    write_float_bin(out_dir / "expected_edge_encoder_mesh.bin", mesh_latents_t.detach().cpu().numpy())
    write_shape(out_dir / "expected_edge_encoder_mesh_shape.json", mesh_latents_t.shape)
    write_float_bin(out_dir / "expected_edge_encoder_coarse0.bin", coarse0_latents_t.detach().cpu().numpy())
    write_shape(out_dir / "expected_edge_encoder_coarse0_shape.json", coarse0_latents_t.shape)
    write_float_bin(out_dir / "expected_edge_encoder_coarse1.bin", coarse1_latents_t.detach().cpu().numpy())
    write_shape(out_dir / "expected_edge_encoder_coarse1_shape.json", coarse1_latents_t.shape)
    write_float_bin(out_dir / "expected_edge_encoder_coarse2.bin", coarse2_latents_t.detach().cpu().numpy())
    write_shape(out_dir / "expected_edge_encoder_coarse2_shape.json", coarse2_latents_t.shape)
    write_float_bin(out_dir / "expected_edge_encoder_world_direct.bin", world_direct_latents_t.detach().cpu().numpy())
    write_shape(out_dir / "expected_edge_encoder_world_direct_shape.json", world_direct_latents_t.shape)
    write_float_bin(out_dir / "expected_edge_encoder_world_inverse.bin", world_inverse_latents_t.detach().cpu().numpy())
    write_shape(out_dir / "expected_edge_encoder_world_inverse_shape.json", world_inverse_latents_t.shape)

    nodes = {
        "cloth": cloth_latents_t.detach().cpu(),
        "obstacle": obstacle_latents_t.detach().cpu(),
    }
    edges = {
        "mesh_edge": mesh_latents_t.detach().cpu(),
        "coarse_edge0": coarse0_latents_t.detach().cpu(),
        "coarse_edge1": coarse1_latents_t.detach().cpu(),
        "coarse_edge2": coarse2_latents_t.detach().cpu(),
        "world_direct": world_direct_latents_t.detach().cpu(),
        "world_inverse": world_inverse_latents_t.detach().cpu(),
    }
    edge_indices = {
        "mesh_edge": edge_index_mesh,
        "coarse_edge0": edge_index_coarse0,
        "coarse_edge1": edge_index_coarse1,
        "coarse_edge2": edge_index_coarse2,
        "world_direct": edge_index_world_direct,
        "world_inverse": edge_index_world_inverse,
    }
    stash_stack: list[dict] = []
    block_dir = out_dir / "blocks"
    block_dir.mkdir(parents=True, exist_ok=True)

    def get_remaining_node_mask(target_edge_keys: list[str]) -> np.ndarray:
        mask = np.zeros((n_cloth,), dtype=bool)
        for key in target_edge_keys:
            if key not in edge_indices:
                continue
            edge_index = edge_indices[key]
            mask[edge_index.reshape(-1)] = True
        return mask

    def filter_world_edges(edge_label: str, nodes_mask: np.ndarray):
        edge_index = edge_indices[edge_label]
        features = edges[edge_label].numpy()
        src = edge_index[0]
        tgt = edge_index[1]
        if edge_label == "world_direct":
            mask = nodes_mask[tgt]
        else:
            mask = nodes_mask[src]
        stashed = {
            "old_edge_index": edge_index.copy(),
            "old_features": features.copy(),
            "mask": mask.copy(),
        }
        edge_indices[edge_label] = edge_index[:, mask]
        edges[edge_label] = torch.from_numpy(features[mask].copy())
        return stashed

    def restore_world_edges(edge_label: str, stashed: dict):
        restored = stashed["old_features"].copy()
        restored[stashed["mask"]] = edges[edge_label].numpy()
        edge_indices[edge_label] = stashed["old_edge_index"]
        edges[edge_label] = torch.from_numpy(restored)

    for step in process_steps:
        step_type = step["type"]
        if step_type == "downsample":
            remaining_nodes_mask = get_remaining_node_mask(step["target_edge_keys"])
            stash_stack.append({
                "world_direct": filter_world_edges("world_direct", remaining_nodes_mask),
                "world_inverse": filter_world_edges("world_inverse", remaining_nodes_mask),
            })
            continue

        if step_type == "upsample":
            stashed = stash_stack.pop()
            restore_world_edges("world_direct", stashed["world_direct"])
            restore_world_edges("world_inverse", stashed["world_inverse"])
            continue

        level_idx = step["level"]
        block_idx = step["block"]
        edge_keys = list(step["edge_keys"])
        block = learned.levels[level_idx][block_idx]
        edge_mlps = block.edge_processor_dict
        node_mlp = block.node_processor_dict["node"]

        def torch_edge_update(edge_key: str, src_key: str, tgt_key: str) -> torch.Tensor:
            src_idx = torch.from_numpy(edge_indices[edge_key][0]).long()
            tgt_idx = torch.from_numpy(edge_indices[edge_key][1]).long()
            src = nodes[src_key][src_idx]
            tgt = nodes[tgt_key][tgt_idx]
            feat = edges[edge_key]
            mlp_key = "world_edge" if "world" in edge_key else edge_key
            return edge_mlps[mlp_key](edge_mlp_input(tgt, src, feat))

        with torch.no_grad():
            updated_world_direct = torch_edge_update("world_direct", "obstacle", "cloth")
            updated_world_inverse = torch_edge_update("world_inverse", "cloth", "obstacle")
            updated_mesh = torch_edge_update("mesh_edge", "cloth", "cloth") if "mesh_edge" in edge_keys else None
            updated_coarse0 = torch_edge_update("coarse_edge0", "cloth", "cloth") if "coarse_edge0" in edge_keys else None
            updated_coarse1 = torch_edge_update("coarse_edge1", "cloth", "cloth") if "coarse_edge1" in edge_keys else None
            updated_coarse2 = torch_edge_update("coarse_edge2", "cloth", "cloth") if "coarse_edge2" in edge_keys else None

        agg_world_cloth = aggregate(edge_indices["world_direct"], updated_world_direct.detach().cpu().numpy(), n_cloth)
        agg_world_obs = aggregate(edge_indices["world_inverse"], updated_world_inverse.detach().cpu().numpy(), n_obstacle)

        agg_map_cloth = {"world_edge": agg_world_cloth}
        agg_map_obs = {"world_edge": agg_world_obs}
        if updated_mesh is not None:
            agg_map_cloth["mesh_edge"] = aggregate(edge_indices["mesh_edge"], updated_mesh.detach().cpu().numpy(), n_cloth)
        if updated_coarse0 is not None:
            agg_map_cloth["coarse_edge0"] = aggregate(edge_indices["coarse_edge0"], updated_coarse0.detach().cpu().numpy(), n_cloth)
        if updated_coarse1 is not None:
            agg_map_cloth["coarse_edge1"] = aggregate(edge_indices["coarse_edge1"], updated_coarse1.detach().cpu().numpy(), n_cloth)
        if updated_coarse2 is not None:
            agg_map_cloth["coarse_edge2"] = aggregate(edge_indices["coarse_edge2"], updated_coarse2.detach().cpu().numpy(), n_cloth)

        node_in_cloth_parts = [nodes["cloth"].numpy()]
        node_in_obs_parts = [nodes["obstacle"].numpy()]
        for key in edge_keys:
            node_in_cloth_parts.append(agg_map_cloth.get(key, np.zeros((n_cloth, latent), dtype=np.float32)))
            if key == "world_edge":
                node_in_obs_parts.append(agg_map_obs.get(key, np.zeros((n_obstacle, latent), dtype=np.float32)))
            else:
                node_in_obs_parts.append(np.zeros((n_obstacle, latent), dtype=np.float32))

        node_in_cloth = np.concatenate(node_in_cloth_parts, axis=1).astype(np.float32)
        node_in_obs = np.concatenate(node_in_obs_parts, axis=1).astype(np.float32)

        with torch.no_grad():
            updated_nodes_cloth = node_mlp(torch.from_numpy(node_in_cloth))
            updated_nodes_obs = node_mlp(torch.from_numpy(node_in_obs))

        edges["world_direct"] = edges["world_direct"] + updated_world_direct.detach().cpu()
        edges["world_inverse"] = edges["world_inverse"] + updated_world_inverse.detach().cpu()
        if updated_mesh is not None:
            edges["mesh_edge"] = edges["mesh_edge"] + updated_mesh.detach().cpu()
        if updated_coarse0 is not None:
            edges["coarse_edge0"] = edges["coarse_edge0"] + updated_coarse0.detach().cpu()
        if updated_coarse1 is not None:
            edges["coarse_edge1"] = edges["coarse_edge1"] + updated_coarse1.detach().cpu()
        if updated_coarse2 is not None:
            edges["coarse_edge2"] = edges["coarse_edge2"] + updated_coarse2.detach().cpu()

        if level_idx == 0 and block_idx == 0:
            write_float_bin(block_dir / "block_0_0_agg_world_cloth.bin", agg_world_cloth)
            write_float_bin(block_dir / "block_0_0_agg_mesh.bin", agg_map_cloth.get("mesh_edge", np.zeros_like(agg_world_cloth)))
            write_float_bin(block_dir / "block_0_0_agg_coarse0.bin", agg_map_cloth.get("coarse_edge0", np.zeros_like(agg_world_cloth)))
            write_float_bin(block_dir / "block_0_0_node_in_cloth.bin", node_in_cloth)
            write_shape(block_dir / "block_0_0_node_in_cloth_shape.json", node_in_cloth.shape)
            write_float_bin(block_dir / "block_0_0_node_out_cloth.bin", updated_nodes_cloth.detach().cpu().numpy())
            write_shape(block_dir / "block_0_0_node_out_cloth_shape.json", updated_nodes_cloth.shape)
            write_float_bin(block_dir / "block_0_0_updated_world_direct.bin", updated_world_direct.detach().cpu().numpy())
            write_float_bin(block_dir / "block_0_0_updated_world_inverse.bin", updated_world_inverse.detach().cpu().numpy())
            if updated_mesh is not None:
                write_float_bin(block_dir / "block_0_0_updated_mesh.bin", updated_mesh.detach().cpu().numpy())
            if updated_coarse0 is not None:
                write_float_bin(block_dir / "block_0_0_updated_coarse0.bin", updated_coarse0.detach().cpu().numpy())

        nodes["cloth"] = nodes["cloth"] + updated_nodes_cloth.detach().cpu()
        nodes["obstacle"] = nodes["obstacle"] + updated_nodes_obs.detach().cpu()
        write_float_bin(block_dir / f"block_{level_idx}_{block_idx}_cloth_nodes.bin", nodes["cloth"].numpy())
        write_shape(block_dir / f"block_{level_idx}_{block_idx}_cloth_nodes_shape.json", nodes["cloth"].shape)

    write_float_bin(out_dir / "cloth_raw.bin", cloth_raw)
    write_shape(out_dir / "cloth_raw_shape.json", cloth_raw.shape)
    write_float_bin(out_dir / "obstacle_raw.bin", obstacle_raw)
    write_shape(out_dir / "obstacle_raw_shape.json", obstacle_raw.shape)
    write_int_bin(out_dir / "obstacle_active_mask.bin", obstacle_active_mask)
    write_shape(out_dir / "obstacle_active_mask_shape.json", obstacle_active_mask.shape)

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

    write_float_bin(out_dir / "cloth_pos.bin", cloth_pos)
    write_shape(out_dir / "cloth_pos_shape.json", cloth_pos.shape)
    write_float_bin(out_dir / "cloth_prev_pos.bin", cloth_prev_pos)
    write_shape(out_dir / "cloth_prev_pos_shape.json", cloth_prev_pos.shape)
    write_float_bin(out_dir / "cloth_target_pos.bin", cloth_target_pos)
    write_shape(out_dir / "cloth_target_pos_shape.json", cloth_target_pos.shape)
    write_float_bin(out_dir / "cloth_rest_pos.bin", cloth_rest_pos)
    write_shape(out_dir / "cloth_rest_pos_shape.json", cloth_rest_pos.shape)
    write_int_bin(out_dir / "cloth_vertex_type.bin", cloth_vertex_type)
    write_shape(out_dir / "cloth_vertex_type_shape.json", cloth_vertex_type.shape)
    write_int_bin(out_dir / "cloth_vertex_level.bin", cloth_vertex_level)
    write_shape(out_dir / "cloth_vertex_level_shape.json", cloth_vertex_level.shape)
    write_int_bin(out_dir / "cloth_faces.bin", cloth_faces)
    write_shape(out_dir / "cloth_faces_shape.json", cloth_faces.shape)
    write_float_bin(out_dir / "cloth_log_v_mass.bin", cloth_log_v_mass)
    write_shape(out_dir / "cloth_log_v_mass_shape.json", cloth_log_v_mass.shape)
    write_float_bin(out_dir / "cloth_bending_coeff_input.bin", cloth_bending_coeff_input)
    write_shape(out_dir / "cloth_bending_coeff_input_shape.json", cloth_bending_coeff_input.shape)
    write_float_bin(out_dir / "cloth_lame_mu_input.bin", cloth_lame_mu_input)
    write_shape(out_dir / "cloth_lame_mu_input_shape.json", cloth_lame_mu_input.shape)
    write_float_bin(out_dir / "cloth_lame_lambda_input.bin", cloth_lame_lambda_input)
    write_shape(out_dir / "cloth_lame_lambda_input_shape.json", cloth_lame_lambda_input.shape)
    write_float_bin(out_dir / "timestep.bin", timestep)
    write_shape(out_dir / "timestep_shape.json", timestep.shape)

    write_float_bin(out_dir / "obstacle_pos.bin", obstacle_pos)
    write_shape(out_dir / "obstacle_pos_shape.json", obstacle_pos.shape)
    write_float_bin(out_dir / "obstacle_prev_pos.bin", obstacle_prev_pos)
    write_shape(out_dir / "obstacle_prev_pos_shape.json", obstacle_prev_pos.shape)
    write_float_bin(out_dir / "obstacle_target_pos.bin", obstacle_target_pos)
    write_shape(out_dir / "obstacle_target_pos_shape.json", obstacle_target_pos.shape)
    write_int_bin(out_dir / "obstacle_vertex_type.bin", obstacle_vertex_type)
    write_shape(out_dir / "obstacle_vertex_type_shape.json", obstacle_vertex_type.shape)
    write_int_bin(out_dir / "obstacle_vertex_level.bin", obstacle_vertex_level)
    write_shape(out_dir / "obstacle_vertex_level_shape.json", obstacle_vertex_level.shape)
    write_int_bin(out_dir / "obstacle_faces.bin", obstacle_faces)
    write_shape(out_dir / "obstacle_faces_shape.json", obstacle_faces.shape)

    write_float_bin(out_dir / "node_type_embedding.bin", node_type_embedding)
    write_shape(out_dir / "node_type_embedding_shape.json", node_type_embedding.shape)
    write_float_bin(out_dir / "vertex_level_embedding.bin", vertex_level_embedding)
    write_shape(out_dir / "vertex_level_embedding_shape.json", vertex_level_embedding.shape)
    write_float_bin(out_dir / "node_norm_mean.bin", node_norm_mean)
    write_shape(out_dir / "node_norm_mean_shape.json", node_norm_mean.shape)
    write_float_bin(out_dir / "node_norm_std.bin", node_norm_std)
    write_shape(out_dir / "node_norm_std_shape.json", node_norm_std.shape)
    write_float_bin(out_dir / "mesh_norm_mean.bin", mesh_norm_mean)
    write_shape(out_dir / "mesh_norm_mean_shape.json", mesh_norm_mean.shape)
    write_float_bin(out_dir / "mesh_norm_std.bin", mesh_norm_std)
    write_shape(out_dir / "mesh_norm_std_shape.json", mesh_norm_std.shape)
    write_float_bin(out_dir / "world_norm_mean.bin", world_norm_mean)
    write_shape(out_dir / "world_norm_mean_shape.json", world_norm_mean.shape)
    write_float_bin(out_dir / "world_norm_std.bin", world_norm_std)
    write_shape(out_dir / "world_norm_std_shape.json", world_norm_std.shape)

    write_float_bin(out_dir / "expected_output.bin", expected_output)
    write_shape(out_dir / "expected_output_shape.json", expected_output.shape)
    write_float_bin(out_dir / "expected_pred_pos.bin", expected_pred_pos)
    write_shape(out_dir / "expected_pred_pos_shape.json", expected_pred_pos.shape)

    print(f"Wrote real prepared frame assets to {out_dir}")
    print(f"cloth nodes={cloth_raw.shape[0]} obstacle nodes={obstacle_raw.shape[0]}")
    print(
        "edge counts "
        f"mesh={mesh_raw.shape[0]} coarse0={coarse0_raw.shape[0]} coarse1={coarse1_raw.shape[0]} "
        f"coarse2={coarse2_raw.shape[0]} world_direct={world_direct_raw.shape[0]} world_inverse={world_inverse_raw.shape[0]}"
    )


if __name__ == "__main__":
    main()
