import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import torch

from hood_minimal import HoodMinimal, DefaultsOverride


class LayerNormExplicit(torch.nn.Module):
    """
    LayerNorm implemented with primitive ops to avoid ONNX LayerNormalization op.
    """

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        if elementwise_affine:
            self.weight = torch.nn.Parameter(torch.ones(*self.normalized_shape))
            self.bias = torch.nn.Parameter(torch.zeros(*self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize over the last dimension(s)
        dims = tuple(range(-len(self.normalized_shape), 0))
        mean = x.mean(dim=dims, keepdim=True)
        var = (x - mean).pow(2).mean(dim=dims, keepdim=True)
        y = (x - mean) / torch.sqrt(var + self.eps)
        if self.weight is not None:
            y = y * self.weight
        if self.bias is not None:
            y = y + self.bias
        return y


def replace_layernorm(module: torch.nn.Module) -> torch.nn.Module:
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.LayerNorm):
            ln = LayerNormExplicit(child.normalized_shape, eps=child.eps, elementwise_affine=child.elementwise_affine)
            # copy weights
            if child.elementwise_affine:
                ln.weight.data.copy_(child.weight.data)
                ln.bias.data.copy_(child.bias.data)
            setattr(module, name, ln)
        else:
            replace_layernorm(child)
    return module


def _export_module(
    module: torch.nn.Module,
    dummy_input: torch.Tensor,
    out_path: Path,
    opset: int,
    dynamic_batch: bool = False,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    module.eval()
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch"},
            "output": {0: "batch"},
        }
    torch.onnx.export(
        module,
        dummy_input,
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )


def _get_model_and_learned(config: str, checkpoint: str, device: str, defaults: DefaultsOverride, rewrite_layernorm: bool):
    hood = HoodMinimal(
        config_name=config,
        checkpoint_path=checkpoint,
        device=device,
        defaults=defaults,
    )
    model = hood.model
    learned = model._learned_model
    if rewrite_layernorm:
        replace_layernorm(learned)
    return hood, model, learned


def _dump_metadata(out_dir: Path, model, learned) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "n_nodefeatures": learned.n_nodefeatures,
        "n_edgefeatures_mesh": learned.n_edgefeatures_mesh,
        "n_edgefeatures_world": learned.n_edgefeatures_world,
        "n_edgefeatures_coarse": learned.n_edgefeatures_coarse,
        "latent_size": learned._latent_size,
        "output_size": learned._output_size,
        "num_layers": learned._num_layers,
        "message_passing_steps": learned._message_passing_steps,
        "n_coarse_levels": learned._n_coarse_levels,
        "architecture": learned.architecture_string,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def export_mlps(
    model,
    learned,
    out_dir: Path,
    opset: int,
    dynamic_batch: bool = False,
) -> Dict[str, str]:
    exported = {}

    # Node encoder / decoder
    dummy_nodes = torch.randn(8, learned.n_nodefeatures)
    _export_module(learned.node_encoder, dummy_nodes, out_dir / "node_encoder.onnx", opset, dynamic_batch=dynamic_batch)
    exported["node_encoder"] = "node_encoder.onnx"

    dummy_latents = torch.randn(8, learned._latent_size)
    _export_module(learned.decoder, dummy_latents, out_dir / "node_decoder.onnx", opset, dynamic_batch=dynamic_batch)
    exported["node_decoder"] = "node_decoder.onnx"

    # Edge encoders
    dummy_mesh = torch.randn(16, learned.n_edgefeatures_mesh)
    _export_module(learned.edgeset_encoders["mesh"], dummy_mesh, out_dir / "edge_encoder_mesh.onnx", opset, dynamic_batch=dynamic_batch)
    exported["edge_encoder_mesh"] = "edge_encoder_mesh.onnx"

    dummy_world = torch.randn(16, learned.n_edgefeatures_world)
    _export_module(learned.edgeset_encoders["world"], dummy_world, out_dir / "edge_encoder_world.onnx", opset, dynamic_batch=dynamic_batch)
    exported["edge_encoder_world"] = "edge_encoder_world.onnx"

    for i in range(learned._n_coarse_levels):
        dummy_coarse = torch.randn(16, learned.n_edgefeatures_coarse)
        name = f"edge_encoder_coarse{i}.onnx"
        _export_module(learned.edgeset_encoders[f"coarse{i}"], dummy_coarse, out_dir / name, opset, dynamic_batch=dynamic_batch)
        exported[f"edge_encoder_coarse{i}"] = name

    # Per-block edge and node processors
    for level_idx, level in enumerate(learned.levels):
        for block_idx, block in enumerate(level):
            # Edge processors per edgeset in this block
            for edge_key, edge_mlp in block.edge_processor_dict.items():
                dummy_edge = torch.randn(8, learned._latent_size * 3)
                name = f"block_{level_idx}_{block_idx}_edge_{edge_key}.onnx"
                _export_module(edge_mlp, dummy_edge, out_dir / "blocks" / name, opset, dynamic_batch=dynamic_batch)
                exported[f"block_{level_idx}_{block_idx}_edge_{edge_key}"] = f"blocks/{name}"

            # Node processor (single)
            # node processor input size depends on number of edge sets used in the block
            num_edgesets = len(block.edge_keys)
            dummy_node = torch.randn(8, learned._latent_size * (1 + num_edgesets))
            node_mlp = block.node_processor_dict["node"]
            name = f"block_{level_idx}_{block_idx}_node.onnx"
            _export_module(node_mlp, dummy_node, out_dir / "blocks" / name, opset, dynamic_batch=dynamic_batch)
            exported[f"block_{level_idx}_{block_idx}_node"] = f"blocks/{name}"

    return exported


def try_full_export(hood, out_dir: Path, opset: int) -> Tuple[bool, str]:
    """
    Attempt to export the full forward pass.
    This is expected to fail due to torch_geometric/HeteroData usage,
    but we capture the error to document operator blockers.
    """
    try:
        sample = hood_minimal_sample(hood.device)
        out_path = out_dir / "hood_full.onnx"
        torch.onnx.export(
            hood.model,
            sample,
            str(out_path),
            input_names=["sample"],
            output_names=["pred"],
            opset_version=opset,
            do_constant_folding=True,
        )
        return True, ""
    except Exception as exc:  # pragma: no cover - export is expected to fail
        return False, str(exc)


def hood_minimal_sample(device: str):
    from hood_minimal import build_dummy_sample

    return build_dummy_sample(device=device)


def validate_onnx(out_path: Path, torch_module: torch.nn.Module, dummy_input: torch.Tensor) -> str:
    try:
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover - optional dependency
        return f"onnxruntime not available: {exc}"

    torch_module.eval()
    with torch.no_grad():
        torch_out = torch_module(dummy_input).cpu().numpy()

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"input": dummy_input.cpu().numpy()})[0]

    diff = abs(torch_out - ort_out).max()
    return f"max_abs_diff={diff}"


def main():
    parser = argparse.ArgumentParser(description="Export HOOD MLPs and attempt full ONNX export.")
    parser.add_argument("--config", default="hood_final")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opset", type=int, default=15)
    parser.add_argument("--out", default="onnx_out")
    parser.add_argument("--export-full", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--rewrite-layernorm", action="store_true", help="Replace LayerNorm with primitive ops before export")
    parser.add_argument("--dynamic-batch", action="store_true", help="Export all MLPs with dynamic batch axis")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--aux-data", default=None)
    parser.add_argument("--project-dir", default=None)
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--cmu-root", default=None)
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    defaults = DefaultsOverride(
        data_root=args.data_root,
        aux_data=args.aux_data,
        project_dir=args.project_dir,
        config_dir=args.config_dir,
        cmu_root=args.cmu_root,
        results_dir=args.results_dir,
    )

    out_dir = Path(args.out)
    hood, model, learned = _get_model_and_learned(args.config, args.checkpoint, args.device, defaults, args.rewrite_layernorm)

    _dump_metadata(out_dir, model, learned)
    exported = export_mlps(model, learned, out_dir, args.opset, dynamic_batch=args.dynamic_batch)
    (out_dir / "exports.json").write_text(json.dumps(exported, indent=2))

    if args.validate:
        results = {}
        dummy_nodes = torch.randn(8, learned.n_nodefeatures)
        results["node_encoder"] = validate_onnx(out_dir / "node_encoder.onnx", learned.node_encoder, dummy_nodes)
        (out_dir / "validation.json").write_text(json.dumps(results, indent=2))

    if args.export_full:
        ok, err = try_full_export(hood, out_dir, args.opset)
        if not ok:
            (out_dir / "full_export_error.txt").write_text(err)


if __name__ == "__main__":
    main()
