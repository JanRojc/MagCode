import argparse
import json
from pathlib import Path

import onnx
from onnx import numpy_helper


def write_float_bin(path: Path, array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(array.astype("float32").tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Export node_encoder weights for the Android GLES compute prototype.")
    parser.add_argument("--model", required=True, help="Path to node_encoder.onnx")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    model_path = Path(args.model)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(model_path), load_external_data=True)
    initializers = {init.name: numpy_helper.to_array(init).astype("float32") for init in model.graph.initializer}

    required = {
        "linear0_weight": initializers["0.layers.0.weight"],
        "linear0_bias": initializers["0.layers.0.bias"],
        "linear1_weight": initializers["0.layers.2.weight"],
        "linear1_bias": initializers["0.layers.2.bias"],
        "linear2_weight": initializers["0.layers.4.weight"],
        "linear2_bias": initializers["0.layers.4.bias"],
        "layernorm_gamma": initializers["1.weight"],
        "layernorm_beta": initializers["1.bias"],
    }

    for name, array in required.items():
        write_float_bin(out_dir / f"{name}.bin", array.reshape(-1))

    manifest = {
        "input_dim": int(required["linear0_weight"].shape[1]),
        "hidden_dim": int(required["linear0_weight"].shape[0]),
        "output_dim": int(required["linear2_weight"].shape[0]),
        "epsilon": 1e-6,
        "source_model": str(model_path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Exported GPU node encoder assets to {out_dir}")


if __name__ == "__main__":
    main()
