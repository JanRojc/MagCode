import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def read_shape(path: Path) -> list[int]:
    return [int(x) for x in json.loads(path.read_text())]


def load_gamma_beta(model_path: Path) -> tuple[np.ndarray, np.ndarray]:
    model = onnx.load(str(model_path), load_external_data=True)
    initializers = {init.name: numpy_helper.to_array(init).astype(np.float32) for init in model.graph.initializer}
    gamma = initializers["1.weight"].reshape(-1).astype(np.float32)
    beta = initializers["1.bias"].reshape(-1).astype(np.float32)
    return gamma, beta


def apply_layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    centered = x - mean
    var = (centered * centered).mean(axis=1, keepdims=True)
    norm = centered / np.sqrt(var + eps)
    return (norm * gamma.reshape(1, -1) + beta.reshape(1, -1)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the real node_encoder LayerNorm to a float32 raw tensor.")
    parser.add_argument("--model", required=True, help="Path to node_encoder.onnx")
    parser.add_argument("--input-raw", required=True, help="Input float32 raw tensor path")
    parser.add_argument("--input-shape", required=True, help="Input tensor shape JSON path")
    parser.add_argument("--output-raw", required=True, help="Output float32 raw tensor path")
    parser.add_argument("--output-shape", required=True, help="Output tensor shape JSON path")
    parser.add_argument("--eps", type=float, default=1e-5, help="LayerNorm epsilon")
    args = parser.parse_args()

    gamma, beta = load_gamma_beta(Path(args.model).resolve())
    shape = read_shape(Path(args.input_shape).resolve())
    x = np.fromfile(Path(args.input_raw).resolve(), dtype=np.float32).reshape(shape)
    y = apply_layernorm(x, gamma, beta, args.eps)

    output_raw = Path(args.output_raw).resolve()
    output_shape = Path(args.output_shape).resolve()
    output_raw.write_bytes(y.tobytes())
    output_shape.write_text(json.dumps([int(v) for v in y.shape]))
    print(f"wrote layernorm output to {output_raw}")
    print(f"output shape={list(y.shape)}")


if __name__ == "__main__":
    main()
