import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def read_shape(path: Path) -> list[int]:
    return [int(x) for x in json.loads(path.read_text())]


def load_initializers(model_path: Path) -> dict[str, np.ndarray]:
    model = onnx.load(str(model_path), load_external_data=True)
    return {init.name: numpy_helper.to_array(init).astype(np.float32) for init in model.graph.initializer}


def build_layernorm_probe(gamma: np.ndarray, beta: np.ndarray, rows: int, eps: float) -> onnx.ModelProto:
    hidden = int(gamma.shape[0])

    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["input", "axes"], ["mean"], name="ln_mean", keepdims=1),
            helper.make_node("Sub", ["input", "mean"], ["centered"], name="ln_centered"),
            helper.make_node("Mul", ["centered", "centered"], ["sq"], name="ln_square"),
            helper.make_node("ReduceMean", ["sq", "axes"], ["var"], name="ln_var", keepdims=1),
            helper.make_node("Add", ["var", "epsilon"], ["var_eps"], name="ln_add_eps"),
            helper.make_node("Sqrt", ["var_eps"], ["stddev"], name="ln_sqrt"),
            helper.make_node("Div", ["centered", "stddev"], ["norm"], name="ln_div"),
            helper.make_node("Mul", ["norm", "gamma"], ["scaled"], name="ln_scale"),
            helper.make_node("Add", ["scaled", "beta"], ["output"], name="ln_bias"),
        ],
        "layernorm_probe",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [rows, hidden])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [rows, hidden])],
        initializer=[
            numpy_helper.from_array(np.array([1], dtype=np.int64), "axes"),
            numpy_helper.from_array(np.array([eps], dtype=np.float32), "epsilon"),
            numpy_helper.from_array(np.ascontiguousarray(gamma.reshape(1, hidden)), "gamma"),
            numpy_helper.from_array(np.ascontiguousarray(beta.reshape(1, hidden)), "beta"),
        ],
    )
    model = helper.make_model(graph, producer_name="magcode_qairt_layernorm_probe")
    model.ir_version = 8
    del model.opset_import[:]
    opset = model.opset_import.add()
    opset.domain = ""
    opset.version = 13
    return model


def run_layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    centered = x - mean
    var = (centered * centered).mean(axis=1, keepdims=True)
    norm = centered / np.sqrt(var + eps)
    y = norm * gamma.reshape(1, -1) + beta.reshape(1, -1)
    return y.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a standalone real-weight LayerNorm probe for QAIRT.")
    parser.add_argument("--model", required=True, help="Path to node_encoder.onnx")
    parser.add_argument("--rows", type=int, required=True, help="Static row count for the probe model")
    parser.add_argument("--out-model", required=True, help="Output ONNX path")
    parser.add_argument("--eps", type=float, default=1e-5, help="LayerNorm epsilon")
    parser.add_argument("--input-raw", help="Optional float32 raw input tensor for expected-output generation")
    parser.add_argument("--input-shape", help="Optional JSON shape file for --input-raw")
    parser.add_argument("--out-expected", help="Optional output raw path for expected probe output")
    parser.add_argument("--out-expected-shape", help="Optional JSON output shape path for expected probe output")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    out_model = Path(args.out_model).resolve()
    out_model.parent.mkdir(parents=True, exist_ok=True)

    initializers = load_initializers(model_path)
    gamma = initializers["1.weight"].reshape(-1)
    beta = initializers["1.bias"].reshape(-1)
    probe_model = build_layernorm_probe(gamma, beta, args.rows, args.eps)
    onnx.save(probe_model, str(out_model))
    print(f"wrote layernorm probe model to {out_model}")

    if args.input_raw or args.input_shape or args.out_expected or args.out_expected_shape:
        if not all([args.input_raw, args.input_shape, args.out_expected, args.out_expected_shape]):
            raise ValueError("Expected all of --input-raw, --input-shape, --out-expected, --out-expected-shape together")
        input_shape = read_shape(Path(args.input_shape).resolve())
        x = np.fromfile(Path(args.input_raw).resolve(), dtype=np.float32).reshape(input_shape)
        y = run_layernorm(x, gamma, beta, args.eps)
        Path(args.out_expected).resolve().write_bytes(y.astype(np.float32).tobytes())
        Path(args.out_expected_shape).resolve().write_text(json.dumps([int(v) for v in y.shape]))
        print(f"wrote expected layernorm probe output to {Path(args.out_expected).resolve()}")
        print(f"expected output shape={list(y.shape)}")


if __name__ == "__main__":
    main()
