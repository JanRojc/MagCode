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


def build_mlp_body_model(initializers: dict[str, np.ndarray], rows: int) -> onnx.ModelProto:
    w0 = initializers["0.layers.0.weight"]
    b0 = initializers["0.layers.0.bias"]
    w1 = initializers["0.layers.2.weight"]
    b1 = initializers["0.layers.2.bias"]
    w2 = initializers["0.layers.4.weight"]
    b2 = initializers["0.layers.4.bias"]

    input_dim = int(w0.shape[1])
    output_dim = int(w2.shape[0])

    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["input", "w0"], ["mm0"], name="linear0_matmul"),
            helper.make_node("Add", ["mm0", "b0"], ["lin0"], name="linear0_add"),
            helper.make_node("Relu", ["lin0"], ["relu0"], name="relu0"),
            helper.make_node("MatMul", ["relu0", "w1"], ["mm1"], name="linear1_matmul"),
            helper.make_node("Add", ["mm1", "b1"], ["lin1"], name="linear1_add"),
            helper.make_node("Relu", ["lin1"], ["relu1"], name="relu1"),
            helper.make_node("MatMul", ["relu1", "w2"], ["mm2"], name="linear2_matmul"),
            helper.make_node("Add", ["mm2", "b2"], ["output"], name="linear2_add"),
        ],
        "mlp_body",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [rows, input_dim])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [rows, output_dim])],
        initializer=[
            numpy_helper.from_array(np.ascontiguousarray(w0.T), "w0"),
            numpy_helper.from_array(np.ascontiguousarray(b0), "b0"),
            numpy_helper.from_array(np.ascontiguousarray(w1.T), "w1"),
            numpy_helper.from_array(np.ascontiguousarray(b1), "b1"),
            numpy_helper.from_array(np.ascontiguousarray(w2.T), "w2"),
            numpy_helper.from_array(np.ascontiguousarray(b2), "b2"),
        ],
    )
    model = helper.make_model(graph, producer_name="magcode_qairt_split")
    model.ir_version = 8
    del model.opset_import[:]
    opset = model.opset_import.add()
    opset.domain = ""
    opset.version = 13
    return model


def write_float_bin(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(array.astype(np.float32).reshape(-1).tobytes())


def run_mlp_body(initializers: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    w0 = initializers["0.layers.0.weight"]
    b0 = initializers["0.layers.0.bias"]
    w1 = initializers["0.layers.2.weight"]
    b1 = initializers["0.layers.2.bias"]
    w2 = initializers["0.layers.4.weight"]
    b2 = initializers["0.layers.4.bias"]

    y = x @ w0.T + b0
    y = np.maximum(y, 0.0)
    y = y @ w1.T + b1
    y = np.maximum(y, 0.0)
    y = y @ w2.T + b2
    return y.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export reusable split artifacts for an MLP+LayerNorm partition.")
    parser.add_argument("--model", required=True, help="Path to the original ONNX partition model")
    parser.add_argument("--rows", type=int, required=True, help="Static row count for the emitted MLP-body ONNX")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--eps", type=float, default=1e-5, help="LayerNorm epsilon to record in the manifest")
    parser.add_argument("--input-raw", help="Optional float32 raw input tensor for expected MLP-body output generation")
    parser.add_argument("--input-shape", help="Optional JSON shape file for --input-raw")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    initializers = load_initializers(model_path)
    mlp_model = build_mlp_body_model(initializers, args.rows)
    onnx.save(mlp_model, str(out_dir / "mlp_body.onnx"))

    write_float_bin(out_dir / "linear0_weight.bin", initializers["0.layers.0.weight"])
    write_float_bin(out_dir / "linear0_bias.bin", initializers["0.layers.0.bias"])
    write_float_bin(out_dir / "linear1_weight.bin", initializers["0.layers.2.weight"])
    write_float_bin(out_dir / "linear1_bias.bin", initializers["0.layers.2.bias"])
    write_float_bin(out_dir / "linear2_weight.bin", initializers["0.layers.4.weight"])
    write_float_bin(out_dir / "linear2_bias.bin", initializers["0.layers.4.bias"])
    write_float_bin(out_dir / "layernorm_gamma.bin", initializers["1.weight"])
    write_float_bin(out_dir / "layernorm_beta.bin", initializers["1.bias"])

    manifest = {
        "source_model": str(model_path),
        "rows": int(args.rows),
        "input_dim": int(initializers["0.layers.0.weight"].shape[1]),
        "hidden_dim": int(initializers["0.layers.0.weight"].shape[0]),
        "output_dim": int(initializers["0.layers.4.weight"].shape[0]),
        "layernorm_dim": int(initializers["1.weight"].shape[0]),
        "layernorm_eps": float(args.eps),
        "pattern": "qairt_mlp_body_plus_cpu_layernorm",
        "files": {
            "mlp_body_onnx": "mlp_body.onnx",
            "layernorm_gamma": "layernorm_gamma.bin",
            "layernorm_beta": "layernorm_beta.bin",
        },
    }

    if args.input_raw or args.input_shape:
        if not all([args.input_raw, args.input_shape]):
            raise ValueError("Expected both --input-raw and --input-shape together")
        input_shape = read_shape(Path(args.input_shape).resolve())
        x = np.fromfile(Path(args.input_raw).resolve(), dtype=np.float32).reshape(input_shape)
        y = run_mlp_body(initializers, x)
        (out_dir / "mlp_body_expected.raw").write_bytes(y.tobytes())
        (out_dir / "mlp_body_expected_shape.json").write_text(json.dumps([int(v) for v in y.shape]))
        manifest["files"]["mlp_body_expected"] = "mlp_body_expected.raw"
        manifest["files"]["mlp_body_expected_shape"] = "mlp_body_expected_shape.json"

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote split artifacts to {out_dir}")


if __name__ == "__main__":
    main()
