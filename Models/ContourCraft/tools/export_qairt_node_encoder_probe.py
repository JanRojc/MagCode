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


def build_probe_model(initializers: dict[str, np.ndarray], rows: int) -> onnx.ModelProto:
    w0 = initializers["0.layers.0.weight"]
    b0 = initializers["0.layers.0.bias"]
    w1 = initializers["0.layers.2.weight"]
    b1 = initializers["0.layers.2.bias"]
    w2 = initializers["0.layers.4.weight"]
    b2 = initializers["0.layers.4.bias"]

    input_dim = int(w0.shape[1])
    hidden_dim = int(w0.shape[0])
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
        "node_encoder_probe",
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
    model = helper.make_model(graph, producer_name="magcode_qairt_probe")
    model.ir_version = 8
    del model.opset_import[:]
    opset = model.opset_import.add()
    opset.domain = ""
    opset.version = 13
    return model


def run_probe(initializers: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
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
    parser = argparse.ArgumentParser(description="Export a QAIRT-friendly real-weight node_encoder probe without LayerNorm.")
    parser.add_argument("--model", required=True, help="Path to node_encoder.onnx")
    parser.add_argument("--rows", type=int, required=True, help="Static row count for the probe model")
    parser.add_argument("--out-model", required=True, help="Output ONNX path")
    parser.add_argument("--input-raw", help="Optional float32 raw input tensor for expected-output generation")
    parser.add_argument("--input-shape", help="Optional JSON shape file for --input-raw")
    parser.add_argument("--out-expected", help="Optional output raw path for expected probe output")
    parser.add_argument("--out-expected-shape", help="Optional JSON output shape path for expected probe output")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    out_model = Path(args.out_model).resolve()
    out_model.parent.mkdir(parents=True, exist_ok=True)

    initializers = load_initializers(model_path)
    probe_model = build_probe_model(initializers, args.rows)
    onnx.save(probe_model, str(out_model))
    print(f"wrote probe model to {out_model}")

    if args.input_raw or args.input_shape or args.out_expected or args.out_expected_shape:
        if not all([args.input_raw, args.input_shape, args.out_expected, args.out_expected_shape]):
            raise ValueError("Expected all of --input-raw, --input-shape, --out-expected, --out-expected-shape together")
        input_shape = read_shape(Path(args.input_shape).resolve())
        x = np.fromfile(Path(args.input_raw).resolve(), dtype=np.float32).reshape(input_shape)
        y = run_probe(initializers, x)
        Path(args.out_expected).resolve().write_bytes(y.astype(np.float32).tobytes())
        Path(args.out_expected_shape).resolve().write_text(json.dumps([int(v) for v in y.shape]))
        print(f"wrote expected probe output to {Path(args.out_expected).resolve()}")
        print(f"expected output shape={list(y.shape)}")


if __name__ == "__main__":
    main()
