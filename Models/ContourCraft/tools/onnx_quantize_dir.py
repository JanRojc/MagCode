import argparse
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
    quantize_dynamic,
)


def iter_onnx_files(root: Path):
    for path in root.rglob("*.onnx"):
        if path.name.endswith(".onnx.data"):
            continue
        yield path


def model_input_shape(model_path: Path):
    model = onnx.load(str(model_path))
    # take first input
    input_tensor = model.graph.input[0]
    shape = []
    for dim in input_tensor.type.tensor_type.shape.dim:
        if dim.dim_value > 0:
            shape.append(dim.dim_value)
        else:
            shape.append(8)
    return shape


class RandomDataReader(CalibrationDataReader):
    def __init__(self, input_name: str, shape, num_batches: int, seed: int):
        self.input_name = input_name
        self.shape = shape
        self.num_batches = num_batches
        self.seed = seed
        self._index = 0

    def get_next(self):
        if self._index >= self.num_batches:
            return None
        np.random.seed(self.seed + self._index)
        data = np.random.randn(*self.shape).astype(np.float32)
        self._index += 1
        return {self.input_name: data}


def quantize_model(
    model_path: Path,
    out_path: Path,
    method: str,
    num_batches: int,
    seed: int,
    op_types: list,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if method == "dynamic":
        quantize_dynamic(
            model_input=str(model_path),
            model_output=str(out_path),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=op_types,
        )
        return

    # static QDQ
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    shape = model_input_shape(model_path)
    reader = RandomDataReader(input_name, shape, num_batches=num_batches, seed=seed)

    quantize_static(
        model_input=str(model_path),
        model_output=str(out_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=op_types,
        per_channel=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Quantize all ONNX models in a directory.")
    parser.add_argument("--onnx-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--method", choices=["static", "dynamic"], default="static")
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--ops", default="Gemm,MatMul", help="Comma-separated op types to quantize")
    args = parser.parse_args()

    in_dir = Path(args.onnx_dir)
    out_dir = Path(args.out)
    op_types = [s.strip() for s in args.ops.split(",") if s.strip()]

    for idx, path in enumerate(sorted(iter_onnx_files(in_dir))):
        rel = path.relative_to(in_dir)
        out_path = out_dir / rel
        quantize_model(
            model_path=path,
            out_path=out_path,
            method=args.method,
            num_batches=args.num_batches,
            seed=args.seed + idx,
            op_types=op_types,
        )

        # copy external data if exists
        data_path = path.with_suffix(path.suffix + ".data")
        if data_path.exists():
            out_data = out_path.with_suffix(out_path.suffix + ".data")
            out_data.parent.mkdir(parents=True, exist_ok=True)
            out_data.write_bytes(data_path.read_bytes())

    print(f"Quantized models written to {out_dir}")


if __name__ == "__main__":
    main()
