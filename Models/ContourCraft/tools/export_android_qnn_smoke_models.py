import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static


ROWS = 8
COLS = 16


class ZeroDataReader(CalibrationDataReader):
    def __init__(self, input_name: str):
        self._sample = {input_name: np.zeros((ROWS, COLS), dtype=np.float32)}
        self._done = False

    def get_next(self):
        if self._done:
            return None
        self._done = True
        return self._sample


def build_float_model(out_path: Path) -> None:
    rng = np.random.default_rng(1234)
    weight = (rng.standard_normal((COLS, COLS)).astype(np.float32) * 0.01)
    bias = np.zeros((COLS,), dtype=np.float32)

    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["input", "weight"], ["matmul_out"], name="smoke_matmul"),
            helper.make_node("Add", ["matmul_out", "bias"], ["linear"], name="smoke_add"),
            helper.make_node("Relu", ["linear"], ["output"], name="smoke_relu"),
        ],
        "qnn_smoke_float",
        [
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [ROWS, COLS]),
        ],
        [
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [ROWS, COLS]),
        ],
        initializer=[
            numpy_helper.from_array(weight, "weight"),
            numpy_helper.from_array(bias, "bias"),
        ],
    )
    model = helper.make_model(graph, producer_name="magcode_qnn_smoke")
    model.ir_version = 10
    del model.opset_import[:]
    opset = model.opset_import.add()
    opset.domain = ""
    opset.version = 13
    onnx.save(model, str(out_path))


def build_quant_model(float_model_path: Path, out_path: Path) -> None:
    quantize_static(
        model_input=str(float_model_path),
        model_output=str(out_path),
        calibration_data_reader=ZeroDataReader("input"),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
    )


def main() -> None:
    out_root = Path("/Users/jan.rojc/Documents/MagCode/Android/HoodOnnxTest/app/src/main/assets/qnn_smoke")
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    float_model = out_root / "float_matmul_relu.onnx"
    build_float_model(float_model)
    build_quant_model(float_model, out_root / "quant_matmul_relu.onnx")
    print(f"Wrote smoke models to {out_root}")


if __name__ == "__main__":
    main()
