#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$CASE_DIR/build"

if [[ -z "${QNN_SDK_ROOT:-}" ]]; then
  echo "ERROR: set QNN_SDK_ROOT to your QAIRT/QNN SDK root" >&2
  exit 1
fi

if [[ -z "${ANDROID_NDK_ROOT:-}" ]]; then
  echo "ERROR: set ANDROID_NDK_ROOT to your Android NDK root" >&2
  exit 1
fi

read_json_field() {
  local field="$1"
  python3 - "$CASE_DIR/case_manifest.json" "$field" <<'PY'
import json, sys
path, field = sys.argv[1], sys.argv[2]
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)
print(data[field])
PY
}

MODEL_NAME="$(read_json_field model_name)"
FLOAT_MODEL_NAME="$(read_json_field float_model_name)"
QUANT_MODEL_NAME="$(read_json_field quant_model_name)"
INPUT_NAME="$(read_json_field input_name)"
ROWS="$(read_json_field rows)"
INPUT_DIM="$(read_json_field input_dim)"

QNN_ONNX_CONVERTER="$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-onnx-converter"
QNN_MODEL_LIB_GENERATOR="$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-model-lib-generator"

mkdir -p "$BUILD_DIR"

set +u
source "$QNN_SDK_ROOT/bin/envsetup.sh"
set -u

python3 - "$CASE_DIR" "$FLOAT_MODEL_NAME" "$QUANT_MODEL_NAME" "$INPUT_NAME" "$ROWS" "$INPUT_DIM" <<'PY'
import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

case_dir = Path(sys.argv[1])
float_model = case_dir / sys.argv[2]
quant_model = case_dir / sys.argv[3]
input_name = sys.argv[4]
rows = int(sys.argv[5])
input_dim = int(sys.argv[6])
input_path = case_dir / 'input.raw'

class RawReader(CalibrationDataReader):
    def __init__(self):
        data = np.fromfile(input_path, dtype=np.float32).reshape(rows, input_dim)
        self._sample = {input_name: data}
        self._done = False

    def get_next(self):
        if self._done:
            return None
        self._done = True
        return self._sample

# Sanity-check model input name when possible.
sess = ort.InferenceSession(str(float_model), providers=['CPUExecutionProvider'])
model_input = sess.get_inputs()[0].name
sess = None
if model_input != input_name:
    raise SystemExit(f'Input name mismatch: manifest={input_name} model={model_input}')

quantize_static(
    model_input=str(float_model),
    model_output=str(quant_model),
    calibration_data_reader=RawReader(),
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    op_types_to_quantize=['MatMul'],
    per_channel=False,
)
print(f'Wrote quantized model: {quant_model}')
PY

echo "Converting quantized ONNX -> QNN C++/bin"
"$QNN_ONNX_CONVERTER" \
  --input_network "$CASE_DIR/$QUANT_MODEL_NAME" \
  --input_dim "$INPUT_NAME" "${ROWS},${INPUT_DIM}" \
  --output_path "$BUILD_DIR/${MODEL_NAME}.cpp"

echo "Building Android model library"
"$QNN_MODEL_LIB_GENERATOR" \
  -c "$BUILD_DIR/${MODEL_NAME}.cpp" \
  -b "$BUILD_DIR/${MODEL_NAME}.bin" \
  -l "$MODEL_NAME" \
  -o "$BUILD_DIR/model_libs" \
  -t aarch64-android

MODEL_LIB="$BUILD_DIR/model_libs/aarch64-android/lib${MODEL_NAME}.so"
if [[ ! -f "$MODEL_LIB" ]]; then
  echo "ERROR: generated model lib not found: $MODEL_LIB" >&2
  exit 1
fi

cp "$MODEL_LIB" "$CASE_DIR/"

echo "Built: $CASE_DIR/lib${MODEL_NAME}.so"
