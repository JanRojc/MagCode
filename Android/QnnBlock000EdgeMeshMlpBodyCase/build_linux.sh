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
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
print(data[field])
PY
}

MODEL_NAME="$(read_json_field model_name)"
INPUT_NAME="$(read_json_field input_name)"
ROWS="$(read_json_field rows)"
INPUT_DIM="$(read_json_field input_dim)"

QNN_ONNX_CONVERTER="$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-onnx-converter"
QNN_MODEL_LIB_GENERATOR="$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-model-lib-generator"

mkdir -p "$BUILD_DIR"

set +u
source "$QNN_SDK_ROOT/bin/envsetup.sh"
set -u

echo "Converting ONNX -> QNN C++/bin"
"$QNN_ONNX_CONVERTER" \
  --input_network "$CASE_DIR/block_0_0_edge_mesh_mlp_body.onnx" \
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
