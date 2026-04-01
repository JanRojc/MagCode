#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE_DIR="/data/local/tmp/qnn_block_0_0_edge_mesh_mlp_body_case"

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
MODEL_LIB="lib${MODEL_NAME}.so"

adb shell "rm -rf '$DEVICE_DIR' && mkdir -p '$DEVICE_DIR'"
for f in qnn-net-run libc++_shared.so libQnnSystem.so libQnnGpuNetRunExtensions.so libQnnGpu.so "$MODEL_LIB" input.raw input_list.txt; do
  adb push "$CASE_DIR/$f" "$DEVICE_DIR/"
done

adb shell "cd '$DEVICE_DIR' && chmod +x qnn-net-run && export LD_LIBRARY_PATH='$DEVICE_DIR' && ./qnn-net-run --backend libQnnGpu.so --model $MODEL_LIB --input_list input_list.txt --output_dir output_probe"

rm -rf "$CASE_DIR/output_probe"
adb pull "$DEVICE_DIR/output_probe" "$CASE_DIR/output_probe"

echo "Backend: gpu"
echo "Pulled probe output to: $CASE_DIR/output_probe"
