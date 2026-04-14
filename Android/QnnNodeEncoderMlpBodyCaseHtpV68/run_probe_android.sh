#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE_DIR="/data/local/tmp/qnn_node_encoder_mlp_body_case"
BACKEND="${1:-gpu}"

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
QNN_SDK_ROOT="${QNN_SDK_ROOT:-}"

if [[ -z "$QNN_SDK_ROOT" ]]; then
  echo "ERROR: set QNN_SDK_ROOT to your QAIRT/QNN SDK root" >&2
  exit 1
fi

required=(
  "$CASE_DIR/qnn-net-run"
  "$CASE_DIR/libc++_shared.so"
  "$CASE_DIR/libQnnSystem.so"
  "$CASE_DIR/$MODEL_LIB"
  "$CASE_DIR/input.raw"
  "$CASE_DIR/input_list.txt"
)

case "$BACKEND" in
  gpu)
    required+=("$CASE_DIR/libQnnGpuNetRunExtensions.so" "$CASE_DIR/libQnnGpu.so")
    ;;
  htp-v68|htp-v69|htp-v73|htp-v75|htp-v79|htp-v81)
    required+=(
      "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so"
      "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpPrepare.so"
    )
    ;;
  *)
    echo "ERROR: unsupported backend '$BACKEND'" >&2
    exit 1
    ;;
esac

for path in "${required[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing required file: $path" >&2
    exit 1
  fi
done

adb shell "rm -rf '$DEVICE_DIR' && mkdir -p '$DEVICE_DIR'"
adb push "$CASE_DIR/qnn-net-run" "$DEVICE_DIR/"
adb push "$CASE_DIR/libc++_shared.so" "$DEVICE_DIR/"
adb push "$CASE_DIR/libQnnSystem.so" "$DEVICE_DIR/"
adb push "$CASE_DIR/$MODEL_LIB" "$DEVICE_DIR/"
adb push "$CASE_DIR/input.raw" "$DEVICE_DIR/"
adb push "$CASE_DIR/input_list.txt" "$DEVICE_DIR/"

case "$BACKEND" in
  gpu)
    adb push "$CASE_DIR/libQnnGpuNetRunExtensions.so" "$DEVICE_DIR/"
    adb push "$CASE_DIR/libQnnGpu.so" "$DEVICE_DIR/"
    RUN_CMD="cd '$DEVICE_DIR' && chmod +x qnn-net-run && export LD_LIBRARY_PATH='$DEVICE_DIR' && ./qnn-net-run --backend libQnnGpu.so --model $MODEL_LIB --input_list input_list.txt --output_dir output_probe"
    ;;
  htp-v68)
    adb push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so" "$DEVICE_DIR/"
    adb push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpPrepare.so" "$DEVICE_DIR/"
    adb push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV68Stub.so" "$DEVICE_DIR/"
    adb push "$QNN_SDK_ROOT/lib/hexagon-v68/unsigned/libQnnHtpV68Skel.so" "$DEVICE_DIR/"
    RUN_CMD="cd '$DEVICE_DIR' && chmod +x qnn-net-run && export LD_LIBRARY_PATH='$DEVICE_DIR' && export ADSP_LIBRARY_PATH='$DEVICE_DIR;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' && ./qnn-net-run --backend libQnnHtp.so --model $MODEL_LIB --input_list input_list.txt --output_dir output_probe"
    ;;
  htp-v69)
    adb push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so" "$DEVICE_DIR/"
    adb push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpPrepare.so" "$DEVICE_DIR/"
    adb push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV69Stub.so" "$DEVICE_DIR/"
    adb push "$QNN_SDK_ROOT/lib/hexagon-v69/unsigned/libQnnHtpV69Skel.so" "$DEVICE_DIR/"
    RUN_CMD="cd '$DEVICE_DIR' && chmod +x qnn-net-run && export LD_LIBRARY_PATH='$DEVICE_DIR' && export ADSP_LIBRARY_PATH='$DEVICE_DIR;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' && ./qnn-net-run --backend libQnnHtp.so --model $MODEL_LIB --input_list input_list.txt --output_dir output_probe"
    ;;
  htp-v73)
    adb push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so" "$DEVICE_DIR/"
    adb push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpPrepare.so" "$DEVICE_DIR/"
    adb push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV73Stub.so" "$DEVICE_DIR/"
    adb push "$QNN_SDK_ROOT/lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so" "$DEVICE_DIR/"
    RUN_CMD="cd '$DEVICE_DIR' && chmod +x qnn-net-run && export LD_LIBRARY_PATH='$DEVICE_DIR' && export ADSP_LIBRARY_PATH='$DEVICE_DIR' && ./qnn-net-run --backend libQnnHtp.so --model $MODEL_LIB --input_list input_list.txt --output_dir output_probe"
    ;;
  htp-v75)
    adb push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so" "$DEVICE_DIR/"
    adb push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpPrepare.so" "$DEVICE_DIR/"
    adb push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV75Stub.so" "$DEVICE_DIR/"
    adb push "$QNN_SDK_ROOT/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so" "$DEVICE_DIR/"
    RUN_CMD="cd '$DEVICE_DIR' && chmod +x qnn-net-run && export LD_LIBRARY_PATH='$DEVICE_DIR' && export ADSP_LIBRARY_PATH='$DEVICE_DIR;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' && ./qnn-net-run --backend libQnnHtp.so --model $MODEL_LIB --input_list input_list.txt --output_dir output_probe"
    ;;
  htp-v79)
    adb push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so" "$DEVICE_DIR/"
    adb push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpPrepare.so" "$DEVICE_DIR/"
    adb push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV79Stub.so" "$DEVICE_DIR/"
    adb push "$QNN_SDK_ROOT/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so" "$DEVICE_DIR/"
    RUN_CMD="cd '$DEVICE_DIR' && chmod +x qnn-net-run && export LD_LIBRARY_PATH='$DEVICE_DIR' && export ADSP_LIBRARY_PATH='$DEVICE_DIR' && ./qnn-net-run --backend libQnnHtp.so --model $MODEL_LIB --input_list input_list.txt --output_dir output_probe"
    ;;
  htp-v81)
    adb push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so" "$DEVICE_DIR/"
    adb push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpPrepare.so" "$DEVICE_DIR/"
    adb push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV81Stub.so" "$DEVICE_DIR/"
    adb push "$QNN_SDK_ROOT/lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so" "$DEVICE_DIR/"
    RUN_CMD="cd '$DEVICE_DIR' && chmod +x qnn-net-run && export LD_LIBRARY_PATH='$DEVICE_DIR' && export ADSP_LIBRARY_PATH='$DEVICE_DIR;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' && ./qnn-net-run --backend libQnnHtp.so --model $MODEL_LIB --input_list input_list.txt --output_dir output_probe"
    ;;
esac

adb shell "$RUN_CMD"

rm -rf "$CASE_DIR/output_probe"
adb pull "$DEVICE_DIR/output_probe" "$CASE_DIR/output_probe"

echo "Backend: $BACKEND"
echo "Pulled probe output to: $CASE_DIR/output_probe"
