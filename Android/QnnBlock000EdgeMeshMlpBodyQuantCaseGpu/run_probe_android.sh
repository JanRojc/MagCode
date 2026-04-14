#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="${1:-htp-v68}"
DEVICE_DIR="/data/local/tmp/qnn_block_0_0_edge_mesh_mlp_body_quant_case_htp_v68"
MODEL_LIB="libblock_0_0_edge_mesh_mlp_body_quant.so"
DEVICE_INPUT_PATH="$(python3 - "$CASE_DIR/case_manifest.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
print(json.loads(p.read_text())["device_input_path"])
PY
)"
printf "%s\n" "$DEVICE_INPUT_PATH" > "$CASE_DIR/input_list.txt"

adb shell "rm -rf '$DEVICE_DIR' && mkdir -p '$DEVICE_DIR'"
for f in qnn-net-run libc++_shared.so libQnnSystem.so libQnnGpuNetRunExtensions.so libQnnGpu.so libQnnHtp.so libQnnHtpPrepare.so libQnnHtpV68Stub.so libQnnHtpV68Skel.so "$MODEL_LIB" input.raw input_list.txt; do
  adb push "$CASE_DIR/$f" "$DEVICE_DIR/"
done

case "$BACKEND" in
  gpu)
    RUN_CMD="cd '$DEVICE_DIR' && chmod +x qnn-net-run && export LD_LIBRARY_PATH='$DEVICE_DIR' && ./qnn-net-run --backend libQnnGpu.so --model $MODEL_LIB --input_list input_list.txt --output_dir output_probe"
    ;;
  htp-v68)
    RUN_CMD="cd '$DEVICE_DIR' && chmod +x qnn-net-run && export LD_LIBRARY_PATH='$DEVICE_DIR' && export ADSP_LIBRARY_PATH='$DEVICE_DIR;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' && ./qnn-net-run --backend libQnnHtp.so --model $MODEL_LIB --input_list input_list.txt --output_dir output_probe"
    ;;
  *)
    echo "Unsupported backend: $BACKEND" >&2
    exit 1
    ;;
esac

HOST_START_NS="$(python3 - <<'PY'
import time
print(time.time_ns())
PY
)"
adb shell "$RUN_CMD"
HOST_END_NS="$(python3 - <<'PY'
import time
print(time.time_ns())
PY
)"
HOST_ELAPSED_MS="$(python3 - "$HOST_START_NS" "$HOST_END_NS" <<'PY'
import sys
start_ns = int(sys.argv[1])
end_ns = int(sys.argv[2])
print(f"{(end_ns - start_ns) / 1_000_000.0:.2f}")
PY
)"
rm -rf "$CASE_DIR/output_probe"
adb pull "$DEVICE_DIR/output_probe" "$CASE_DIR/output_probe"
python3 "$CASE_DIR/compare_probe.py"
echo "Backend: $BACKEND"
echo "single_run_wall_ms: $HOST_ELAPSED_MS"
echo "Pulled probe output to: $CASE_DIR/output_probe"
