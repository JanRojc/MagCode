#!/usr/bin/env bash
set -euo pipefail
CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE_DIR="/data/local/tmp/qnn_block_0_1_edge_mesh_mlp_body_case"
ADB="${ADB:-adb}"
$ADB shell rm -rf "$DEVICE_DIR"
$ADB shell mkdir -p "$DEVICE_DIR"
for f in qnn-net-run libc++_shared.so libQnnSystem.so libQnnGpuNetRunExtensions.so libQnnGpu.so libblock_0_1_edge_mesh_mlp_body.so input.raw input_list.txt; do
  $ADB push "$CASE_DIR/$f" "$DEVICE_DIR/$f"
done
$ADB shell chmod 755 "$DEVICE_DIR/qnn-net-run"
$ADB shell "cd $DEVICE_DIR && LD_LIBRARY_PATH=$DEVICE_DIR ./qnn-net-run --backend libQnnGpu.so --model libblock_0_1_edge_mesh_mlp_body.so --input_list input_list.txt --output_dir output_probe"
$ADB pull "$DEVICE_DIR/output_probe" "$CASE_DIR/output_probe"
