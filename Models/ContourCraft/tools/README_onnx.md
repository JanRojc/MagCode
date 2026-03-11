# HOOD minimal inference + ONNX export

This folder contains:
- `hood_minimal.py`: minimal inference loader for HOOD (config + checkpoint + forward/rollout).
- `hood_onnx_export.py`: exports HOOD MLPs to ONNX and optionally attempts a full export.

## Requirements
- Python 3
- `torch`
- `torch_geometric` (for `HeteroData` in the dummy sample)
- `onnxruntime` (optional, only for validation)

## Checkpoint path
The repo does **not** include `hood_final.pth`. Provide a checkpoint path or a data root.

Default expected path:
```
${DATA_ROOT}/trained_models/hood_final.pth
```

## Environment overrides
The original code uses hard-coded paths in `utils/defaults.py`.
You can override them via args or env vars.

Supported env vars (used by `hood_minimal.py`):
- `HOOD_DATA_ROOT`
- `HOOD_AUX_DATA`
- `HOOD_PROJECT_DIR`
- `HOOD_CONFIG_DIR`
- `HOOD_CMU_ROOT`
- `HOOD_RESULTS_DIR`
- `HOOD_CHECKPOINT`
- `HOOD_CONFIG`
- `HOOD_DEVICE`

## Minimal inference smoke test
```
PYTHONPATH=Models/ContourCraft \
HOOD_CHECKPOINT=/path/to/hood_final.pth \
python3 Models/ContourCraft/tools/hood_minimal.py
```

## Export MLPs to ONNX
```
PYTHONPATH=Models/ContourCraft \
python3 Models/ContourCraft/tools/hood_onnx_export.py \
  --checkpoint /path/to/hood_final.pth \
  --opset 15 \
  --out Models/ContourCraft/tools/onnx_out
```

### Export with LayerNorm rewritten to primitive ops
```
PYTHONPATH=Models/ContourCraft \
python3 Models/ContourCraft/tools/hood_onnx_export.py \
  --checkpoint /path/to/hood_final.pth \
  --opset 18 \
  --rewrite-layernorm \
  --out Models/ContourCraft/tools/onnx_out_ln_prim
```

## Validate a sample MLP export
```
PYTHONPATH=Models/ContourCraft \
python3 Models/ContourCraft/tools/hood_onnx_export.py \
  --checkpoint /path/to/hood_final.pth \
  --opset 15 \
  --out Models/ContourCraft/tools/onnx_out \
  --validate
```

## Attempt full export (expected to fail)
```
PYTHONPATH=Models/ContourCraft \
python3 Models/ContourCraft/tools/hood_onnx_export.py \
  --checkpoint /path/to/hood_final.pth \
  --opset 15 \
  --out Models/ContourCraft/tools/onnx_out \
  --export-full
```

If it fails, the exception is written to:
```
Models/ContourCraft/tools/onnx_out/full_export_error.txt
```

## Audit ONNX ops
```
PYTHONPATH=Models/ContourCraft \
/Users/jan.rojc/Documents/MagCode/.venv_py310/bin/python Models/ContourCraft/tools/onnx_ops_audit.py \
  Models/ContourCraft/tools/onnx_out_opset18 \
  --out Models/ContourCraft/tools/onnx_ops_report.json
```

## Fail-fast runtime (GPU/NPU only)
Use `ort_runtime.create_session()` and pass preferred providers. It will throw if no preferred provider is available.

## Generate golden IO for mobile validation
```
/Users/jan.rojc/Documents/MagCode/.venv_py310/bin/python Models/ContourCraft/tools/onnx_golden_io.py \
  Models/ContourCraft/tools/onnx_out_opset18/node_encoder.onnx \
  --out Models/ContourCraft/tools/golden/node_encoder.npz
```

## Prepare Android assets (all MLPs)
```
/Users/jan.rojc/Documents/MagCode/.venv_py310/bin/python Models/ContourCraft/tools/prepare_android_assets.py \
  --onnx-dir Models/ContourCraft/tools/onnx_out_opset18 \
  --assets Android/HoodOnnxTest/app/src/main/assets
```

## Analyze NNAPI profiling (which ops fell back to CPU)
After running the Android app, pull the profile:
```
adb pull /sdcard/Android/data/com.magcode.hoodonnxtest/files/ort_profile_<model>.json
```

Then analyze:
```
/Users/jan.rojc/Documents/MagCode/.venv_py310/bin/python Models/ContourCraft/tools/onnx_profile_analyze.py \
  --profile /path/to/ort_profile_<model>.json \
  --model Models/ContourCraft/tools/onnx_out_opset18_lnprim/blocks/block_0_0_edge_coarse_edge0.onnx
```
