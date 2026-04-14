#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import numpy as np

case_dir = Path(__file__).resolve().parent
expected_path = case_dir / 'expected_output.raw'
shape_path = case_dir / 'expected_output_shape.json'
probe_dir = case_dir / 'output_probe'
if not probe_dir.exists():
    raise SystemExit('output_probe missing')
raw_files = sorted(probe_dir.rglob('*.raw'))
if not raw_files:
    raise SystemExit('no .raw files found under output_probe')
expected = np.fromfile(expected_path, dtype=np.float32)
shape = json.loads(shape_path.read_text())
expected_size = int(np.prod(shape))
for path in raw_files:
    data = np.fromfile(path, dtype=np.float32)
    if data.size == expected_size:
        diff = np.abs(data - expected)
        print(f'compareProbeOutput file={path.relative_to(case_dir)} max_abs={diff.max()} mismatch={(diff > 1e-5).sum()}')
        break
else:
    sizes = ', '.join(f'{p.name}:{np.fromfile(p, dtype=np.float32).size}' for p in raw_files[:8])
    raise SystemExit(f'no output raw matched expected size {expected_size}; saw {sizes}')
