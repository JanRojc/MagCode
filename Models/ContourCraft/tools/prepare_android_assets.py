import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def iter_onnx_files(root: Path):
    for path in root.rglob("*.onnx"):
        if path.name.endswith(".onnx.data"):
            continue
        yield path


def make_random_input(input_shape, seed):
    np.random.seed(seed)
    shape = [dim if isinstance(dim, int) else 8 for dim in input_shape]
    return np.random.randn(*shape).astype(np.float32)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Prepare Android assets for NNAPI test.")
    parser.add_argument("--onnx-dir", required=True, help="Directory with exported ONNX models")
    parser.add_argument("--assets", required=True, help="Android assets directory")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    onnx_dir = Path(args.onnx_dir)
    assets_dir = Path(args.assets)

    models_dir = assets_dir / "models"
    golden_dir = assets_dir / "golden"

    tests = []

    for idx, model_path in enumerate(sorted(iter_onnx_files(onnx_dir))):
        rel = model_path.relative_to(onnx_dir)
        model_asset = models_dir / rel
        model_asset.parent.mkdir(parents=True, exist_ok=True)
        model_asset.write_bytes(model_path.read_bytes())

        # copy external data if exists
        data_path = model_path.with_suffix(model_path.suffix + ".data")
        if data_path.exists():
            data_asset = models_dir / data_path.relative_to(onnx_dir)
            data_asset.parent.mkdir(parents=True, exist_ok=True)
            data_asset.write_bytes(data_path.read_bytes())

        # generate golden IO
        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        input_shape = sess.get_inputs()[0].shape

        x = make_random_input(input_shape, args.seed + idx)
        y = sess.run(None, {input_name: x})[0]

        golden_base = golden_dir / rel.with_suffix("")
        golden_base.parent.mkdir(parents=True, exist_ok=True)

        input_bin = golden_base.with_suffix(".input.bin")
        output_bin = golden_base.with_suffix(".output.bin")
        input_shape_json = golden_base.with_suffix(".input_shape.json")
        output_shape_json = golden_base.with_suffix(".output_shape.json")

        input_bin.write_bytes(x.tobytes(order="C"))
        output_bin.write_bytes(y.astype(np.float32).tobytes(order="C"))
        input_shape_json.write_text(json.dumps(list(x.shape)))
        output_shape_json.write_text(json.dumps(list(y.shape)))

        tests.append({
            "name": rel.as_posix(),
            "model": f"models/{rel.as_posix()}",
            "input_bin": f"golden/{rel.with_suffix('').as_posix()}.input.bin",
            "output_bin": f"golden/{rel.with_suffix('').as_posix()}.output.bin",
            "input_shape": f"golden/{rel.with_suffix('').as_posix()}.input_shape.json",
            "output_shape": f"golden/{rel.with_suffix('').as_posix()}.output_shape.json",
        })

    write_json(assets_dir / "tests.json", tests)
    print(f"Prepared {len(tests)} tests in {assets_dir}")


if __name__ == "__main__":
    main()
