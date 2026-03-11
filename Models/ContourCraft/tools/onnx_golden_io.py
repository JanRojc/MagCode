import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(description="Generate golden input/output for an ONNX model.")
    parser.add_argument("model", help="Path to .onnx file")
    parser.add_argument("--out", default="golden_io.npz")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    np.random.seed(args.seed)

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape

    # Replace dynamic dims with a small fixed batch
    shape = [dim if isinstance(dim, int) else 8 for dim in input_shape]
    x = np.random.randn(*shape).astype(np.float32)

    y = sess.run(None, {input_name: x})[0]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, input=x, output=y)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
