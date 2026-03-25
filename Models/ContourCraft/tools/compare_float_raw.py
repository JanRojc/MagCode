import argparse
import json
import math
import struct
from pathlib import Path


def read_shape(path: Path) -> list[int]:
    return [int(x) for x in json.loads(path.read_text())]


def read_f32(path: Path) -> list[float]:
    data = path.read_bytes()
    if len(data) % 4 != 0:
        raise ValueError(f"{path} length {len(data)} is not divisible by 4")
    count = len(data) // 4
    return list(struct.unpack("<" + ("f" * count), data))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two float32 raw tensors.")
    parser.add_argument("--expected", type=Path, required=True, help="Expected float32 .raw/.bin file.")
    parser.add_argument("--actual", type=Path, required=True, help="Actual float32 .raw/.bin file.")
    parser.add_argument("--shape", type=Path, help="Optional shape JSON used for reporting.")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance for mismatch counting.")
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance for mismatch counting.")
    args = parser.parse_args()

    expected = read_f32(args.expected.resolve())
    actual = read_f32(args.actual.resolve())
    if len(expected) != len(actual):
        raise ValueError(f"size mismatch expected={len(expected)} actual={len(actual)}")

    max_abs_diff = 0.0
    max_rel_diff = 0.0
    mismatch_count = 0
    max_idx = 0
    for idx, (exp_val, act_val) in enumerate(zip(expected, actual)):
        abs_diff = abs(exp_val - act_val)
        rel_base = max(abs(exp_val), abs(act_val), 1e-12)
        rel_diff = abs_diff / rel_base
        if abs_diff > max_abs_diff:
            max_abs_diff = abs_diff
            max_rel_diff = rel_diff
            max_idx = idx
        if abs_diff > args.atol + args.rtol * rel_base:
            mismatch_count += 1

    summary = {
        "num_values": len(expected),
        "shape": read_shape(args.shape.resolve()) if args.shape else None,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "max_diff_index": max_idx,
        "expected_at_max": expected[max_idx] if expected else math.nan,
        "actual_at_max": actual[max_idx] if actual else math.nan,
        "mismatch_count": mismatch_count,
        "rtol": args.rtol,
        "atol": args.atol,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
