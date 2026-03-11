import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import onnx


def iter_onnx_files(root: Path):
    for path in root.rglob("*.onnx"):
        if path.name.endswith(".onnx.data"):
            continue
        yield path


def load_ops(path: Path):
    model = onnx.load(path)
    opset = [(op.domain, op.version) for op in model.opset_import]
    ops = [node.op_type for node in model.graph.node]
    return opset, ops


def audit_dir(root: Path):
    per_file = {}
    aggregate = Counter()
    opset_versions = Counter()

    for path in iter_onnx_files(root):
        opset, ops = load_ops(path)
        per_file[str(path)] = {
            "opset_import": opset,
            "ops": Counter(ops),
        }
        aggregate.update(ops)
        for domain, version in opset:
            if domain == "":
                opset_versions[version] += 1

    return {
        "root": str(root),
        "opset_versions": opset_versions,
        "aggregate_ops": aggregate,
        "per_file": per_file,
    }


def to_json_serializable(report):
    report = dict(report)
    report["opset_versions"] = {str(k): v for k, v in report["opset_versions"].items()}
    report["aggregate_ops"] = {k: v for k, v in report["aggregate_ops"].items()}
    per_file = {}
    for k, v in report["per_file"].items():
        per_file[k] = {
            "opset_import": v["opset_import"],
            "ops": {op: cnt for op, cnt in v["ops"].items()},
        }
    report["per_file"] = per_file
    return report


def main():
    parser = argparse.ArgumentParser(description="Audit ONNX ops used in exported models.")
    parser.add_argument("roots", nargs="+", help="One or more directories containing .onnx files")
    parser.add_argument("--out", default="onnx_ops_report.json")
    args = parser.parse_args()

    reports = []
    for root in args.roots:
        reports.append(audit_dir(Path(root)))

    out = {"reports": [to_json_serializable(r) for r in reports]}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
