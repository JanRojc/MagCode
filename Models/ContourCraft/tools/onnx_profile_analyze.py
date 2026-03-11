import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import onnx


def load_profile(path: Path):
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "traceEvents" in data:
        events = data["traceEvents"]
    elif isinstance(data, list):
        events = data
    else:
        events = []
    return events


def load_onnx_node_map(path: Path):
    model = onnx.load(path)
    node_map = {}
    for idx, node in enumerate(model.graph.node):
        name = node.name or f"node_{idx}"
        node_map[name] = node.op_type
    return node_map


def analyze(profile_path: Path, model_path: Path):
    events = load_profile(profile_path)
    node_map = load_onnx_node_map(model_path)

    by_provider = Counter()
    by_op = Counter()
    unknown = Counter()

    for ev in events:
        if ev.get("cat") not in ("Node", "Kernel"):
            continue
        args = ev.get("args", {})
        provider = args.get("provider", "unknown")
        op_name = args.get("op_name")
        node_name = args.get("node_name") or args.get("name") or ev.get("name")

        by_provider[provider] += 1

        if op_name:
            by_op[op_name] += 1
        elif node_name and node_name in node_map:
            by_op[node_map[node_name]] += 1
        else:
            unknown[node_name or "unknown"] += 1

    return {
        "providers": by_provider,
        "ops": by_op,
        "unknown_nodes": unknown,
    }


def to_json(report):
    return {
        "providers": {k: v for k, v in report["providers"].items()},
        "ops": {k: v for k, v in report["ops"].items()},
        "unknown_nodes": {k: v for k, v in report["unknown_nodes"].items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze ORT profiling JSON and map providers/op types.")
    parser.add_argument("--profile", required=True, help="Path to ORT profiling JSON from device")
    parser.add_argument("--model", required=True, help="Path to ONNX model used")
    parser.add_argument("--out", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    report = analyze(Path(args.profile), Path(args.model))
    out = to_json(report)
    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
