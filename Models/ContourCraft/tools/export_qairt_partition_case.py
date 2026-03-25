import argparse
import json
import shutil
import struct
from pathlib import Path

PARTITIONS = {
    "edge_encoder_mesh": {
        "input_raw": "mesh_raw.bin",
        "input_shape": "mesh_raw_shape.json",
        "expected_raw": "expected_edge_encoder_mesh.bin",
        "expected_shape": "expected_edge_encoder_mesh_shape.json",
    },
    "edge_encoder_coarse0": {
        "input_raw": "coarse0_raw.bin",
        "input_shape": "coarse0_raw_shape.json",
        "expected_raw": "expected_edge_encoder_coarse0.bin",
        "expected_shape": "expected_edge_encoder_coarse0_shape.json",
    },
    "edge_encoder_coarse1": {
        "input_raw": "coarse1_raw.bin",
        "input_shape": "coarse1_raw_shape.json",
        "expected_raw": "expected_edge_encoder_coarse1.bin",
        "expected_shape": "expected_edge_encoder_coarse1_shape.json",
    },
    "edge_encoder_coarse2": {
        "input_raw": "coarse2_raw.bin",
        "input_shape": "coarse2_raw_shape.json",
        "expected_raw": "expected_edge_encoder_coarse2.bin",
        "expected_shape": "expected_edge_encoder_coarse2_shape.json",
    },
    "edge_encoder_world": {
        "input_raw": None,
        "input_shape": None,
        "expected_raw": None,
        "expected_shape": None,
    },
    "block_0_0_node_cloth": {
        "input_raw": "blocks/block_0_0_node_in_cloth.bin",
        "input_shape": "blocks/block_0_0_node_in_cloth_shape.json",
        "expected_raw": "blocks/block_0_0_node_out_cloth.bin",
        "expected_shape": "blocks/block_0_0_node_out_cloth_shape.json",
    },
    "block_0_0_edge_mesh_edge": {
        "builder": "block_edge_mesh",
    },
    "block_0_0_edge_world_direct": {
        "builder": "block_edge_world_direct",
    },
}


def read_shape(path: Path) -> list[int]:
    return [int(x) for x in json.loads(path.read_text())]


def read_raw_checked(path: Path, shape: list[int], item_size: int = 4) -> bytes:
    data = path.read_bytes()
    expected = item_size
    for dim in shape:
        expected *= int(dim)
    if len(data) != expected:
        raise ValueError(f"{path} has {len(data)} bytes, expected {expected} for shape {shape}")
    return data


def read_ints(path: Path, shape: list[int]) -> list[int]:
    data = read_raw_checked(path, shape, 4)
    count = len(data) // 4
    return list(struct.unpack("<" + ("i" * count), data))


def row_slice(data: bytes, cols: int, row_idx: int) -> bytes:
    row_bytes = cols * 4
    start = row_idx * row_bytes
    return data[start : start + row_bytes]


def build_block_edge_case(
    *,
    frame_dir: Path,
    src_raw_name: str,
    src_shape_name: str,
    tgt_raw_name: str,
    tgt_shape_name: str,
    edge_raw_name: str,
    edge_shape_name: str,
    edge_index_name: str,
    edge_index_shape_name: str,
    expected_raw_name: str,
    out_dir: Path,
) -> tuple[list[int], list[int]]:
    src_shape = read_shape(frame_dir / src_shape_name)
    tgt_shape = read_shape(frame_dir / tgt_shape_name)
    edge_shape = read_shape(frame_dir / edge_shape_name)
    edge_index_shape = read_shape(frame_dir / edge_index_shape_name)

    src_raw = read_raw_checked(frame_dir / src_raw_name, src_shape)
    tgt_raw = read_raw_checked(frame_dir / tgt_raw_name, tgt_shape)
    edge_raw = read_raw_checked(frame_dir / edge_raw_name, edge_shape)
    edge_index_vals = read_ints(frame_dir / edge_index_name, edge_index_shape)
    expected_raw = (frame_dir / expected_raw_name).read_bytes()

    num_edges = edge_shape[0]
    latent = edge_shape[1]
    if edge_index_shape != [2, num_edges]:
        raise ValueError(f"edge index shape mismatch for {edge_index_name}: {edge_index_shape} vs expected [2, {num_edges}]")

    src_indices = edge_index_vals[:num_edges]
    tgt_indices = edge_index_vals[num_edges:]
    input_rows = []
    for edge_idx in range(num_edges):
        input_rows.append(row_slice(tgt_raw, tgt_shape[1], tgt_indices[edge_idx]))
        input_rows.append(row_slice(src_raw, src_shape[1], src_indices[edge_idx]))
        input_rows.append(row_slice(edge_raw, latent, edge_idx))

    input_shape = [num_edges, tgt_shape[1] + src_shape[1] + latent]
    expected_shape = [num_edges, latent]

    (out_dir / "input.raw").write_bytes(b"".join(input_rows))
    (out_dir / "input_shape.json").write_text(json.dumps(input_shape))
    (out_dir / "expected_output.raw").write_bytes(expected_raw)
    (out_dir / "expected_output_shape.json").write_text(json.dumps(expected_shape))
    return input_shape, expected_shape


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble a QAIRT shell-validation case for a simple rank-2 HOOD partition.")
    parser.add_argument("--partition", choices=sorted(PARTITIONS.keys()), required=True)
    parser.add_argument(
        "--frame-dir",
        type=Path,
        default=Path("/Users/jan.rojc/Documents/MagCode/Android/HoodOnnxTest/app/src/main/assets/pipeline_real_sequence/frame_0000"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device-input-path", required=True)
    args = parser.parse_args()

    spec = PARTITIONS[args.partition]
    frame_dir = args.frame_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    input_shape = None
    expected_shape = None

    if spec.get("builder") == "block_edge_mesh":
        input_shape, expected_shape = build_block_edge_case(
            frame_dir=frame_dir,
            src_raw_name="expected_node_encoder_cloth.bin",
            src_shape_name="expected_node_encoder_cloth_shape.json",
            tgt_raw_name="expected_node_encoder_cloth.bin",
            tgt_shape_name="expected_node_encoder_cloth_shape.json",
            edge_raw_name="expected_edge_encoder_mesh.bin",
            edge_shape_name="expected_edge_encoder_mesh_shape.json",
            edge_index_name="edge_index_mesh.bin",
            edge_index_shape_name="edge_index_mesh_shape.json",
            expected_raw_name="blocks/block_0_0_updated_mesh.bin",
            out_dir=out_dir,
        )
    elif spec.get("builder") == "block_edge_world_direct":
        input_shape, expected_shape = build_block_edge_case(
            frame_dir=frame_dir,
            src_raw_name="expected_node_encoder_obstacle.bin",
            src_shape_name="expected_node_encoder_obstacle_shape.json",
            tgt_raw_name="expected_node_encoder_cloth.bin",
            tgt_shape_name="expected_node_encoder_cloth_shape.json",
            edge_raw_name="expected_edge_encoder_world_direct.bin",
            edge_shape_name="expected_edge_encoder_world_direct_shape.json",
            edge_index_name="edge_index_world_direct.bin",
            edge_index_shape_name="edge_index_world_direct_shape.json",
            expected_raw_name="blocks/block_0_0_updated_world_direct.bin",
            out_dir=out_dir,
        )
    elif args.partition == "edge_encoder_world":
        direct_shape = json.loads((frame_dir / "world_direct_raw_shape.json").read_text())
        inverse_shape = json.loads((frame_dir / "world_inverse_raw_shape.json").read_text())
        expected_direct_shape = json.loads((frame_dir / "expected_edge_encoder_world_direct_shape.json").read_text())
        expected_inverse_shape = json.loads((frame_dir / "expected_edge_encoder_world_inverse_shape.json").read_text())
        direct_raw = (frame_dir / "world_direct_raw.bin").read_bytes()
        inverse_raw = (frame_dir / "world_inverse_raw.bin").read_bytes()
        expected_direct_raw = (frame_dir / "expected_edge_encoder_world_direct.bin").read_bytes()
        expected_inverse_raw = (frame_dir / "expected_edge_encoder_world_inverse.bin").read_bytes()

        (out_dir / "input.raw").write_bytes(direct_raw + inverse_raw)
        input_shape = [int(direct_shape[0] + inverse_shape[0]), int(direct_shape[1])]
        (out_dir / "input_shape.json").write_text(json.dumps(input_shape))
        (out_dir / "expected_output.raw").write_bytes(expected_direct_raw + expected_inverse_raw)
        expected_shape = [int(expected_direct_shape[0] + expected_inverse_shape[0]), int(expected_direct_shape[1])]
        (out_dir / "expected_output_shape.json").write_text(json.dumps(expected_shape))
    else:
        shutil.copy2(frame_dir / spec["input_raw"], out_dir / "input.raw")
        shutil.copy2(frame_dir / spec["input_shape"], out_dir / "input_shape.json")
        shutil.copy2(frame_dir / spec["expected_raw"], out_dir / "expected_output.raw")
        shutil.copy2(frame_dir / spec["expected_shape"], out_dir / "expected_output_shape.json")
        input_shape = read_shape(out_dir / "input_shape.json")
        expected_shape = read_shape(out_dir / "expected_output_shape.json")
    (out_dir / "input_list.txt").write_text(args.device_input_path + "\n")
    (out_dir / "case_manifest.json").write_text(
        json.dumps(
            {
                "partition": args.partition,
                "frame_dir": str(frame_dir),
                "device_input_path": args.device_input_path,
                "input_raw": spec.get("input_raw"),
                "input_shape": spec.get("input_shape"),
                "expected_raw": spec.get("expected_raw"),
                "expected_shape": spec.get("expected_shape"),
                "builder": spec.get("builder"),
                "resolved_input_shape": input_shape,
                "resolved_expected_shape": expected_shape,
            },
            indent=2,
        )
    )
    print(f"wrote {args.partition} QAIRT case to {out_dir}")


if __name__ == "__main__":
    main()
