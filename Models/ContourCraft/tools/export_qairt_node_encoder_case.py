import argparse
import json
import struct
from pathlib import Path


def read_shape(path: Path) -> list[int]:
    return [int(x) for x in json.loads(path.read_text())]


def expected_bytes(shape: list[int], item_size: int) -> int:
    count = 1
    for dim in shape:
        count *= dim
    return count * item_size


def read_raw_checked(path: Path, shape: list[int], item_size: int) -> bytes:
    data = path.read_bytes()
    expected = expected_bytes(shape, item_size)
    if len(data) != expected:
        raise ValueError(f"{path} has {len(data)} bytes, expected {expected} for shape {shape}")
    return data


def read_ints(path: Path, shape: list[int]) -> list[int]:
    data = read_raw_checked(path, shape, 4)
    count = len(data) // 4
    return list(struct.unpack("<" + ("i" * count), data))


def select_rows(raw: bytes, rows: int, cols: int, selected_rows: list[int]) -> bytes:
    row_bytes = cols * 4
    expected = rows * row_bytes
    if len(raw) != expected:
        raise ValueError(f"raw payload has {len(raw)} bytes, expected {expected} for rows={rows} cols={cols}")
    chunks = []
    for row_idx in selected_rows:
        start = row_idx * row_bytes
        chunks.append(raw[start : start + row_bytes])
    return b"".join(chunks)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.write_bytes(src.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble a QAIRT shell-validation case for the HOOD node_encoder partition.")
    parser.add_argument(
        "--frame-dir",
        type=Path,
        default=Path("/Users/jan.rojc/Documents/MagCode/Android/HoodOnnxTest/app/src/main/assets/pipeline_real_sequence/frame_0000"),
        help="Prepared frame asset directory containing cloth_raw.bin, obstacle_raw.bin, mask, and expected node encoder outputs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/jan.rojc/Documents/MagCode/Android/QnnNodeEncoderCase"),
        help="Output directory for the assembled shell-validation case.",
    )
    parser.add_argument(
        "--device-input-path",
        default="/data/local/tmp/qnn_node_encoder/node_encoder_input.raw",
        help="Absolute device path written into input_list.txt for qnn-net-run.",
    )
    args = parser.parse_args()

    frame_dir = args.frame_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cloth_shape = read_shape(frame_dir / "cloth_raw_shape.json")
    obstacle_shape = read_shape(frame_dir / "obstacle_raw_shape.json")
    mask_shape = read_shape(frame_dir / "obstacle_active_mask_shape.json")
    expected_cloth_shape = read_shape(frame_dir / "expected_node_encoder_cloth_shape.json")
    expected_obstacle_shape = read_shape(frame_dir / "expected_node_encoder_obstacle_shape.json")

    if len(cloth_shape) != 2 or len(obstacle_shape) != 2:
        raise ValueError("cloth_raw and obstacle_raw must be rank-2 tensors")
    if len(mask_shape) != 2:
        raise ValueError("obstacle_active_mask must be rank-2")
    if cloth_shape[1] != obstacle_shape[1]:
        raise ValueError("cloth and obstacle node feature dimensions must match")
    if expected_cloth_shape[1] != expected_obstacle_shape[1]:
        raise ValueError("cloth and obstacle latent dimensions must match")
    if obstacle_shape[0] != mask_shape[0]:
        raise ValueError("obstacle rows and obstacle mask rows must match")

    cloth_raw = read_raw_checked(frame_dir / "cloth_raw.bin", cloth_shape, 4)
    obstacle_raw = read_raw_checked(frame_dir / "obstacle_raw.bin", obstacle_shape, 4)
    mask_values = read_ints(frame_dir / "obstacle_active_mask.bin", mask_shape)
    expected_cloth = read_raw_checked(frame_dir / "expected_node_encoder_cloth.bin", expected_cloth_shape, 4)
    expected_obstacle = read_raw_checked(frame_dir / "expected_node_encoder_obstacle.bin", expected_obstacle_shape, 4)

    mask_stride = mask_shape[1]
    active_rows = [idx for idx in range(mask_shape[0]) if mask_values[idx * mask_stride] != 0]

    combined_input = cloth_raw + select_rows(obstacle_raw, obstacle_shape[0], obstacle_shape[1], active_rows)
    combined_expected = expected_cloth + select_rows(expected_obstacle, expected_obstacle_shape[0], expected_obstacle_shape[1], active_rows)

    combined_input_shape = [cloth_shape[0] + len(active_rows), cloth_shape[1]]
    combined_expected_shape = [expected_cloth_shape[0] + len(active_rows), expected_cloth_shape[1]]

    (out_dir / "node_encoder_input.raw").write_bytes(combined_input)
    (out_dir / "node_encoder_input_shape.json").write_text(json.dumps(combined_input_shape))
    (out_dir / "expected_node_encoder_output.raw").write_bytes(combined_expected)
    (out_dir / "expected_node_encoder_output_shape.json").write_text(json.dumps(combined_expected_shape))
    (out_dir / "input_list.txt").write_text(args.device_input_path + "\n")

    copy_if_exists(frame_dir / "cloth_raw.bin", out_dir / "cloth_raw.bin")
    copy_if_exists(frame_dir / "obstacle_raw.bin", out_dir / "obstacle_raw.bin")
    copy_if_exists(frame_dir / "obstacle_active_mask.bin", out_dir / "obstacle_active_mask.bin")
    copy_if_exists(frame_dir / "expected_node_encoder_cloth.bin", out_dir / "expected_node_encoder_cloth.bin")
    copy_if_exists(frame_dir / "expected_node_encoder_obstacle.bin", out_dir / "expected_node_encoder_obstacle.bin")

    metadata = {
        "frame_dir": str(frame_dir),
        "cloth_shape": cloth_shape,
        "obstacle_shape": obstacle_shape,
        "obstacle_active_mask_shape": mask_shape,
        "expected_cloth_shape": expected_cloth_shape,
        "expected_obstacle_shape": expected_obstacle_shape,
        "combined_input_shape": combined_input_shape,
        "combined_expected_output_shape": combined_expected_shape,
        "cloth_rows": cloth_shape[0],
        "obstacle_rows": obstacle_shape[0],
        "active_obstacle_rows": len(active_rows),
        "active_obstacle_indices_path": "active_obstacle_indices.json",
        "device_input_path": args.device_input_path,
    }
    (out_dir / "active_obstacle_indices.json").write_text(json.dumps(active_rows))
    (out_dir / "case_manifest.json").write_text(json.dumps(metadata, indent=2))

    print(f"wrote node_encoder QAIRT case to {out_dir}")
    print(f"combined input shape={combined_input_shape}")
    print(f"combined expected output shape={combined_expected_shape}")
    print(f"active obstacle rows={len(active_rows)} / {obstacle_shape[0]}")


if __name__ == "__main__":
    main()
