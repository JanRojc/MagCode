import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export multiple real HOOD-prepared frames for Android.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default="hood_final")
    parser.add_argument("--data-root", type=str, default="Data/ccraft_data")
    parser.add_argument("--project-dir", type=str, default="Models/ContourCraft")
    parser.add_argument("--config-dir", type=str, default="Models/ContourCraft/configs")
    parser.add_argument("--sequence-path", type=str, required=True)
    parser.add_argument("--garment-template-path", type=str, required=True)
    parser.add_argument("--gender", type=str, default="male")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--separate-arms", action="store_true", default=True)
    parser.add_argument("--out-dir", type=str, default="Android/HoodOnnxTest/app/src/main/assets/pipeline_real_sequence")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_exporter = Path(__file__).with_name("export_android_real_prepared_frame.py").resolve()
    python_exe = Path(sys.executable)

    frame_entries = []
    asset_root = Path("Android/HoodOnnxTest/app/src/main/assets").resolve()

    for local_idx in range(args.num_frames):
        frame_idx = args.start_frame + local_idx
        frame_dir = out_dir / f"frame_{local_idx:04d}"
        cmd = [str(python_exe), str(frame_exporter)]
        if args.checkpoint:
            cmd.extend(["--checkpoint", args.checkpoint])
        cmd.extend([
            "--config", args.config,
            "--data-root", args.data_root,
            "--project-dir", args.project_dir,
            "--config-dir", args.config_dir,
            "--sequence-path", args.sequence_path,
            "--garment-template-path", args.garment_template_path,
            "--gender", args.gender,
            "--frame-idx", str(frame_idx),
            "--fps", str(args.fps),
            "--out-dir", str(frame_dir),
        ])
        if args.separate_arms:
            cmd.append("--separate-arms")
        subprocess.run(cmd, check=True)

        asset_base = frame_dir.relative_to(asset_root).as_posix()
        frame_entries.append({
            "local_index": local_idx,
            "frame_idx": frame_idx,
            "asset_base": asset_base,
        })

    config = {
        "mode": "prepared_real_sequence",
        "sequence_path": str(Path(args.sequence_path).resolve()),
        "garment_template_path": str(Path(args.garment_template_path).resolve()),
        "gender": args.gender,
        "start_frame": args.start_frame,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "frames": frame_entries,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Wrote real prepared sequence assets to {out_dir}")
    print(f"frames={args.num_frames} start_frame={args.start_frame}")


if __name__ == "__main__":
    main()
