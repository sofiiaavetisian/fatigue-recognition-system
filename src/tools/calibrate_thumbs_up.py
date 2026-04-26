from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.pipelines.gesture import HandGestureDetector


VIDEO_EXTS = {".mov", ".mp4", ".m4v", ".avi", ".mkv", ".MOV"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate thumbs-up thresholds from labeled videos")
    parser.add_argument("--input-dir", type=str, default="data/labels/thumbs_up")
    parser.add_argument("--output", type=str, default="models/thumbs_up_calibration.json")
    parser.add_argument("--preferred-hand", choices=["right", "left", "any"], default="right")
    parser.add_argument("--sample-every", type=int, default=2, help="Use every Nth frame")
    return parser.parse_args()


def _iter_videos(root: Path) -> list[Path]:
    files = []
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix in VIDEO_EXTS:
            files.append(p)
    return files


def _q(values: list[float], q: float) -> float:
    return float(np.quantile(np.array(values, dtype=np.float32), q))


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_path = Path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    videos = _iter_videos(input_dir)
    if not videos:
        raise RuntimeError(f"No videos found in: {input_dir}")

    detector = HandGestureDetector(preferred_hand=args.preferred_hand, calibration_path=None)

    dy_vals: list[float] = []
    verticality_vals: list[float] = []
    tip_y_vals: list[float] = []
    ip_y_vals: list[float] = []
    index_mcp_y_vals: list[float] = []

    try:
        for vp in videos:
            cap = cv2.VideoCapture(str(vp))
            if not cap.isOpened():
                continue

            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1
                if frame_idx % max(1, args.sample_every) != 0:
                    continue

                metrics = detector.extract_thumb_metrics(frame)
                if metrics is None:
                    continue

                dy_vals.append(metrics["dy"])
                verticality_vals.append(metrics["verticality"])
                tip_y_vals.append(metrics["tip_y"])
                ip_y_vals.append(metrics["ip_y"])
                index_mcp_y_vals.append(metrics["index_mcp_y"])
            cap.release()
    finally:
        detector.close()

    if len(dy_vals) < 30:
        raise RuntimeError(
            f"Too few valid hand frames ({len(dy_vals)}). Make sure hand is visible and preferred hand is correct."
        )

    # Conservative thresholds: accept most of your positive frames while limiting extreme slack.
    dy_max = min(-0.005, _q(dy_vals, 0.80))
    verticality_min = max(0.10, _q(verticality_vals, 0.20))

    above_margin = np.array(np.minimum(ip_y_vals, index_mcp_y_vals)) - np.array(tip_y_vals)
    tip_above_offset = max(0.0, float(_q(above_margin.tolist(), 0.10) * -1.0) + 0.02)

    calibration = {
        "thumb_tip_to_index_mcp_ratio_min": 1.05,
        "thumb_tip_wrist_to_ip_wrist_ratio_min": 1.01,
        "thumb_up_dy_max": float(dy_max),
        "thumb_verticality_min": float(verticality_min),
        "thumb_tip_above_knuckle_offset": float(tip_above_offset),
        "thumbs_up_score_threshold": 0.55,
        "meta": {
            "source_dir": str(input_dir),
            "videos": len(videos),
            "valid_frames": len(dy_vals),
            "preferred_hand": args.preferred_hand,
            "sample_every": args.sample_every,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)

    print(f"Saved calibration to {out_path}")
    print(json.dumps(calibration, indent=2))


if __name__ == "__main__":
    main()
