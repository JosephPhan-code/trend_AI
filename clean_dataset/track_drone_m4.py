"""
Real-Time Drone Tracking Script for Mac M4
Usage:
  python3 track_drone_m4.py --source path/to/video.mp4
  python3 track_drone_m4.py --source 0   # webcam / capture card
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
from pathlib import Path
import torch
from ultralytics import YOLO


def main():
    base_dir = Path(__file__).resolve().parent
    default_tracker = base_dir / "drone_tracker.yaml"

    # Auto-locate latest best.pt inside runs_mac_m4
    default_weights = None
    runs_dir = base_dir / "runs_mac_m4"
    if runs_dir.exists():
        pts = sorted(runs_dir.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if pts:
            default_weights = pts[0]

    parser = argparse.ArgumentParser(description="Run Drone Tracking on Mac M4")
    parser.add_argument("--weights", type=str, default=str(default_weights) if default_weights else "yolo26s.pt")
    parser.add_argument("--source", type=str, required=True, help="Video file path, RTSP stream, or camera ID (0)")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--show", action="store_true", default=True)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("0" if torch.cuda.is_available() else "cpu")
    print(f"Tracking on device={device} | weights={args.weights} | tracker={default_tracker.name}")

    model = YOLO(args.weights)
    src = int(args.source) if args.source.isdigit() else args.source
    model.track(
        source=src,
        tracker=str(default_tracker),
        imgsz=args.imgsz,
        conf=args.conf,
        device=device,
        persist=True,
        show=args.show,
        save=args.save,
        project=str(base_dir / "tracking_outputs"),
        name="drone_track",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
