"""
record_camera_check.py

Records a short clip using picamera2 with the same camera settings as
inference.py (same ScalerCrop / zoom). Saves to videos/manual_camera_check.mp4.

Press Ctrl+C to stop early.

Usage:
    python record_camera_check.py
    python record_camera_check.py --duration 30
"""

import argparse
import logging
import time
from pathlib import Path

from libcamera import Transform
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput

logging.getLogger("picamera2").setLevel(logging.ERROR)
logging.getLogger("libcamera").setLevel(logging.ERROR)

OUT_PATH = Path(__file__).resolve().parent / "videos" / "manual_camera_check.mp4"


def main():
    ap = argparse.ArgumentParser(description="Record a camera check clip with correct zoom.")
    ap.add_argument("--duration", type=int, default=60, help="Recording duration in seconds (default: 60)")
    args = ap.parse_args()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (1536, 864), "format": "BGR888"},
        transform=Transform(hflip=True, vflip=True),
        controls={
            "FrameDurationLimits": (8333, 8333),
            "ScalerCrop": (1152, 648, 2304, 1296),  # 50% zoom
        },
    )
    picam2.configure(config)

    print(f"Recording for up to {args.duration}s (Ctrl+C to stop early)...")
    print(f"Output: {OUT_PATH}")

    try:
        picam2.start_recording(H264Encoder(), FfmpegOutput(str(OUT_PATH)))
        time.sleep(args.duration)
    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        picam2.stop_recording()
        print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
