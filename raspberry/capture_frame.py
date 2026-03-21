"""
capture_frame.py

Captures a single frame from the Pi camera using the same settings as
inference.py and saves it as calibration_frame.jpg.

Usage:
    python capture_frame.py
"""

import logging
from pathlib import Path

import cv2
from libcamera import Transform
from picamera2 import Picamera2

logging.getLogger("picamera2").setLevel(logging.ERROR)
logging.getLogger("libcamera").setLevel(logging.ERROR)

OUT_PATH = Path(__file__).resolve().parent / "calibration_frame.jpg"


def main():
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (1536, 864), "format": "BGR888"},
        transform=Transform(hflip=True, vflip=True),
        controls={
            "FrameDurationLimits": (8333, 8333),
            "ScalerCrop": (1237, 783, 2334, 1626),
        },
    )
    picam2.configure(config)
    picam2.start()
    frame = cv2.cvtColor(picam2.capture_array("main"), cv2.COLOR_RGB2BGR)
    picam2.stop()

    cv2.imwrite(str(OUT_PATH), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
