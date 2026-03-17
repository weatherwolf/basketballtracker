import time
import shutil
import logging
import threading
import numpy as np
import cv2
from pathlib import Path
from picamera2 import Picamera2
from libcamera import Transform

logging.getLogger("picamera2").setLevel(logging.ERROR)
logging.getLogger("libcamera").setLevel(logging.ERROR)

LOWER_ORANGE = np.array([5, 150, 100])
UPPER_ORANGE = np.array([20, 255, 255])
MIN_BALL_PIXELS = 50
FILTER_EVERY_N  = 4
DOWNSCALE_W, DOWNSCALE_H = 384, 216
PENDING_DIR = Path(__file__).parent / "pending"


def ball_present(frame_bgr: np.ndarray) -> bool:
    small = cv2.resize(frame_bgr, (DOWNSCALE_W, DOWNSCALE_H))
    hsv   = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask  = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)
    return cv2.countNonZero(mask) >= MIN_BALL_PIXELS


def save_pending(frames: list) -> None:
    if PENDING_DIR.exists():
        shutil.rmtree(PENDING_DIR)
    PENDING_DIR.mkdir(parents=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(str(PENDING_DIR / f"frame_{i:06d}.jpg"), frame)


def main() -> None:
    PENDING_DIR.mkdir(parents=True, exist_ok=True)

    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (1536, 864), "format": "BGR888"},
        transform=Transform(hflip=True, vflip=True),
        controls={
            "FrameDurationLimits": (8333, 8333),  # 1 / 120 s in µs
            "ScalerCrop": (1037, 583, 2534, 1426),  # 45% zoom
        },
    )
    picam2.configure(config)
    picam2.start()

    ball_in_frame = False
    frame_count   = 0
    shot_frames   = 0
    frame_buffer  = []
    save_thread   = None

    try:
        while True:
            frame = cv2.cvtColor(picam2.capture_array("main"), cv2.COLOR_RGB2BGR)
            frame_count += 1

            if ball_in_frame:
                frame_buffer.append(frame)
                shot_frames += 1

            if frame_count % FILTER_EVERY_N != 0:
                continue

            detected = ball_present(frame)

            if detected and not ball_in_frame:
                ball_in_frame = True
                shot_frames   = 0
                frame_buffer  = [frame]
                print(f"[{time.strftime('%H:%M:%S')}] Ball entered frame", flush=True)

            elif not detected and ball_in_frame:
                ball_in_frame = False
                print(f"[{time.strftime('%H:%M:%S')}] Ball left frame  ({shot_frames} frames)", flush=True)
                save_thread = threading.Thread(target=save_pending, args=(frame_buffer.copy(),))
                save_thread.start()
                frame_buffer = []

    except KeyboardInterrupt:
        pass
    finally:
        if frame_buffer:
            save_thread = threading.Thread(target=save_pending, args=(frame_buffer.copy(),))
            save_thread.start()
        if save_thread and save_thread.is_alive():
            save_thread.join()
        picam2.stop()


if __name__ == "__main__":
    main()
