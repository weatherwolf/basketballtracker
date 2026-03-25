import pickle
import time
from pathlib import Path

from toascii import Video
from toascii.converters import ColorConverter, ConverterOptions

CELEBRATIONS_DIR = Path(__file__).parent / "celebrations"
SUPPORTED_EXTENSIONS = {".gif", ".mp4", ".mov", ".avi"}

WIDTH = 80
HEIGHT = 100
X_STRETCH = 2.1
BLUR = 20
CONTRAST = 0.0
SATURATION = 0.0
GRADIENT = " .:-=+*#%@"


def convert_file(path: Path):
    pkl_path = path.with_suffix(".pkl")

    if pkl_path.exists():
        print(f"  skipping {path.name} (already converted)")
        return

    print(f"  converting {path.name}...")
    t0 = time.time()

    options = ConverterOptions(gradient=GRADIENT, width=WIDTH, height=HEIGHT, x_stretch=X_STRETCH, blur=BLUR, contrast=CONTRAST, saturation=SATURATION)
    video = Video(str(path), ColorConverter(options))

    frames = list(video.get_ascii_frames())
    fps = len(frames) / 2

    with open(pkl_path, "wb") as f:
        pickle.dump({"fps": fps, "frames": frames}, f)

    print(f"  done ({len(frames)} frames, {fps:.1f} fps, {time.time() - t0:.1f}s)")


def main():
    for player_dir in sorted(CELEBRATIONS_DIR.iterdir()):
        if not player_dir.is_dir():
            continue

        print(f"\n{player_dir.name}/")

        for file in sorted(player_dir.iterdir()):
            if file.suffix.lower() in SUPPORTED_EXTENSIONS:
                convert_file(file)


if __name__ == "__main__":
    main()
