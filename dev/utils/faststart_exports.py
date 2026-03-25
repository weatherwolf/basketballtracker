"""
faststart_exports.py

Re-muxes all goal/miss export mp4s in media/exports/ to move the moov atom
to the front of the file (-movflags +faststart). No re-encoding — video
quality is unchanged. Fixes the ~10 second delay when opening clips in
review_wrong_predictions.py.

Usage (from repo root):
    python dev/utils/faststart_exports.py
    python dev/utils/faststart_exports.py --dry-run
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import EXPORTS_DIR


def needs_faststart(path: Path) -> bool:
    """Return True if the moov atom is NOT at the start of the file."""
    result = subprocess.run(
        ["ffprobe", "-v", "trace", "-i", str(path)],
        capture_output=True, text=True,
    )
    output = result.stderr
    moov_pos = output.find("moov")
    mdat_pos = output.find("mdat")
    if moov_pos == -1 or mdat_pos == -1:
        return False
    return mdat_pos < moov_pos


def remux(path: Path) -> bool:
    """Re-mux a single file in-place. Returns True on success."""
    tmp = path.with_suffix(".faststart_tmp.mp4")
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(path),
                "-c", "copy",
                "-movflags", "+faststart",
                str(tmp),
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr.strip()}")
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(path)
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        tmp.unlink(missing_ok=True)
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Show what would be done without changing files")
    args = ap.parse_args()

    mp4s = sorted(EXPORTS_DIR.rglob("*.mp4"))
    if not mp4s:
        print(f"No mp4 files found under {EXPORTS_DIR}")
        return

    print(f"Found {len(mp4s)} mp4 files under {EXPORTS_DIR}")
    if args.dry_run:
        print("Dry run — no files will be changed.\n")

    fixed = skipped = failed = 0
    for i, path in enumerate(mp4s, 1):
        rel = path.relative_to(EXPORTS_DIR)
        if not needs_faststart(path):
            print(f"[{i}/{len(mp4s)}] skip (already faststart)  {rel}")
            skipped += 1
            continue

        if args.dry_run:
            print(f"[{i}/{len(mp4s)}] would fix  {rel}")
            fixed += 1
            continue

        print(f"[{i}/{len(mp4s)}] fixing  {rel} ... ", end="", flush=True)
        if remux(path):
            print("done")
            fixed += 1
        else:
            print("FAILED")
            failed += 1

    print(f"\nDone. fixed={fixed}  skipped={skipped}  failed={failed}")


if __name__ == "__main__":
    main()
