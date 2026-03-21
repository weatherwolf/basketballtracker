"""
label_shots.py

Label shot datasets produced under work/runs/<batch_id>/frames_batch/<shot_folder>/.

For each shot folder:
- Ensure per-shot ellipse metadata exists under a per-batch ellipse folder
  (passed via --ellipse-meta-dir), seeded from a per-batch global ellipse file
  (passed via --global-ellipse) if missing.
- Generate/overwrite a 60fps preview.mp4 inside the shot folder.
- Optionally open the preview in the default player.
- Prompt for label (goal/miss/unclear/skip).
- If goal/miss, export an mp4 into the per-batch export folder
  (passed via --videos-dir) as goal_<i>.mp4 / miss_<i>.mp4.

Writes outputs under data/ (global, across batches):
- data/shot_labels.csv
- data/shot_labels.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_rel_path(p: Path, root: Path) -> str:
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except Exception:
        return str(p)


def _parse_goal_miss_index(name: str) -> Optional[Tuple[str, int]]:
    """
    Parses either:
      - goal_12.mp4 / miss_12.mp4  (file name)
      - goal_12 / miss_12          (directory name)
    Returns (label, idx) or None.
    """
    m = re.match(r"^(goal|miss)_(\d+)(?:\.mp4)?$", name, re.IGNORECASE)
    if not m:
        return None
    try:
        return m.group(1).lower(), int(m.group(2))
    except Exception:
        return None


def _next_global_goal_miss_index(videos_dir: Path, debug_root: Path) -> int:
    """
    Returns next integer id that is globally unique across BOTH:
      - videos/goal_<id>.mp4 and videos/miss_<id>.mp4
      - videos/debug_frames/goal_<id>/ and videos/debug_frames/miss_<id>/
    """
    best = -1

    if videos_dir.exists():
        for p in videos_dir.iterdir():
            if not p.is_file():
                continue
            parsed = _parse_goal_miss_index(p.name)
            if parsed is None:
                continue
            _lbl, idx = parsed
            best = max(best, idx)

    if debug_root.exists():
        for p in debug_root.iterdir():
            if not p.is_dir():
                continue
            parsed = _parse_goal_miss_index(p.name)
            if parsed is None:
                continue
            _lbl, idx = parsed
            best = max(best, idx)

    return best + 1


def _try_extract_existing_goal_miss_id(rel_export_mp4: str) -> Optional[Tuple[str, int]]:
    """
    Pull (label, id) from an existing rel_export_mp4 path like videos/goal_12.mp4.
    """
    if not rel_export_mp4:
        return None
    try:
        stem = Path(rel_export_mp4).name
    except Exception:
        return None
    return _parse_goal_miss_index(stem)


def _maybe_rename_existing_exports(
    *,
    repo_root: Path,
    videos_dir: Path,
    debug_root: Path,
    existing_rel_export_mp4: str,
    new_label: str,
    existing_id: int,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    If the existing export/debug folder uses the old label (goal vs miss) but the user changed it,
    rename them to the new label while keeping the SAME numeric id.
    Returns (new_mp4_path_or_none, new_debug_dir_or_none).
    """
    new_stem = f"{new_label}_{existing_id}"

    new_mp4_path: Optional[Path] = None
    try:
        old_mp4 = (repo_root / existing_rel_export_mp4).resolve()
    except Exception:
        old_mp4 = None  # type: ignore[assignment]

    if old_mp4 and old_mp4.exists() and old_mp4.is_file():
        candidate = videos_dir / f"{new_stem}{old_mp4.suffix}"
        # If it's already the right name, keep it.
        if old_mp4.name.lower() == candidate.name.lower():
            new_mp4_path = old_mp4
        else:
            # Only rename if it won't clobber something else.
            if not candidate.exists():
                old_mp4.rename(candidate)
                new_mp4_path = candidate

    new_debug_dir: Optional[Path] = None
    old_debug_dir = None
    if debug_root.exists():
        # try both possibilities: directory named by old mp4 stem, or directly by parsed label/id
        old_mp4_stem = Path(existing_rel_export_mp4).stem if existing_rel_export_mp4 else ""
        if old_mp4_stem:
            old_debug_dir = debug_root / old_mp4_stem
        if not old_debug_dir or not old_debug_dir.exists():
            # fallback: scan for any goal_/miss_ dir with the same id
            for p in debug_root.iterdir():
                if not p.is_dir():
                    continue
                parsed = _parse_goal_miss_index(p.name)
                if parsed and parsed[1] == existing_id:
                    old_debug_dir = p
                    break

    if old_debug_dir and old_debug_dir.exists() and old_debug_dir.is_dir():
        candidate = debug_root / new_stem
        if old_debug_dir.name.lower() == candidate.name.lower():
            new_debug_dir = old_debug_dir
        else:
            if not candidate.exists():
                old_debug_dir.rename(candidate)
                new_debug_dir = candidate

    return new_mp4_path, new_debug_dir


def _parse_frame_filename(filename: str):
    """
    Matches:
      <dataset_name>_000000.jpg
    Returns (dataset_name, frame_index)
    """
    stem = Path(filename).stem
    m = re.match(r"^(.*)_(\d+)$", stem)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def _open_file(path: Path) -> None:
    try:
        os.startfile(str(path))  # type: ignore[attr-defined]
    except Exception as e:
        print(f"Could not open: {path} ({e})")


def _prompt_label(default: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    prompt = "Label [g=goal, m=miss, u=unclear, s=skip, q=quit"
    if default:
        prompt += f", enter=keep({default})"
    prompt += "]: "

    while True:
        try:
            raw = input(prompt).strip()
        except EOFError:
            return None, None

        if raw == "" and default is not None:
            return default, None
        if raw.lower() == "q":
            return None, None
        if raw.lower() == "s":
            return "skip", None
        if raw.lower() == "g":
            return "goal", None
        if raw.lower() == "m":
            return "miss", None
        if raw.lower() == "u":
            return "unclear", None
        if raw:
            return raw, None
        print("Invalid input. Try g/m/u/s/q, or type a custom label.")


def discover_shot_dirs(frames_dir: Path) -> List[Path]:
    if not frames_dir.exists():
        return []
    out = [p for p in frames_dir.iterdir() if p.is_dir()]
    out.sort(key=lambda p: p.name.lower())
    return out


def _find_any_frame_file(shot_dir: Path) -> Optional[Path]:
    for p in sorted(shot_dir.iterdir(), key=lambda x: x.name.lower()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            return p
    return None


def _dataset_name_for_shot_dir(shot_dir: Path) -> Optional[str]:
    """
    IMPORTANT: the dataset_name used by downstream code (goal_detector grouping, ellipse meta naming)
    comes from the frame file prefix, not necessarily the folder name.
    """
    p = _find_any_frame_file(shot_dir)
    if p is None:
        return None
    dataset_name, _idx = _parse_frame_filename(p.name)
    return dataset_name


def _ensure_ellipse_meta(
    dataset_name: str,
    ellipse_meta_dir: Path,
    global_ellipse_path: Path,
) -> Path:
    """
    Ensures ellipse_<dataset_name>.json exists inside ellipse_meta_dir.
    If missing, seeds it from global_ellipse_path.
    """
    ellipse_meta_dir.mkdir(parents=True, exist_ok=True)
    dst = ellipse_meta_dir / f"ellipse_{dataset_name}.json"
    if dst.exists():
        return dst

    data = json.loads(global_ellipse_path.read_text(encoding="utf-8"))
    # minimal validation
    e = data.get("ellipse") if isinstance(data, dict) else None
    if not isinstance(e, dict) or "center" not in e or "axes" not in e or "angle" not in e:
        raise ValueError(f"Global ellipse file has unexpected format: {global_ellipse_path}")

    dst.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return dst


def _run_ffmpeg(cmd: List[str]) -> None:
    # show the command only if something fails
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        print("FFMPEG failed.")
        print("Command:", " ".join(cmd))
        if e.stdout:
            print("--- stdout ---")
            print(e.stdout)
        if e.stderr:
            print("--- stderr ---")
            print(e.stderr)
        raise


def make_preview_mp4(shot_dir: Path, dataset_name: str, fps: int, overwrite: bool) -> Path:
    """
    Creates shot_dir/preview.mp4 from shot_dir/<dataset_name>_%06d.jpg (start at 0).
    """
    out = shot_dir / "preview.mp4"
    if out.exists() and not overwrite:
        return out

    # Find extension from any frame
    any_frame = _find_any_frame_file(shot_dir)
    if any_frame is None:
        raise FileNotFoundError(f"No frames found in: {shot_dir}")
    ext = any_frame.suffix.lower()

    # Use sequential pattern for deterministic ordering
    pattern = str(shot_dir / f"{dataset_name}_%06d{ext}")
    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-start_number",
        "0",
        "-i",
        pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(fps),
        str(out),
    ]
    _run_ffmpeg(cmd)
    return out


def export_labeled_mp4(
    shot_dir: Path,
    dataset_name: str,
    videos_dir: Path,
    label: str,
    global_id: Optional[int],
    fps: int,
) -> Optional[Path]:
    if label not in {"goal", "miss"}:
        return None

    if global_id is None:
        raise ValueError("global_id is required when exporting goal/miss mp4")

    out = videos_dir / f"{label}_{global_id}.mp4"
    videos_dir.mkdir(parents=True, exist_ok=True)

    any_frame = _find_any_frame_file(shot_dir)
    if any_frame is None:
        raise FileNotFoundError(f"No frames found in: {shot_dir}")
    ext = any_frame.suffix.lower()
    pattern = str(shot_dir / f"{dataset_name}_%06d{ext}")

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-start_number",
        "0",
        "-i",
        pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(fps),
        str(out),
    ]
    _run_ffmpeg(cmd)
    return out


def export_debug_frames(
    shot_dir: Path,
    debug_root: Path,
    label: str,
    dataset_name: str,
    global_id: Optional[int],
    overwrite: bool,
) -> Optional[Path]:
    """
    Copies the shot folder frames into:
      - For labeled goals/misses: <debug_dir>/<label>_<global_id>/          (globally unique)
      - For other labels:         <debug_dir>/<label>/<dataset_name>/
    This is intentionally "dumb": no ball/hoop tracking, just copying frames for inspection/training.
    """
    if label in {"skip"}:
        return None

    if label in {"goal", "miss"}:
        if global_id is None:
            raise ValueError("global_id is required when exporting goal/miss debug frames")
        dst_dir = debug_root / f"{label}_{global_id}"
    else:
        dst_dir = debug_root / label / dataset_name

    if dst_dir.exists():
        if not overwrite:
            return dst_dir
        shutil.rmtree(dst_dir)

    dst_dir.mkdir(parents=True, exist_ok=True)

    # Copy frames in order and rename them so they match the globally-unique export name.
    # This avoids collisions between different source datasets.
    frame_items: List[Tuple[int, Path]] = []
    for p in shot_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        _ds, idx = _parse_frame_filename(p.name)
        if idx is None:
            continue
        frame_items.append((idx, p))
    frame_items.sort(key=lambda t: t[0])

    if label in {"goal", "miss"}:
        export_stem = f"{label}_{global_id}"
        for new_idx, (_old_idx, src) in enumerate(frame_items):
            dst = dst_dir / f"{export_stem}_{new_idx:06d}{src.suffix.lower()}"
            shutil.copy2(str(src), str(dst))
    else:
        for _old_idx, src in frame_items:
            shutil.copy2(str(src), str(dst_dir / src.name))

    return dst_dir


@dataclass
class ShotLabel:
    dataset_name: str
    rel_shot_dir: str
    label: str
    rel_preview_mp4: str = ""
    rel_export_mp4: str = ""
    ellipse_meta: str = ""
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""
    has_stickers: bool = False


def load_existing(json_path: Path) -> Dict[str, ShotLabel]:
    if not json_path.exists():
        return {}
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return {}
    out: Dict[str, ShotLabel] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        rel_shot_dir = it.get("rel_shot_dir")
        dataset_name = it.get("dataset_name")
        if not isinstance(rel_shot_dir, str) or not rel_shot_dir:
            continue
        if not isinstance(dataset_name, str) or not dataset_name:
            continue
        out[rel_shot_dir] = ShotLabel(
            dataset_name=dataset_name,
            rel_shot_dir=rel_shot_dir,
            label=str(it.get("label", "")),
            rel_preview_mp4=str(it.get("rel_preview_mp4", "")),
            rel_export_mp4=str(it.get("rel_export_mp4", "")),
            ellipse_meta=str(it.get("ellipse_meta", "")),
            notes=str(it.get("notes", "")) if it.get("notes") is not None else "",
            created_at=str(it.get("created_at", "")),
            updated_at=str(it.get("updated_at", "")),
            has_stickers=bool(it.get("has_stickers", False)),
        )
    return out


def write_outputs(csv_path: Path, json_path: Path, labels: Dict[str, ShotLabel], meta: Dict[str, object]) -> None:
    rows = list(labels.values())
    rows.sort(key=lambda r: r.rel_shot_dir.lower())

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "dataset_name",
                "rel_shot_dir",
                "label",
                "rel_preview_mp4",
                "rel_export_mp4",
                "ellipse_meta",
                "notes",
                "created_at",
                "updated_at",
                "has_stickers",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.dataset_name,
                    r.rel_shot_dir,
                    r.label,
                    r.rel_preview_mp4,
                    r.rel_export_mp4,
                    r.ellipse_meta,
                    r.notes,
                    r.created_at,
                    r.updated_at,
                    r.has_stickers,
                ]
            )

    payload = {"meta": meta, "items": [asdict(r) for r in rows]}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Label shot folders under work/frames_batch and export goal/miss mp4s.")
    ap.add_argument("--frames-dir", default="work/frames_batch", help="Directory containing shot folders (default: work/frames_batch)")
    ap.add_argument("--videos-dir", default="media/exports", help="Where to export goal_*/miss_* mp4s (default: media/exports)")
    ap.add_argument("--debug-dir", default="work/debug/labels", help="Where to copy labeled frames (default: work/debug/labels)")
    ap.add_argument("--ellipse-meta-dir", default="data/ball_tracking", help="Ellipse meta output dir (default: data/ball_tracking)")
    ap.add_argument("--global-ellipse", default="assets/hoop_ellipses.json", help="Global ellipse JSON path (default: assets/hoop_ellipses.json)")
    ap.add_argument("--fps", type=int, default=30, help="Preview/export fps (default: 30)")
    ap.add_argument("--export-debug-frames", action="store_true", help="Copy labeled shot frames into debug dir")
    ap.add_argument("--overwrite-debug-frames", action="store_true", help="Overwrite debug frame folders if they exist")
    ap.add_argument("--open", action="store_true", help="Open preview.mp4 for each shot")
    ap.add_argument("--overwrite-preview", action="store_true", help="Overwrite preview.mp4 if it already exists")
    ap.add_argument("--relabel", action="store_true", help="Prompt even if a shot was already labeled")
    ap.add_argument("--list", action="store_true", help="List discovered shot folders and exit")
    ap.add_argument("--out-base", default="data/shot_labels", help="Output base name (default: data/shot_labels)")
    ap.add_argument("--stickers", type=lambda x: x.lower() not in ("false", "0", "no"), default=True,
                    metavar="BOOL", help="Whether stickers are on the rim for this batch (default: true)")
    args = ap.parse_args()

    repo_root = Path(".").resolve()
    frames_dir = (repo_root / args.frames_dir).resolve()
    videos_dir = (repo_root / args.videos_dir).resolve()
    debug_dir = (repo_root / args.debug_dir).resolve()
    ellipse_meta_dir = (repo_root / args.ellipse_meta_dir).resolve()
    global_ellipse_path = (repo_root / args.global_ellipse).resolve()

    if not global_ellipse_path.exists():
        print(f"Global ellipse file not found: {global_ellipse_path}")
        return 2

    shot_dirs = discover_shot_dirs(frames_dir)
    if not shot_dirs:
        print(f"No shot folders found under: {frames_dir}")
        return 1

    if args.list:
        for d in shot_dirs:
            print(_to_rel_path(d, repo_root))
        return 0

    csv_path = repo_root / f"{args.out_base}.csv"
    json_path = repo_root / f"{args.out_base}.json"
    labels_by_rel = load_existing(json_path)
    meta = {
        "generated_at": _now_iso(),
        "frames_dir": str(frames_dir),
        "videos_dir": str(videos_dir),
        "debug_dir": str(debug_dir),
        "ellipse_meta_dir": str(ellipse_meta_dir),
        "global_ellipse": str(global_ellipse_path),
        "fps": args.fps,
    }

    total = len(shot_dirs)
    for idx, shot_dir in enumerate(shot_dirs, start=1):
        rel_shot_dir = _to_rel_path(shot_dir, repo_root)
        existing = labels_by_rel.get(rel_shot_dir)
        if existing and not args.relabel:
            continue

        dataset_name = _dataset_name_for_shot_dir(shot_dir)
        if not dataset_name:
            print(f"Skipping (no frames): {rel_shot_dir}")
            continue

        ellipse_path = _ensure_ellipse_meta(dataset_name, ellipse_meta_dir, global_ellipse_path)
        preview_path = make_preview_mp4(shot_dir, dataset_name, args.fps, overwrite=args.overwrite_preview)

        print()
        print(f"[{idx}/{total}] {rel_shot_dir}")
        if existing:
            print(f"Current label: {existing.label}")

        if args.open:
            _open_file(preview_path)

        default_label = existing.label if existing else None
        label, _ = _prompt_label(default_label)
        if label is None:
            break

        # Allocate a globally unique id shared across both goals and misses.
        # If we're relabeling and the shot already has an export id, keep it.
        existing_parsed = _try_extract_existing_goal_miss_id(existing.rel_export_mp4) if existing else None
        global_id: Optional[int] = None
        if label in {"goal", "miss"}:
            if existing_parsed is not None:
                _old_label, old_id = existing_parsed
                global_id = old_id
                # If the label changed (goal<->miss), rename existing exports to keep id stable.
                if existing and _old_label != label:
                    _maybe_rename_existing_exports(
                        repo_root=repo_root,
                        videos_dir=videos_dir,
                        debug_root=debug_dir,
                        existing_rel_export_mp4=existing.rel_export_mp4,
                        new_label=label,
                        existing_id=old_id,
                    )
            else:
                global_id = _next_global_goal_miss_index(videos_dir, debug_dir)

        export_path = export_labeled_mp4(shot_dir, dataset_name, videos_dir, label, global_id, args.fps)
        debug_export_dir = (
            export_debug_frames(
                shot_dir=shot_dir,
                debug_root=debug_dir,
                label=label,
                dataset_name=dataset_name,
                global_id=global_id,
                overwrite=args.overwrite_debug_frames,
            )
            if args.export_debug_frames
            else None
        )

        now = _now_iso()
        created_at = existing.created_at if existing and existing.created_at else now
        labels_by_rel[rel_shot_dir] = ShotLabel(
            dataset_name=dataset_name,
            rel_shot_dir=rel_shot_dir,
            label=label,
            rel_preview_mp4=_to_rel_path(preview_path, repo_root),
            rel_export_mp4=_to_rel_path(export_path, repo_root) if export_path else "",
            ellipse_meta=_to_rel_path(ellipse_path, repo_root),
            notes="",
            created_at=created_at,
            updated_at=now,
            has_stickers=args.stickers,
        )

        write_outputs(csv_path, json_path, labels_by_rel, meta)

    write_outputs(csv_path, json_path, labels_by_rel, meta)
    print()
    print(f"Wrote: {csv_path.name}")
    print(f"Wrote: {json_path.name}")
    print(f"Labeled shots: {len(labels_by_rel)} (found {total})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

