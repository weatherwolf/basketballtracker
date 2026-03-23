"""
shot_io.py

Shared data structures and file I/O for shot labeling.
Copied from dev/label_shots.py with minimal changes for live use.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Copied verbatim from label_shots.py
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_rel_path(p: Path, root: Path) -> str:
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except Exception:
        return str(p)


def _parse_goal_miss_index(name: str) -> Optional[Tuple[str, int]]:
    m = re.match(r"^(goal|miss)_(\d+)(?:\.mp4)?$", name, re.IGNORECASE)
    if not m:
        return None
    try:
        return m.group(1).lower(), int(m.group(2))
    except Exception:
        return None


def _try_extract_existing_goal_miss_id(rel_export_mp4: str) -> Optional[Tuple[str, int]]:
    if not rel_export_mp4:
        return None
    try:
        stem = Path(rel_export_mp4).name
    except Exception:
        return None
    return _parse_goal_miss_index(stem)


def _parse_frame_filename(filename: str):
    stem = Path(filename).stem
    m = re.match(r"^(.*)_(\d+)$", stem)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def _find_any_frame_file(shot_dir: Path) -> Optional[Path]:
    for p in sorted(shot_dir.iterdir(), key=lambda x: x.name.lower()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            return p
    return None


def _next_global_goal_miss_index(videos_dir: Path) -> int:
    best = -1
    if videos_dir.exists():
        for p in videos_dir.iterdir():
            if not p.is_file():
                continue
            parsed = _parse_goal_miss_index(p.name)
            if parsed is None:
                continue
            best = max(best, parsed[1])
    return best + 1


def _run_ffmpeg(cmd: List[str]) -> None:
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
    out = shot_dir / "preview.mp4"
    if out.exists() and not overwrite:
        return out
    any_frame = _find_any_frame_file(shot_dir)
    if any_frame is None:
        raise FileNotFoundError(f"No frames found in: {shot_dir}")
    ext = any_frame.suffix.lower()
    pattern = str(shot_dir / f"{dataset_name}_%06d{ext}")
    cmd = [
        "ffmpeg", "-y" if overwrite else "-n",
        "-hide_banner", "-loglevel", "error",
        "-framerate", str(fps), "-start_number", "0",
        "-i", pattern,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-r", str(fps), str(out),
    ]
    _run_ffmpeg(cmd)
    return out


def export_labeled_mp4(
    shot_dir: Path, dataset_name: str, videos_dir: Path,
    label: str, global_id: int, fps: int,
) -> Optional[Path]:
    if label not in {"goal", "miss"}:
        return None
    out = videos_dir / f"{label}_{global_id}.mp4"
    videos_dir.mkdir(parents=True, exist_ok=True)
    any_frame = _find_any_frame_file(shot_dir)
    if any_frame is None:
        raise FileNotFoundError(f"No frames found in: {shot_dir}")
    ext = any_frame.suffix.lower()
    pattern = str(shot_dir / f"{dataset_name}_%06d{ext}")
    cmd = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        "-framerate", str(fps), "-start_number", "0",
        "-i", pattern,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-r", str(fps), str(out),
    ]
    _run_ffmpeg(cmd)
    return out


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


def write_outputs(
    csv_path: Path, json_path: Path,
    labels: Dict[str, ShotLabel], meta: Dict[str, object],
) -> None:
    rows = list(labels.values())
    rows.sort(key=lambda r: r.rel_shot_dir.lower())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset_name", "rel_shot_dir", "label", "rel_preview_mp4",
            "rel_export_mp4", "ellipse_meta", "notes", "created_at", "updated_at", "has_stickers",
        ])
        for r in rows:
            w.writerow([
                r.dataset_name, r.rel_shot_dir, r.label, r.rel_preview_mp4,
                r.rel_export_mp4, r.ellipse_meta, r.notes, r.created_at, r.updated_at,
                r.has_stickers,
            ])
    payload = {"meta": meta, "items": [asdict(r) for r in rows]}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Adapted from label_shots.py: single-keypress version of _prompt_label
# ---------------------------------------------------------------------------

def prompt_label_cbreak(default: Optional[str] = None) -> Optional[str]:
    """Single-keypress label prompt. Terminal must already be in cbreak mode.
    Returns the chosen label, or None if the user pressed 'q'.
    """
    hint = f"  enter=keep({default})" if default else ""
    print(f"\n  Label [g=goal  m=miss  u=unclear  s=skip  q=quit{hint}]: ", end="", flush=True)
    while True:
        ch = sys.stdin.read(1)
        if ch == "g":
            print("goal", flush=True)
            return "goal"
        elif ch == "m":
            print("miss", flush=True)
            return "miss"
        elif ch == "u":
            print("unclear", flush=True)
            return "unclear"
        elif ch == "s":
            print("skip", flush=True)
            return "skip"
        elif ch == "q":
            print("quit", flush=True)
            return None
        elif ch in ("\r", "\n") and default is not None:
            print(default, flush=True)
            return default


# ---------------------------------------------------------------------------
# New: accumulates labels for a live session and writes them to disk on close
# ---------------------------------------------------------------------------

class LabelSession:
    def __init__(self, repo_root: Path, batch_id: str, has_stickers: bool = True):
        self.repo_root  = repo_root
        self.batch_id   = batch_id
        self.has_stickers = has_stickers
        self.videos_dir = repo_root / "media" / "exports" / batch_id
        self.json_path  = repo_root / "data" / "shot_labels.json"
        self.csv_path   = repo_root / "data" / "shot_labels.csv"
        self._labels    = {}   # start fresh each session; merging into historical data happens on the Windows side
        self._lock      = threading.Lock()
        self.videos_dir.mkdir(parents=True, exist_ok=True)

    def record(self, shot_dir: Path, dataset_name: str, label: str) -> None:
        """Save a label for a shot and export the mp4. Thread-safe."""
        if label not in ("goal", "miss"):
            return
        rel_shot_dir = _to_rel_path(shot_dir, self.repo_root)
        with self._lock:
            existing      = self._labels.get(rel_shot_dir)
            existing_parsed = _try_extract_existing_goal_miss_id(
                existing.rel_export_mp4 if existing else ""
            )
            global_id = existing_parsed[1] if existing_parsed else _next_global_goal_miss_index(self.videos_dir)
            try:
                export_path = export_labeled_mp4(
                    shot_dir, dataset_name, self.videos_dir, label, global_id, fps=30
                )
            except Exception as e:
                print(f"\n  [warning] mp4 export failed: {e}")
                export_path = None
            now = _now_iso()
            self._labels[rel_shot_dir] = ShotLabel(
                dataset_name=dataset_name,
                rel_shot_dir=rel_shot_dir,
                label=label,
                rel_preview_mp4="",
                rel_export_mp4=_to_rel_path(export_path, self.repo_root) if export_path else "",
                ellipse_meta="",
                notes="",
                created_at=existing.created_at if existing and existing.created_at else now,
                updated_at=now,
                has_stickers=self.has_stickers,
            )

    def flip_last(self) -> tuple | None:
        """Flip the most recent goal/miss label. Returns (old, new) or None if nothing to flip."""
        with self._lock:
            if not self._labels:
                return None
            last_key = list(self._labels.keys())[-1]
            entry = self._labels[last_key]
            old = entry.label
            if old == "goal":
                new = "miss"
            elif old == "miss":
                new = "goal"
            else:
                return None
            # Rename the exported mp4 to match the new label
            new_export = entry.rel_export_mp4
            if entry.rel_export_mp4:
                old_path = self.repo_root / entry.rel_export_mp4
                if old_path.exists():
                    new_path = old_path.with_name(old_path.name.replace(old, new, 1))
                    old_path.rename(new_path)
                    new_export = _to_rel_path(new_path, self.repo_root)

            self._labels[last_key] = ShotLabel(
                dataset_name=entry.dataset_name,
                rel_shot_dir=entry.rel_shot_dir,
                label=new,
                rel_preview_mp4=entry.rel_preview_mp4,
                rel_export_mp4=new_export,
                ellipse_meta=entry.ellipse_meta,
                notes=entry.notes,
                created_at=entry.created_at,
                updated_at=_now_iso(),
                has_stickers=entry.has_stickers,
            )
            return (old, new)

    def save(self) -> int:
        """Write all labels to shot_labels.json/.csv. Returns total label count."""
        with self._lock:
            meta = {"generated_at": _now_iso(), "batch_id": self.batch_id}
            write_outputs(self.csv_path, self.json_path, self._labels, meta)
            return len(self._labels)
