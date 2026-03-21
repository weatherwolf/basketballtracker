"""
pull_merge.py

Called by pull_livestream_data.bat.
- Creates per-shot ellipse files from the batch global ellipse
- Merges live shot_labels_live_tmp.json into data/shot_labels.json
- Regenerates data/shot_labels.csv
"""

import csv
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    if len(sys.argv) < 2:
        print("Usage: pull_merge.py <batch_id>")
        sys.exit(1)

    batch_id  = sys.argv[1]
    global_e  = REPO / "assets" / "hoop_ellipses" / batch_id / "global.json"
    per_shot  = REPO / "assets" / "hoop_ellipses" / batch_id / "per_shot"
    per_shot.mkdir(parents=True, exist_ok=True)

    live_path = REPO / "data" / "shot_labels_live_tmp.json"
    live_data = json.loads(live_path.read_text(encoding="utf-8"))
    live_items = {i["rel_shot_dir"]: i for i in live_data.get("items", [])}

    for item in live_items.values():
        ds = item.get("dataset_name", "")
        if not ds:
            continue
        dst = per_shot / f"ellipse_{ds}.json"
        shutil.copy2(str(global_e), str(dst))
        item["ellipse_meta"] = dst.relative_to(REPO).as_posix()

    local_path = REPO / "data" / "shot_labels.json"
    if local_path.exists():
        local_data  = json.loads(local_path.read_text(encoding="utf-8"))
        local_items = {i["rel_shot_dir"]: i for i in local_data.get("items", [])}
    else:
        local_items = {}
        local_data  = {}

    local_items.update(live_items)
    local_data["items"] = sorted(
        local_items.values(), key=lambda r: r.get("rel_shot_dir", "").lower()
    )
    local_path.write_text(json.dumps(local_data, indent=2), encoding="utf-8")

    fields = [
        "dataset_name", "rel_shot_dir", "label", "rel_preview_mp4",
        "rel_export_mp4", "ellipse_meta", "notes", "created_at", "updated_at", "has_stickers",
    ]
    with open(REPO / "data" / "shot_labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(local_data["items"])

    live_path.unlink()
    print(f"Merged {len(live_items)} live shots — total: {len(local_items)}")


if __name__ == "__main__":
    main()
