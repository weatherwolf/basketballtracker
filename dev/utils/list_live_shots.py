"""
list_live_shots.py

Outputs the shot folder names (e.g. shot_000001) that are labeled in
data/shot_labels_live_tmp.json for a given batch, one per line.
Used by pull_livestream_data.bat to pull only stage-2 (close-to-rim) shots.
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import REPO_ROOT as REPO


def main():
    if len(sys.argv) < 2:
        print("Usage: list_live_shots.py <batch_id>", file=sys.stderr)
        sys.exit(1)

    batch_id = sys.argv[1]
    tmp = REPO / "data" / "shot_labels_live_tmp.json"
    data = json.loads(tmp.read_text(encoding="utf-8"))
    for item in data.get("items", []):
        rel = item.get("rel_shot_dir", "")
        if batch_id in rel:
            print(Path(rel).name)


if __name__ == "__main__":
    main()
