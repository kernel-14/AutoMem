"""Export AppWorld task metadata to the AutoMem JSONL dataset contract.

The runner consumes a JSONL file whose rows are 1-indexed by file order. Each
row must carry at least `task_id` and `instruction`. We additionally export
split, difficulty, and app statistics so the search engine's data splits and
the observation graph can stratify by them.

Usage:
    python -m automem.benchmarks.appworld.prepare_data \
        --splits train dev --outfile data/appworld/appworld_tasks.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _appworld_data_root() -> Path:
    from appworld.common.path_store import path_store

    return Path(path_store.data)


def load_task_metadata(task_id: str, data_root: Path) -> dict:
    task_dir = data_root / "tasks" / task_id
    specs = json.loads((task_dir / "specs.json").read_text(encoding="utf-8"))
    gt_meta_path = task_dir / "ground_truth" / "metadata.json"
    gt_meta = (
        json.loads(gt_meta_path.read_text(encoding="utf-8"))
        if gt_meta_path.is_file()
        else {}
    )
    return {
        "task_id": task_id,
        "instruction": specs["instruction"],
        "world_datetime": specs.get("datetime"),
        "difficulty": gt_meta.get("difficulty"),
        "num_apps": gt_meta.get("num_apps"),
        "num_apis": gt_meta.get("num_apis"),
    }


def export_split(split: str, data_root: Path) -> list[dict]:
    from appworld import load_task_ids

    task_ids = load_task_ids(split)
    rows = []
    for task_id in task_ids:
        row = load_task_metadata(task_id, data_root)
        row["split"] = split
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        help="AppWorld splits to export (train/dev/test_normal/test_challenge)",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        required=True,
        help="Output JSONL path",
    )
    args = parser.parse_args()

    data_root = _appworld_data_root()
    if not (data_root / "tasks").is_dir():
        raise FileNotFoundError(
            f"AppWorld data not found under {data_root}. Run `appworld download data` first."
        )

    rows: list[dict] = []
    for split in args.splits:
        split_rows = export_split(split, data_root)
        print(f"{split}: {len(split_rows)} tasks")
        rows.extend(split_rows)

    outfile = Path(args.outfile).expanduser()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} tasks to {outfile}")


if __name__ == "__main__":
    main()
