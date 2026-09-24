#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


KS = [1, 2, 4, 8, 16, 32]


def load_rows(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def success_at(rows: list[dict], k: int) -> int:
    return int(any(bool(r.get("success")) for r in rows[:k]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    ep_dir = run_root / "episodes"
    out_dir = run_root / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_task = []
    for path in sorted(ep_dir.glob("*.jsonl")):
        stem = path.stem
        if "__" not in stem:
            continue
        backbone, task = stem.split("__", 1)
        rows = load_rows(path)[:32]
        record = {
            "backbone": backbone,
            "task": task,
            "episodes": len(rows),
            "successes": sum(int(bool(r.get("success"))) for r in rows),
            "success_rate": (sum(int(bool(r.get("success"))) for r in rows) / len(rows)) if rows else 0.0,
        }
        for k in KS:
            record[f"success@{k}"] = success_at(rows, k)
        per_task.append(record)

    with (out_dir / "per_task_passk.json").open("w", encoding="utf-8") as f:
        json.dump(per_task, f, indent=2, ensure_ascii=False)
    with (out_dir / "per_task_passk.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["backbone", "task", "episodes", "successes", "success_rate"] + [f"success@{k}" for k in KS]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_task)

    by_backbone: dict[str, list[dict]] = {}
    for row in per_task:
        by_backbone.setdefault(row["backbone"], []).append(row)

    averages = []
    for backbone, rows in sorted(by_backbone.items()):
        avg = {
            "backbone": backbone,
            "tasks": len(rows),
            "episodes": sum(int(r["episodes"]) for r in rows),
            "successes": sum(int(r["successes"]) for r in rows),
        }
        avg["success_rate"] = avg["successes"] / avg["episodes"] if avg["episodes"] else 0.0
        for k in KS:
            avg[f"success@{k}"] = sum(float(r[f"success@{k}"]) for r in rows) / len(rows) if rows else 0.0
        averages.append(avg)

    with (out_dir / "backbone_average_passk.json").open("w", encoding="utf-8") as f:
        json.dump(averages, f, indent=2, ensure_ascii=False)
    with (out_dir / "backbone_average_passk.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["backbone", "tasks", "episodes", "successes", "success_rate"] + [f"success@{k}" for k in KS]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(averages)

    print(f"wrote {out_dir / 'per_task_passk.csv'}")
    print(f"wrote {out_dir / 'backbone_average_passk.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
