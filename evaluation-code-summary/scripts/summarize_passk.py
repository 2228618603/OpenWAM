#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


KS = (1, 2, 4, 8, 16, 32)


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: int(r.get("episode_id", 0)))
    return records


def passk(successes: list[bool]) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    for k in KS:
        out[f"success@{k}"] = int(any(successes[:k])) if len(successes) >= k else None
    return out


def canonical_name(path: Path) -> str:
    name = path.name.removesuffix("_episodes.jsonl")
    for suffix in ("_resume", "_part2", "_part3"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} RUN_ROOT", file=sys.stderr)
        return 2

    run_root = Path(sys.argv[1]).expanduser().resolve()
    grouped: dict[str, list[dict]] = {}
    for path in sorted(run_root.glob("*_episodes.jsonl")):
        grouped.setdefault(canonical_name(path), []).extend(load_records(path))

    rows = []
    for name, records in sorted(grouped.items()):
        successes = [bool(r.get("success")) for r in records]
        row = {
            "backbone": name,
            "episodes": len(records),
            "successes": sum(successes),
            "success_rate": (sum(successes) / len(successes)) if successes else None,
        }
        row.update(passk(successes))
        rows.append(row)

    if not rows:
        print(f"no *_episodes.jsonl files found under {run_root}", file=sys.stderr)
        return 1

    summary_json = run_root / "passk_summary.json"
    summary_csv = run_root / "passk_summary.csv"
    summary_json.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fieldnames = ["backbone", "episodes", "successes", "success_rate", *(f"success@{k}" for k in KS)]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(rows, indent=2, sort_keys=True))
    print(f"wrote {summary_json}")
    print(f"wrote {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
