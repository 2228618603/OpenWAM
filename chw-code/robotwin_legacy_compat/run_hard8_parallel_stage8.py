#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import subprocess
import threading
import time
import shutil
from datetime import datetime
from pathlib import Path


TASKS = [
    "put_bottles_dustbin",
    "open_microwave",
    "stack_blocks_three",
    "stack_bowls_three",
    "blocks_ranking_rgb",
    "hanging_mug",
    "put_object_cabinet",
    "handover_block",
]

CKPTS = {
    "wan21_vace_1_3b": "/mnt/data/chw/model/openwam_ckpt/openwam_study/video_backbone/robotwin_dual_system_joint_self_attention_wan21_vace_1_3b",
    "cosmos25": "/mnt/data/chw/model/openwam_ckpt/openwam_study/video_backbone/robotwin_dual_system_joint_self_attention_cosmos25",
    "cosmos3": "/mnt/data/chw/model/openwam_ckpt/openwam_study/video_backbone/robotwin_dual_system_joint_self_attention_cosmos3",
    "wan22_ti2v_5b": "/mnt/data/chw/model/openwam_ckpt/openwam_study/video_backbone/robotwin_dual_system_joint_self_attention",
    "wan21_i2v_14b": "/mnt/data/chw/model/openwam_ckpt/openwam_study/video_backbone/robotwin_dual_system_joint_self_attention_wan21_i2v_14b",
}


KS = [1, 2, 4, 8, 16, 32]
ROBOTWIN_PATH = Path("/home/chw/code/packages/OpenWAM/chw-code/robotwin_legacy_compat_copy")


def count_records(run_root: Path, backbone: str, task: str) -> int:
    path = run_root / "episodes" / f"{backbone}__{task}.jsonl"
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def write_config(run_root: Path, gpu_slots: list[int], ports: list[int]) -> None:
    config = run_root / "config"
    config.mkdir(parents=True, exist_ok=True)
    (config / "hard8_tasks.txt").write_text("\n".join(TASKS) + "\n", encoding="utf-8")
    (config / "backbone_ckpts.json").write_text(json.dumps(CKPTS, indent=2), encoding="utf-8")
    (config / "gpu_plan.json").write_text(
        json.dumps({"gpu_slots": gpu_slots, "ports": ports}, indent=2),
        encoding="utf-8",
    )
    with (config / "run.env").open("w", encoding="utf-8") as f:
        f.write(f"created_at={datetime.now().isoformat()}\n")
        f.write(f"target_n={os.environ.get('TARGET_N', '32')}\n")
        f.write(f"task_config={os.environ.get('TASK_CONFIG', 'demo_clean')}\n")


def parse_gpu_slots(value: str) -> list[int]:
    slots = [int(x) for x in value.split(",") if x.strip()]
    if not slots:
        raise ValueError("gpu slot list is empty")
    return slots


def parse_gpu_slot_overrides(values: list[str]) -> dict[str, list[int]]:
    overrides: dict[str, list[int]] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"invalid --gpu-slots-override value: {item!r}; expected BACKBONE=GPU,GPU")
        backbone, slots = item.split("=", 1)
        backbone = backbone.strip()
        if backbone not in CKPTS:
            raise ValueError(f"unknown backbone in --gpu-slots-override: {backbone}")
        overrides[backbone] = parse_gpu_slots(slots)
    return overrides


def summarize_status(run_root: Path, target_n: int) -> dict:
    rows = []
    done = 0
    for backbone in CKPTS:
        for task in TASKS:
            n = count_records(run_root, backbone, task)
            done += int(n >= target_n)
            rows.append({"backbone": backbone, "task": task, "records": n, "target": target_n})
    status = {"done_jobs": done, "total_jobs": len(CKPTS) * len(TASKS), "rows": rows}
    status_path = run_root / "status.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    csv_path = run_root / "status.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["backbone", "task", "records", "target"])
        writer.writeheader()
        writer.writerows(rows)
    return status


def run_stage9_summary(run_root: Path) -> None:
    script = Path("/home/chw/code/packages/OpenWAM/chw-code/robotwin_legacy_compat/summarize_hard8_passk.py")
    subprocess.run([str(script), str(run_root)], check=True)


def collect_success_videos(run_root: Path) -> list[dict]:
    video_dir = run_root / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for task in TASKS:
        candidates = []
        for backbone in CKPTS:
            root = (
                ROBOTWIN_PATH
                / "eval_result"
                / task
                / "openwam2robotwin_interface"
                / "demo_clean"
                / f"hard8_{backbone}"
            )
            candidates.extend(root.glob("*/episode*_success-true.mp4"))
        if not candidates:
            copied.append({"task": task, "status": "missing", "source": ""})
            continue
        source = max(candidates, key=lambda p: p.stat().st_mtime)
        dest = video_dir / f"{task}.mp4"
        shutil.copy2(source, dest)
        copied.append({"task": task, "status": "copied", "source": str(source), "dest": str(dest)})

    report = run_root / "videos" / "video_sources.json"
    report.write_text(json.dumps(copied, indent=2, ensure_ascii=False), encoding="utf-8")
    return copied


def load_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_stage9_note(run_root: Path) -> None:
    summaries = run_root / "summaries"
    notes = run_root / "notes"
    notes.mkdir(parents=True, exist_ok=True)
    averages = load_json(summaries / "backbone_average_passk.json")
    per_task = load_json(summaries / "per_task_passk.json")
    videos = load_json(run_root / "videos" / "video_sources.json")

    lines = [
        "# Stage 9 Hard8 Analysis",
        "",
        f"Run root: `{run_root}`",
        "",
        "## Backbone Average",
        "",
        "| backbone | episodes | successes | success_rate | success@1 | success@2 | success@4 | success@8 | success@16 | success@32 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in averages:
        lines.append(
            "| {backbone} | {episodes} | {successes} | {success_rate:.4f} | {s1:.4f} | {s2:.4f} | {s4:.4f} | {s8:.4f} | {s16:.4f} | {s32:.4f} |".format(
                backbone=row["backbone"],
                episodes=int(row["episodes"]),
                successes=int(row["successes"]),
                success_rate=float(row["success_rate"]),
                s1=float(row["success@1"]),
                s2=float(row["success@2"]),
                s4=float(row["success@4"]),
                s8=float(row["success@8"]),
                s16=float(row["success@16"]),
                s32=float(row["success@32"]),
            )
        )

    lines.extend(["", "## Task Notes", ""])
    task_rows: dict[str, list[dict]] = {}
    for row in per_task:
        task_rows.setdefault(row["task"], []).append(row)
    for task in TASKS:
        rows = task_rows.get(task, [])
        solved = [r["backbone"] for r in rows if int(r.get("success@32", 0)) > 0]
        if not rows:
            note = "missing evaluation rows"
        elif not solved:
            note = "no backbone solved within 32 trials"
        elif len(solved) == len(CKPTS):
            note = "all backbones solved within 32 trials"
        else:
            note = "solved by: " + ", ".join(solved)
        lines.append(f"- `{task}`: {note}")

    lines.extend(["", "## Videos", ""])
    for item in videos:
        if item["status"] == "copied":
            lines.append(f"- `{item['task']}`: `{item['dest']}`")
        else:
            lines.append(f"- `{item['task']}`: missing OpenWAM success video")

    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- Treat this as an empirical smoke-test style comparison, not a causal proof.",
            "- If large backbones improve small-k success but converge by success@32, the main effect is sampling efficiency.",
            "- If large backbones solve tasks that smaller backbones never solve by success@32, that suggests a stronger capability boundary.",
            "- If most entries are all-zero or all-one, add a medium-difficulty task slice before drawing backbone conclusions.",
        ]
    )
    (notes / "stage9-hard8-analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_job_workers(
    *,
    run_root: Path,
    jobs_to_run: list[tuple[str, str]],
    gpu_slots: list[int],
    base_port: int,
    stop_on_error: bool,
    target_n: int,
    worker_offset: int,
) -> None:
    q: "queue.Queue[tuple[str, str]]" = queue.Queue()
    for job in jobs_to_run:
        q.put(job)

    ports = [base_port + i for i in range(len(gpu_slots))]
    threads = []
    for i, (gpu, port) in enumerate(zip(gpu_slots, ports)):
        t = threading.Thread(
            target=worker_loop,
            args=(worker_offset + i, gpu, port, run_root, q, stop_on_error, target_n),
            daemon=False,
        )
        t.start()
        threads.append(t)

    while any(t.is_alive() for t in threads):
        status = summarize_status(run_root, target_n)
        print(
            f"[monitor] {datetime.now().isoformat(timespec='seconds')} "
            f"{status['done_jobs']}/{status['total_jobs']} jobs complete",
            flush=True,
        )
        time.sleep(60)

    for t in threads:
        t.join()
    summarize_status(run_root, target_n)


def worker_loop(
    worker_id: int,
    gpu: int,
    port: int,
    run_root: Path,
    jobs: "queue.Queue[tuple[str, str]]",
    stop_on_error: bool,
    target_n: int,
) -> None:
    script = Path("/home/chw/code/packages/OpenWAM/chw-code/robotwin_legacy_compat/run_one_hard8_job.sh")
    worker_log = run_root / "logs" / f"worker_{worker_id}.log"
    worker_log.parent.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            backbone, task = jobs.get_nowait()
        except queue.Empty:
            return

        if count_records(run_root, backbone, task) >= target_n:
            jobs.task_done()
            continue

        cmd = [str(script), str(run_root), backbone, task, str(gpu), str(port), CKPTS[backbone]]
        started = datetime.now().isoformat(timespec="seconds")
        with worker_log.open("a", encoding="utf-8") as log:
            log.write(f"\n[{started}] worker={worker_id} gpu={gpu} port={port} start {backbone} {task}\n")
            log.flush()
            env = os.environ.copy()
            env["TARGET_N"] = str(target_n)
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            ended = datetime.now().isoformat(timespec="seconds")
            log.write(f"[{ended}] worker={worker_id} exit={proc.returncode} {backbone} {task}\n")

        summarize_status(run_root, target_n)
        jobs.task_done()
        if proc.returncode != 0 and stop_on_error:
            raise SystemExit(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default="/home/chw/code/packages/OpenWAM/chw-code/robotwin_legacy_compat/result")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--gpu-slots", default="1,2,7")
    parser.add_argument(
        "--gpu-slots-override",
        action="append",
        default=[],
        help="Per-backbone slot override, e.g. wan21_i2v_14b=1,2,7. Can be repeated.",
    )
    parser.add_argument("--base-port", type=int, default=9100)
    parser.add_argument("--target-n", type=int, default=32)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--phase-by-backbone",
        action="store_true",
        help="Run one backbone at a time, using --gpu-slots-override where provided.",
    )
    args = parser.parse_args()

    run_name = args.run_name or f"hard8_stage8_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root = Path(args.result_root) / run_name
    (run_root / "logs").mkdir(parents=True, exist_ok=True)
    (run_root / "episodes").mkdir(parents=True, exist_ok=True)
    (run_root / "summaries").mkdir(parents=True, exist_ok=True)
    (run_root / "videos").mkdir(parents=True, exist_ok=True)
    (run_root / "notes").mkdir(parents=True, exist_ok=True)

    gpu_slots = parse_gpu_slots(args.gpu_slots)
    gpu_slot_overrides = parse_gpu_slot_overrides(args.gpu_slots_override)
    ports = [args.base_port + i for i in range(len(gpu_slots))]
    write_config(run_root, gpu_slots, ports)
    if gpu_slot_overrides:
        (run_root / "config" / "gpu_slot_overrides.json").write_text(
            json.dumps(gpu_slot_overrides, indent=2),
            encoding="utf-8",
        )

    summarize_status(run_root, args.target_n)

    if args.phase_by_backbone:
        for phase_idx, backbone in enumerate(CKPTS):
            phase_slots = gpu_slot_overrides.get(backbone, gpu_slots)
            phase_jobs = [(backbone, task) for task in TASKS]
            print(
                f"[phase] backbone={backbone} slots={','.join(str(x) for x in phase_slots)}",
                flush=True,
            )
            run_job_workers(
                run_root=run_root,
                jobs_to_run=phase_jobs,
                gpu_slots=phase_slots,
                base_port=args.base_port + phase_idx * 100,
                stop_on_error=args.stop_on_error,
                target_n=args.target_n,
                worker_offset=phase_idx * 100,
            )
    else:
        jobs = [(backbone, task) for backbone in CKPTS for task in TASKS]
        run_job_workers(
            run_root=run_root,
            jobs_to_run=jobs,
            gpu_slots=gpu_slots,
            base_port=args.base_port,
            stop_on_error=args.stop_on_error,
            target_n=args.target_n,
            worker_offset=0,
        )
    print("[stage9] summarizing pass@k", flush=True)
    run_stage9_summary(run_root)
    print("[stage9] collecting success videos", flush=True)
    collect_success_videos(run_root)
    print("[stage9] writing analysis note", flush=True)
    write_stage9_note(run_root)
    print(f"[done] run_root={run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
