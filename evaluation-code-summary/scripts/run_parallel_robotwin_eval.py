#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path


KS = [1, 2, 4, 8, 16, 32]


def load_tasks(path: Path) -> list[str]:
    tasks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            tasks.append(line)
    if not tasks:
        raise ValueError(f"no tasks found in {path}")
    return tasks


def load_ckpts(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{path} must be a non-empty JSON object")
    return {str(k): str(v) for k, v in data.items()}


def count_records(run_root: Path, backbone: str, task: str) -> int:
    path = run_root / "episodes" / f"{backbone}__{task}.jsonl"
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def write_config(
    run_root: Path,
    *,
    tasks: list[str],
    ckpts: dict[str, str],
    gpu_slots: list[int],
    ports: list[int],
    target_n: int,
) -> None:
    config = run_root / "config"
    config.mkdir(parents=True, exist_ok=True)
    (config / "tasks.txt").write_text("\n".join(tasks) + "\n", encoding="utf-8")
    (config / "backbone_ckpts.json").write_text(json.dumps(ckpts, indent=2), encoding="utf-8")
    (config / "gpu_plan.json").write_text(
        json.dumps({"gpu_slots": gpu_slots, "ports": ports}, indent=2),
        encoding="utf-8",
    )
    with (config / "run.env").open("w", encoding="utf-8") as f:
        f.write(f"created_at={datetime.now().isoformat()}\n")
        f.write(f"target_n={target_n}\n")
        f.write(f"task_config={os.environ.get('TASK_CONFIG', 'demo_clean')}\n")
        for key in ("OPENWAM_ROOT", "ROBOTWIN_PATH", "ROBOTWIN_PYTHON"):
            f.write(f"{key}={os.environ.get(key, '')}\n")


def parse_gpu_slots(value: str) -> list[int]:
    slots = [int(x) for x in value.split(",") if x.strip()]
    if not slots:
        raise ValueError("gpu slot list is empty")
    return slots


def parse_gpu_slot_overrides(values: list[str], ckpts: dict[str, str]) -> dict[str, list[int]]:
    overrides: dict[str, list[int]] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"invalid --gpu-slots-override value: {item!r}; expected BACKBONE=GPU,GPU")
        backbone, slots = item.split("=", 1)
        backbone = backbone.strip()
        if backbone not in ckpts:
            raise ValueError(f"unknown backbone in --gpu-slots-override: {backbone}")
        overrides[backbone] = parse_gpu_slots(slots)
    return overrides


def summarize_status(run_root: Path, *, ckpts: dict[str, str], tasks: list[str], target_n: int) -> dict:
    rows = []
    done = 0
    for backbone in ckpts:
        for task in tasks:
            n = count_records(run_root, backbone, task)
            done += int(n >= target_n)
            rows.append({"backbone": backbone, "task": task, "records": n, "target": target_n})
    status = {"done_jobs": done, "total_jobs": len(ckpts) * len(tasks), "rows": rows}
    (run_root / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    with (run_root / "status.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["backbone", "task", "records", "target"])
        writer.writeheader()
        writer.writerows(rows)
    return status


def run_summary(run_root: Path, summary_script: Path) -> None:
    subprocess.run([str(summary_script), str(run_root)], check=True)


def collect_success_videos(run_root: Path, *, robotwin_path: Path, ckpts: dict[str, str], tasks: list[str]) -> list[dict]:
    video_dir = run_root / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    task_config = os.environ.get("TASK_CONFIG", "demo_clean")
    for task in tasks:
        candidates = []
        for backbone in ckpts:
            root = (
                robotwin_path
                / "eval_result"
                / task
                / "openwam2robotwin_interface"
                / task_config
                / f"{os.environ.get('RUN_LABEL_PREFIX', 'eval')}_{backbone}"
            )
            candidates.extend(root.glob("*/episode*_success-true.mp4"))
        if not candidates:
            copied.append({"task": task, "status": "missing", "source": ""})
            continue
        source = max(candidates, key=lambda p: p.stat().st_mtime)
        dest = video_dir / f"{task}.mp4"
        shutil.copy2(source, dest)
        copied.append({"task": task, "status": "copied", "source": str(source), "dest": str(dest)})

    report = video_dir / "video_sources.json"
    report.write_text(json.dumps(copied, indent=2, ensure_ascii=False), encoding="utf-8")
    return copied


def run_job_workers(
    *,
    run_root: Path,
    jobs_to_run: list[tuple[str, str]],
    ckpts: dict[str, str],
    tasks: list[str],
    gpu_slots: list[int],
    base_port: int,
    stop_on_error: bool,
    target_n: int,
    worker_offset: int,
    retry_incomplete_rounds: int,
    job_script: Path,
) -> None:
    ports = [base_port + i for i in range(len(gpu_slots))]
    rounds = 0
    while True:
        incomplete = [
            (backbone, task)
            for backbone, task in jobs_to_run
            if count_records(run_root, backbone, task) < target_n
        ]
        if not incomplete:
            summarize_status(run_root, ckpts=ckpts, tasks=tasks, target_n=target_n)
            return
        if rounds >= retry_incomplete_rounds:
            missing = ", ".join(
                f"{backbone}__{task}:{count_records(run_root, backbone, task)}/{target_n}"
                for backbone, task in incomplete
            )
            raise RuntimeError(f"incomplete jobs after {rounds} rounds: {missing}")

        rounds += 1
        print(
            f"[round] {datetime.now().isoformat(timespec='seconds')} "
            f"round={rounds}/{retry_incomplete_rounds} incomplete_jobs={len(incomplete)}",
            flush=True,
        )

        q: "queue.Queue[tuple[str, str]]" = queue.Queue()
        for job in incomplete:
            q.put(job)

        threads = []
        for i, (gpu, port) in enumerate(zip(gpu_slots, ports)):
            t = threading.Thread(
                target=worker_loop,
                kwargs={
                    "worker_id": worker_offset + i,
                    "gpu": gpu,
                    "port": port,
                    "run_root": run_root,
                    "jobs": q,
                    "ckpts": ckpts,
                    "stop_on_error": stop_on_error,
                    "target_n": target_n,
                    "job_script": job_script,
                },
                daemon=False,
            )
            t.start()
            threads.append(t)

        while any(t.is_alive() for t in threads):
            status = summarize_status(run_root, ckpts=ckpts, tasks=tasks, target_n=target_n)
            print(
                f"[monitor] {datetime.now().isoformat(timespec='seconds')} "
                f"{status['done_jobs']}/{status['total_jobs']} jobs complete",
                flush=True,
            )
            time.sleep(60)

        for t in threads:
            t.join()

        summarize_status(run_root, ckpts=ckpts, tasks=tasks, target_n=target_n)
        remaining = sum(count_records(run_root, backbone, task) < target_n for backbone, task in jobs_to_run)
        print(
            f"[round] {datetime.now().isoformat(timespec='seconds')} "
            f"round={rounds} complete remaining_jobs={remaining}",
            flush=True,
        )


def worker_loop(
    *,
    worker_id: int,
    gpu: int,
    port: int,
    run_root: Path,
    jobs: "queue.Queue[tuple[str, str]]",
    ckpts: dict[str, str],
    stop_on_error: bool,
    target_n: int,
    job_script: Path,
) -> None:
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

        cmd = [str(job_script), str(run_root), backbone, task, str(gpu), str(port), ckpts[backbone]]
        started = datetime.now().isoformat(timespec="seconds")
        with worker_log.open("a", encoding="utf-8") as log:
            log.write(f"\n[{started}] worker={worker_id} gpu={gpu} port={port} start {backbone} {task}\n")
            log.flush()
            env = os.environ.copy()
            env["TARGET_N"] = str(target_n)
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            ended = datetime.now().isoformat(timespec="seconds")
            log.write(f"[{ended}] worker={worker_id} exit={proc.returncode} {backbone} {task}\n")

        jobs.task_done()
        if proc.returncode != 0 and stop_on_error:
            raise SystemExit(proc.returncode)


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    package_root = script_dir.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default="./eval-result")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--tasks-file", default=str(package_root / "configs" / "hard8_tasks.txt"))
    parser.add_argument("--ckpts-json", default=str(package_root / "configs" / "backbone_ckpts.json"))
    parser.add_argument("--gpu-slots", default="0")
    parser.add_argument(
        "--gpu-slots-override",
        action="append",
        default=[],
        help="Per-backbone slot override, e.g. wan21_i2v_14b=0. Can be repeated.",
    )
    parser.add_argument("--base-port", type=int, default=9100)
    parser.add_argument("--target-n", type=int, default=32)
    parser.add_argument("--retry-incomplete-rounds", type=int, default=20)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--phase-by-backbone", action="store_true")
    parser.add_argument("--skip-video-collection", action="store_true")
    parser.add_argument("--job-script", default=str(script_dir / "run_one_robotwin_job.sh"))
    parser.add_argument("--summary-script", default=str(script_dir / "summarize_hard8_passk.py"))
    args = parser.parse_args()

    tasks = load_tasks(Path(args.tasks_file))
    ckpts = load_ckpts(Path(args.ckpts_json))
    gpu_slots = parse_gpu_slots(args.gpu_slots)
    gpu_slot_overrides = parse_gpu_slot_overrides(args.gpu_slots_override, ckpts)

    run_name = args.run_name or f"robotwin_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root = Path(args.result_root).expanduser().resolve() / run_name
    for name in ("logs", "episodes", "summaries", "videos", "config"):
        (run_root / name).mkdir(parents=True, exist_ok=True)

    ports = [args.base_port + i for i in range(len(gpu_slots))]
    write_config(run_root, tasks=tasks, ckpts=ckpts, gpu_slots=gpu_slots, ports=ports, target_n=args.target_n)
    if gpu_slot_overrides:
        (run_root / "config" / "gpu_slot_overrides.json").write_text(
            json.dumps(gpu_slot_overrides, indent=2),
            encoding="utf-8",
        )

    summarize_status(run_root, ckpts=ckpts, tasks=tasks, target_n=args.target_n)

    if args.phase_by_backbone:
        for phase_idx, backbone in enumerate(ckpts):
            phase_slots = gpu_slot_overrides.get(backbone, gpu_slots)
            phase_jobs = [(backbone, task) for task in tasks]
            print(f"[phase] backbone={backbone} slots={','.join(str(x) for x in phase_slots)}", flush=True)
            run_job_workers(
                run_root=run_root,
                jobs_to_run=phase_jobs,
                ckpts=ckpts,
                tasks=tasks,
                gpu_slots=phase_slots,
                base_port=args.base_port + phase_idx * 100,
                stop_on_error=args.stop_on_error,
                target_n=args.target_n,
                worker_offset=phase_idx * 100,
                retry_incomplete_rounds=args.retry_incomplete_rounds,
                job_script=Path(args.job_script),
            )
    else:
        jobs = [(backbone, task) for backbone in ckpts for task in tasks]
        run_job_workers(
            run_root=run_root,
            jobs_to_run=jobs,
            ckpts=ckpts,
            tasks=tasks,
            gpu_slots=gpu_slots,
            base_port=args.base_port,
            stop_on_error=args.stop_on_error,
            target_n=args.target_n,
            worker_offset=0,
            retry_incomplete_rounds=args.retry_incomplete_rounds,
            job_script=Path(args.job_script),
        )

    print("[summary] summarizing pass@k", flush=True)
    run_summary(run_root, Path(args.summary_script))
    if not args.skip_video_collection:
        robotwin_path = os.environ.get("ROBOTWIN_PATH", "")
        if robotwin_path:
            print("[summary] collecting success videos", flush=True)
            collect_success_videos(run_root, robotwin_path=Path(robotwin_path), ckpts=ckpts, tasks=tasks)
        else:
            print("[summary] skip video collection: ROBOTWIN_PATH is not set", flush=True)
    print(f"[done] run_root={run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
