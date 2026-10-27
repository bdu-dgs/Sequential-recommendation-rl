"""Run lines of experiments/jobs/<stage>.txt locally with a small process pool (same commands as on RIS).

Usage: python scripts/run_local.py --stages s5b_offline s6_online --seed 0 --parallel 3
Logs: logs/local/<stage>/line<N>.log. Stages run one after another; lines inside a stage run in parallel.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_line(stage: str, i: int, line: str, threads: int) -> tuple[int, int, float]:
    log = ROOT / "logs" / "local" / stage / f"line{i}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "SEQRL_THREADS": str(threads), "OMP_NUM_THREADS": str(threads)}
    cmd = line.replace("python ", f'"{sys.executable}" ', 1)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.call(cmd, shell=True, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
    return i, rc, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=None, help="only lines with this --seed")
    ap.add_argument("--grep", nargs="*", default=None, help="keep only lines containing any of these substrings")
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    for stage in a.stages:
        lines = (ROOT / "experiments" / "jobs" / f"{stage}.txt").read_text().splitlines()
        todo = [(i + 1, l) for i, l in enumerate(lines)
                if l.strip() and (a.seed is None or f"--seed {a.seed} " in l + " ")
                and (a.grep is None or any(g in l for g in a.grep))]
        print(f"[{time.strftime('%H:%M:%S')}] {stage}: {len(todo)} commands, {a.parallel} in parallel", flush=True)
        with ThreadPoolExecutor(a.parallel) as ex:
            futs = [ex.submit(run_line, stage, i, l, a.threads) for i, l in todo]
            for f in as_completed(futs):
                i, rc, dt = f.result()
                status = "ok" if rc == 0 else f"FAILED rc={rc}"
                print(f"[{time.strftime('%H:%M:%S')}] {stage} line {i} {status} ({dt / 60:.1f} min): "
                      f"{lines[i - 1].split('scripts/')[1][:90]}", flush=True)


if __name__ == "__main__":
    main()
