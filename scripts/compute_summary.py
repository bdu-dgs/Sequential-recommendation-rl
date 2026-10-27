"""Sum recorded compute (compute.json in every run directory) by experiment family.

Each run records its own wall-clock time on the device it used. Runs were executed 3 at a time on one GPU, so
"GPU-process-hours" over-counts physical GPU time by up to ~3x; both numbers are reported.
Usage: python scripts/compute_summary.py
"""

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "runs"
TAGS = ("narrow", "diag", "learnedsim", "lam0", "kappa0.3", "kappa3.0")

if __name__ == "__main__":
    fam = defaultdict(lambda: [0, 0.0])
    for p in RUNS.rglob("compute.json"):
        rel = p.relative_to(RUNS).parts
        if "archive" in rel[0] or "smoke" in "/".join(rel):
            continue
        if rel[0] in ("offline", "online"):
            tag = next((t for t in TAGS if rel[1].endswith("_" + t)), "base")
            group = f"{rel[0]}/{tag}"
        else:
            group = rel[0]
        d = json.loads(p.read_text())
        fam[group][0] += 1
        fam[group][1] += d["total_wall_s"] / 3600
    lines = ["| experiment family | runs | GPU-process-hours |", "|---|---|---|"]
    tot_r, tot_h = 0, 0.0
    for k in sorted(fam):
        n, h = fam[k]
        tot_r, tot_h = tot_r + n, tot_h + h
        lines.append(f"| {k} | {n} | {h:.2f} |")
    lines.append(f"| **total** | **{tot_r}** | **{tot_h:.2f}** |")
    txt = ("Recorded compute per run (RTX 4060 Laptop GPU, 8 GB). Runs were executed 3 in parallel, so physical "
           "GPU time is roughly one third to one half of the GPU-process-hours.\n\n" + "\n".join(lines) + "\n")
    (ROOT / "reports" / "tables" / "compute.md").write_text(txt, encoding="utf-8")
    print(txt)
