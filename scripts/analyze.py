"""Aggregate run metrics into tables and figures (step 9). Works with any number of finished seeds.

Outputs reports/tables/{offline,online}_<tag>.md and reports/figures/finetune_curves_<tag>.png.
All policy numbers are SIMULATED returns in KuaiRecFO-v1.
Usage: python scripts/analyze.py [--tag ""] [--seeds 0 1 2 3 4]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "runs"
OFFLINE = ["random", "popularity", "behavior", "ranker", "bc", "dqn", "cql", "calql", "bcq"]
ONLINE = ["scratch_dqn", "scratch_rlpd", "bandit", "ft_dqn", "ft_cql", "ft_cql_decay", "ft_calql",
          "ft_support_oracle", "ft_support_bc", "ft_support_uniform", "ft_support_uniform_tracked",
          "ft_support_shuffled", "ft_support_inverted"]


def load(kind, method, tag, seeds):
    out = {}
    for s in seeds:
        p = RUNS / kind / (method + (f"_{tag}" if tag else "")) / f"seed{s}" / "metrics.json"
        if p.exists():
            out[s] = json.loads(p.read_text())
    return out


def ci(x, n_boot=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    if len(x) == 1:
        return x[0], np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, size=(n_boot, len(x))).mean(1)
    return x.mean(), *np.quantile(bs, [0.025, 0.975])


def fmt(x):
    m, lo, hi = ci(x)
    if np.isnan(m):
        return "–"
    return f"{m:.2f}" if np.isnan(lo) else f"{m:.2f} [{lo:.2f}, {hi:.2f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    tdir, fdir = ROOT / "reports" / "tables", ROOT / "reports" / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    sfx = f"_{a.tag}" if a.tag else ""

    # ---------------- offline ------------------------------------------------------------------------
    rows = ["| method | seeds | test return | length | leave rate | coverage | long-tail share | "
            "chosen with logging prob < 1e-4 |", "|---|---|---|---|---|---|---|---|"]
    behavior_val = []
    for m in OFFLINE:
        r = load("offline", m, a.tag, a.seeds)
        if not r:
            continue
        T = [v["test"] for v in r.values()]
        if m == "behavior":
            behavior_val = [v["val"]["sim_return"] for v in r.values()]
        rows.append(f"| {m} | {len(r)} | {fmt([t['sim_return'] for t in T])} | {fmt([t['length'] for t in T])} | "
                    f"{fmt([t['leave_rate'] for t in T])} | {fmt([t['coverage'] for t in T])} | "
                    f"{fmt([t['long_tail_share'] for t in T])} | "
                    f"{fmt([t.get('frac_chosen_below_1e-4', np.nan) for t in T])} |")
    (tdir / f"offline{sfx}.md").write_text(
        "Simulated returns on env-test users (KuaiRecFO-v1, max 60). Mean [95% bootstrap CI] over seeds.\n\n"
        + "\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))

    # ---------------- online -------------------------------------------------------------------------
    beh = float(np.mean(behavior_val)) if behavior_val else np.nan
    rows = ["| method | seeds | val start | worst drop | val final | mean val (AUC/len) | "
            "transitions to exceed logging policy | test return |", "|---|---|---|---|---|---|---|---|"]
    # left panel: baselines and fine-tuning schedules; right panel: support-weighting variants vs constant CQL
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    n_drawn = [0, 0]
    for m in ONLINE:
        r = load("online", m, a.tag, a.seeds)
        if not r:
            continue
        starts, drops, finals, aucs, reach, tests, curves = [], [], [], [], [], [], []
        for v in r.values():
            h = v["history"]
            used = np.array([e["used"] for e in h])
            ret = np.array([e["sim_return"] for e in h])
            starts.append(ret[0])
            drops.append(min(0.0, ret.min() - ret[0]))
            finals.append(ret[-1])
            aucs.append(ret.mean())
            hit = np.flatnonzero(ret > beh)
            reach.append(used[hit[0]] if len(hit) else np.nan)
            tests.append(v["test"]["sim_return"])
            curves.append((used, ret))
        reach_s = fmt([x for x in reach if not np.isnan(x)]) + f" ({sum(~np.isnan(reach))}/{len(reach)} reached)"
        rows.append(f"| {m} | {len(r)} | {fmt(starts)} | {fmt(drops)} | {fmt(finals)} | {fmt(aucs)} | {reach_s} | "
                    f"{fmt(tests)} |")
        L = min(len(c[0]) for c in curves)
        used = curves[0][0][:L]
        Y = np.stack([c[1][:L] for c in curves])
        panels = [1] if m.startswith("ft_support") else ([0, 1] if m == "ft_cql" else [0])
        for pi in panels:
            ax, c = axes[pi], f"C{n_drawn[pi]}"
            n_drawn[pi] += 1
            style = "--" if m.startswith("scratch") or m == "bandit" else "-"
            ax.plot(used, Y.mean(0), style, color=c, label=m, lw=1.6)
            if len(Y) > 1:
                ax.fill_between(used, Y.min(0), Y.max(0), color=c, alpha=0.10)
    for ax, title in zip(axes, ("baselines and fine-tuning schedules", "support-weighted CQL vs controls")):
        if not np.isnan(beh):
            ax.axhline(beh, color="k", lw=0.8, ls="-.", label="logging policy (stochastic)")
        ax.set_xlabel("online environment transitions")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("simulated return on val users (KuaiRecFO-v1)")
    fig.suptitle("Offline-to-online fine-tuning (mean over seeds; band = min-max)" + (f" - {a.tag}" if a.tag else ""))
    fig.tight_layout()
    fig.savefig(fdir / f"finetune_curves{sfx}.png", dpi=150)
    (tdir / f"online{sfx}.md").write_text(
        "Validation curves every 2.5k transitions (50k budget); test = env-test users after fine-tuning.\n"
        "Simulated returns, KuaiRecFO-v1. Mean [95% bootstrap CI] over seeds.\n\n" + "\n".join(rows) + "\n",
        encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
