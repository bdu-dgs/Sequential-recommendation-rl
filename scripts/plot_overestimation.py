"""Mechanism analysis: Q over-estimation by logging support during offline-to-online fine-tuning.

At every validation evaluation, for each greedy recommendation we compare the network's Q(s, a) with the true value
of that (s, a) under the current greedy policy — the discounted return actually collected (exact, because the
simulator is deterministic with noise_std = 0). Recommendations are bucketed by the logging probability of the
chosen item. SIMULATED environment (KuaiRecFO-v1).

Usage: python scripts/plot_overestimation.py --tag diag --seeds 0 1 2
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
METHODS = {"ft_dqn": "naive fine-tune (no penalty)", "ft_cql": "constant CQL",
           "ft_support_oracle": "support-weighted CQL (oracle)", "ft_support_uniform_tracked": "uniform, strength-matched"}
BUCKETS = ["p<1e-4", "1e-4<=p<1e-2", "p>=1e-2"]


def series(method, tag, seeds):
    out = []
    for s in seeds:
        p = ROOT / "experiments" / "runs" / "online" / f"{method}_{tag}" / f"seed{s}" / "metrics.json"
        if p.exists():
            out.append(json.loads(p.read_text())["history"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="diag")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    rows = ["| method | seeds | return (start → end) | over-estimation overall (start → end) | "
            "share of recs with 1e-4 ≤ p < 1e-2 (end) | over-est. in that bucket (end) | "
            "share with p < 1e-4 (end) | over-est. p ≥ 1e-2 (end) |", "|---|---|---|---|---|---|---|---|"]
    for k, (m, label) in enumerate(METHODS.items()):
        H = series(m, a.tag, a.seeds)
        if not H:
            continue
        L = min(len(h) for h in H)
        used = np.array([e["used"] for e in H[0][:L]])

        def get(f):
            return np.array([[f(e) for e in h[:L]] for h in H], dtype=float)

        ret = get(lambda e: e["sim_return"])
        over = get(lambda e: e["q_diag"]["all"]["overestimation"])
        c = f"C{k}"
        axes[0, 0].plot(used, ret.mean(0), color=c, label=label)
        axes[0, 1].plot(used, over.mean(0), color=c, label=label)
        for b_i, b in enumerate(BUCKETS):
            share = get(lambda e: e["q_diag"][b]["share"])
            ob = get(lambda e: e["q_diag"][b]["overestimation"])
            ls = ["-", "--", ":"][b_i]
            axes[1, b_i].plot(used, share.mean(0), ls, color=c, label=label)
            ax2 = axes[0, 2]
            if b == "1e-4<=p<1e-2":
                ax2.plot(used, np.nanmean(ob, 0), color=c, label=label)
        end = lambda b, key: np.nanmean([h[L - 1]["q_diag"][b][key] for h in H])  # noqa: E731
        rows.append(f"| {label} | {len(H)} | {ret[:, 0].mean():.1f} → {ret[:, -1].mean():.1f} | "
                    f"{over[:, 0].mean():+.1f} → {over[:, -1].mean():+.1f} | {end('1e-4<=p<1e-2', 'share'):.2f} | "
                    f"{end('1e-4<=p<1e-2', 'overestimation'):+.1f} | {end('p<1e-4', 'share'):.2f} | "
                    f"{end('p>=1e-2', 'overestimation'):+.1f} |")
    titles = [["simulated return (val users)", "Q over-estimation, all recommendations",
               "Q over-estimation, mid-support items (1e-4 ≤ p < 1e-2)"],
              ["share of recs: unsupported (p < 1e-4)", "share of recs: mid support (1e-4 ≤ p < 1e-2)",
               "share of recs: high support (p ≥ 1e-2)"]]
    for i in range(2):
        for j in range(3):
            axes[i, j].set_title(titles[i][j], fontsize=10)
            axes[i, j].set_xlabel("online transitions")
            axes[i, j].grid(alpha=0.3)
    axes[0, 1].axhline(0, color="k", lw=0.7)
    axes[0, 2].axhline(0, color="k", lw=0.7)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Where fine-tuning goes wrong: Q over-estimation by logging support "
                 f"(mean of {len(a.seeds)} seeds; p = logging probability of the chosen item; simulated)", fontsize=11)
    fig.tight_layout()
    sfx = "" if a.tag == "diag" else f"_{a.tag}"
    out = ROOT / "reports" / "figures" / f"overestimation_by_support{sfx}.png"
    fig.savefig(out, dpi=150)
    tab = ROOT / "reports" / "tables" / f"overestimation{sfx}.md"
    tab.write_text("Q over-estimation = predicted Q(s,a) − true discounted return of (s,a) under the greedy policy "
                   "(exact in the deterministic simulator), on val users. γ = 0.95. p = logging probability.\n\n"
                   + "\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))
    print("saved", out)


if __name__ == "__main__":
    main()
