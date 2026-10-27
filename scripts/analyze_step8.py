"""Step-8 analysis: simulator exploitation, generalisation stress tests, OPE, learned-simulator error by support.

Outputs reports/tables/{exploitation,generalization,ope,learned_sim_error}.md and reports/figures/ope_error.png.
All policy numbers are SIMULATED returns.
Usage: python scripts/analyze_step8.py --seeds 0 1 2 3 4
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
T = ROOT / "reports" / "tables"
F = ROOT / "reports" / "figures"


def mci(x):
    x = np.asarray([v for v in x if v is not None and not np.isnan(v)], float)
    if len(x) == 0:
        return "–"
    if len(x) == 1:
        return f"{x[0]:.2f}"
    bs = np.random.default_rng(0).choice(x, (2000, len(x))).mean(1)
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return f"{x.mean():.2f} [{lo:.2f}, {hi:.2f}]"


def jl(p):
    return json.loads(p.read_text()) if p.exists() else None


def exploitation(seeds):
    rows = ["| method | seeds | test return, trained in FO sim | trained in learned sim: return in FO sim | "
            "same policy scored inside learned sim | gap (learned − FO) |", "|---|---|---|---|---|---|"]
    for m in ("scratch_dqn", "ft_dqn", "ft_cql", "ft_support_oracle"):
        base, fo, learned = [], [], []
        for s in seeds:
            b = jl(RUNS / "online" / m / f"seed{s}" / "metrics.json")
            l = jl(RUNS / "online" / f"{m}_learnedsim" / f"seed{s}" / "metrics.json")
            if b:
                base.append(b["test"]["sim_return"])
            if l:
                fo.append(l["test"]["sim_return"])
                learned.append(l["test_in_training_sim"]["sim_return"])
        gap = [a - b for a, b in zip(learned, fo)]
        rows.append(f"| {m} | {len(fo)} | {mci(base)} | {mci(fo)} | {mci(learned)} | {mci(gap)} |")
    txt = ("Policies interacting online with the LEARNED simulator (response-model predictions instead of the "
           "fully-observed matrix), evaluated on test users in the FO simulator; the last column is the over-statement "
           "an evaluator would make if trusting the learned simulator. Simulated returns.\n\n" + "\n".join(rows) + "\n")
    (T / "exploitation.md").write_text(txt, encoding="utf-8")
    print(txt)


def generalization(seeds):
    R = [jl(RUNS / "eval_generalization" / f"seed{s}" / "metrics.json") for s in seeds]
    R = [r for r in R if r]
    if not R:
        return
    settings = list(R[0]["settings"])
    rows = ["| policy | " + " | ".join(settings) + " |", "|---|" + "---|" * len(settings)]
    for pol in R[0]["results"]:
        cells = [mci([r["results"][pol][st]["sim_return"] for r in R if pol in r["results"]]) for st in settings]
        rows.append(f"| {pol} | " + " | ".join(cells) + " |")
    txt = (f"Test users, {len(R)} seeds, simulated return under stress settings (synthetic drift / cold start / "
           "observation noise / stronger fatigue). Mean [95% bootstrap CI].\n\n" + "\n".join(rows) + "\n")
    (T / "generalization.md").write_text(txt, encoding="utf-8")
    print(txt)


def ope(seeds):
    R = [jl(RUNS / "ope" / f"seed{s}" / "metrics.json") for s in seeds]
    R = [r for r in R if r]
    if not R:
        return
    rows = ["| target policy | true V1 | IPS rel. err | SNIPS rel. err | DM rel. err | DR rel. err | ESS | "
            "unsupported mass |", "|---|---|---|---|---|---|---|---|"]
    pts = []
    for pol in R[0]["results"]:
        X = [r["results"][pol] for r in R if pol in r["results"]]
        rel = {k: [x["rel_error"][k] for x in X] for k in ("IPS", "SNIPS", "DM", "DR")}
        rows.append(f"| {pol} | {mci([x['true_V1'] for x in X])} | " +
                    " | ".join(mci(rel[k]) for k in ("IPS", "SNIPS", "DM", "DR")) +
                    f" | {mci([x['diagnostics']['ess'] for x in X])} | "
                    f"{mci([x['diagnostics']['unsupported_mass'] for x in X])} |")
        for x in X:
            pts.append((x["diagnostics"]["unsupported_mass"], x["rel_error"]["SNIPS"], x["rel_error"]["DR"],
                        x["rel_error"]["DM"]))
    txt = ("One-step OPE on SIMULATED logs with KNOWN propensities (50k logged states per seed), ground truth computed "
           "exactly in the simulator. Relative error = |estimate − truth| / truth. Unsupported mass = share of logged "
           f"states where the target's action has logging probability < 1e-4. {len(R)} seeds.\n\n"
           + "\n".join(rows) + "\n")
    (T / "ope.md").write_text(txt, encoding="utf-8")
    print(txt)
    p = np.array(pts)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for j, (lab, mk) in enumerate((("SNIPS", "o"), ("DR", "s"), ("DM", "^")), start=1):
        ax.scatter(p[:, 0], p[:, j], label=lab, marker=mk, alpha=0.7)
    ax.set_xlabel("unsupported mass of the target policy")
    ax.set_ylabel("relative error of the estimate")
    ax.set_title("OPE error grows with unsupported mass (simulated logs, known propensities)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(F / "ope_error.png", dpi=150)


def learned_sim_error():
    pred_p = RUNS / "response_model_wd" / "seed0" / "fo_pred_wr.npy"
    data = ROOT / "data" / "processed"
    off = sorted((ROOT / "data" / "offline").glob("T0.1_eps0.05_headNone_*_s0.npz"))
    if not pred_p.exists() or not off:
        return
    pred = np.load(pred_p)
    W = np.nan_to_num(np.minimum(np.load(data / "fo_watch_ratio.npy"), 2.0), nan=0.0)
    counts = np.bincount(np.load(off[0])["action"], minlength=W.shape[1])
    err = pred - W
    rows = ["| item log count (seed-0 offline data) | items | mean error (pred − true) | RMSE | "
            "share of items with pred > true + 0.5 |", "|---|---|---|---|---|"]
    for lab, lo, hi in (("0", 0, 1), ("1–9", 1, 10), ("10–99", 10, 100), ("≥100", 100, 10 ** 9)):
        m = (counts >= lo) & (counts < hi)
        e = err[:, m]
        rows.append(f"| {lab} | {int(m.sum())} | {e.mean():+.3f} | {np.sqrt((e ** 2).mean()):.3f} | "
                    f"{(e.mean(0) > 0.5).mean():.2f} |")
    txt = ("Error of the learned response model (real-data GRU, seed 0) against the true FO watch ratio (capped at 2), "
           "for all 1,411 FO users, grouped by how often each item appears in the simulated offline log.\n\n"
           + "\n".join(rows) + "\n")
    (T / "learned_sim_error.md").write_text(txt, encoding="utf-8")
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    T.mkdir(parents=True, exist_ok=True)
    F.mkdir(parents=True, exist_ok=True)
    learned_sim_error()
    exploitation(a.seeds)
    generalization(a.seeds)
    ope(a.seeds)


if __name__ == "__main__":
    main()
