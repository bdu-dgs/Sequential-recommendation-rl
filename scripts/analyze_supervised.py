"""Tables for the real-data supervised evidence and the simulator reference policies.

Outputs reports/tables/supervised.md and reports/tables/env_reference.md.
Usage: python scripts/analyze_supervised.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "runs"


def ms(x):
    x = np.asarray(x, float)
    return f"{x.mean():.3f} ± {x.std():.3f}" if len(x) > 1 else f"{x.mean():.3f}"


def main():
    seeds = sorted(int(p.parent.name[4:]) for p in (RUNS / "response_model_wd").glob("seed*/metrics.json"))
    M = [json.loads((RUNS / "response_model_wd" / f"seed{s}" / "metrics.json").read_text()) for s in seeds]
    lines = [f"Real KuaiRec data. GRU two-tower response model (`response_model_wd`), mean ± std over seeds {seeds}. "
             "Baselines are deterministic. Targets: watch ratio capped at 2 (RMSE); engaged = watch ratio ≥ 1 "
             "(AUC, GAUC, ECE). Temporal test window: 2020-09-01..05.", "",
             "| test set | model | RMSE ↓ | AUC ↑ | GAUC ↑ | logloss ↓ | ECE ↓ |", "|---|---|---|---|---|---|---|"]
    sets = [r["set"] for r in M[0]["temporal_test"]]
    for st in sets:
        rows = [next(r for r in m["temporal_test"] if r["set"] == st) for m in M]
        group, model = st.split("/")[1], st.split("/")[2]
        lines.append(f"| {group} users | {model} | {ms([r['rmse_wr'] for r in rows])} | {ms([r['auc_eng'] for r in rows])} | "
                     f"{ms([r['gauc_eng'] for r in rows])} | {ms([r['logloss_eng'] for r in rows])} | "
                     f"{ms([r['ece_eng'] for r in rows])} |")
    lines += ["", "Full ranking of all 3,327 items for the 283 env-test users of the fully-observed matrix (no exposure "
              "bias; history = last 20 real views before 2020-08-10).", "",
              "| model | NDCG@10 ↑ | NDCG@50 ↑ | Spearman ↑ | mean true watch ratio of top-10 ↑ |", "|---|---|---|---|---|"]
    for name in ("gru_two_tower", "item_mean", "random"):
        R = [m["fo_matrix"]["test"][name] for m in M]
        lines.append(f"| {name} | {ms([r['ndcg@10'] for r in R])} | {ms([r['ndcg@50'] for r in R])} | "
                     f"{ms([r['spearman'] for r in R])} | {ms([r['mean_true_wr_top10'] for r in R])} |")
    orc = np.mean([m["fo_matrix"]["test"]["oracle_mean_true_wr_top10"] for m in M])
    g = [m["fo_matrix"]["test"]["gru_two_tower"] for m in M]
    lines += [f"| oracle (true values) | 1 | – | – | {orc:.3f} |", "",
              f"GRU on the FO matrix: engaged AUC {ms([r['auc_eng'] for r in g])}, ECE {ms([r['ece_eng'] for r in g])}, "
              f"RMSE {ms([r['rmse_wr'] for r in g])}."]
    comp = [json.loads((RUNS / "response_model_wd" / f"seed{s}" / "compute.json").read_text())["gpu_hours"] for s in seeds]
    lines.append(f"Compute: {np.mean(comp):.2f} GPU-hours per seed (RTX 4060 Laptop).")
    (ROOT / "reports" / "tables" / "supervised.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    ch = json.loads((RUNS / "characterize_env" / "seed0" / "metrics.json").read_text())
    L = ["Reference policies in KuaiRecFO-v1 on the 282 val users (simulated return, max 60). Oracle policies read the "
         "true response table and are reference points only.", "",
         "| policy | simulated return | length | leave rate | coverage |", "|---|---|---|---|---|"]
    for k, v in ch["policies"].items():
        L.append(f"| {k} | {v['sim_return']:.2f} | {v['length']:.1f} | {v['leave_rate']:.2f} | {v['coverage']:.3f} |")
    d = ch["dataset"]
    L += ["", f"Offline dataset (seed 0): {d['transitions']:,} transitions, {d['episodes']:,} episodes; "
          f"{d['items_logged_at_least_once']:,} items logged at least once, {d['items_logged_10plus']:,} logged ≥ 10 times; "
          f"top-100 items = {d['top100_item_share']:.0%} of the log; logging-policy entropy {d['logging_entropy_nats_mean']:.2f} nats "
          f"(effective support ≈ {d['logging_effective_support_exp_entropy']:.0f} items per state)."]
    (ROOT / "reports" / "tables" / "env_reference.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
