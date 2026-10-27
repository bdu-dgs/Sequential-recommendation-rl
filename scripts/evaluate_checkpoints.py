"""Generalisation stress tests (step 8): evaluate trained policies on TEST users under changed environments.
Evaluation only; no parameters are updated. All returns are SIMULATED returns.

Settings: base, preference drift (delta 0.2 / 0.5), cold start (0 / 2 warm-start items), observation noise 0.3,
stronger fatigue (beta 0.6).
Usage: python scripts/evaluate_checkpoints.py --seed 0
"""

from __future__ import annotations

import argparse

import numpy as np

from seqrec_rl.evaluation.rollout import evaluate
from seqrec_rl.training.checkpoints import load_agent
from seqrec_rl.training.runs import Run, limit_threads, load_config, set_seed
from seqrec_rl.training.setup import Setup

SETTINGS = {
    "base": {},
    "drift0.2": {"drift_delta": 0.2, "drift_seed": 99},
    "drift0.5": {"drift_delta": 0.5, "drift_seed": 99},
    "cold0": {"cold_start_hist": 0},
    "cold2": {"cold_start_hist": 2},
    "noise0.3": {"noise_std": 0.3},
    "fatigue0.6": {"fatigue_beta": 0.6},
}
POLICIES = [("offline", m) for m in ("random", "popularity", "behavior", "ranker", "bc", "dqn", "cql", "calql", "bcq")] + \
           [("online", m) for m in ("scratch_dqn", "scratch_rlpd", "bandit", "ft_dqn", "ft_cql", "ft_cql_decay",
                                    "ft_calql", "ft_support_oracle", "ft_support_bc", "ft_support_uniform", "ft_support_uniform_tracked")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment/base.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_users", type=int, default=None)
    ap.add_argument("--tag", default="", help="tag of the runs to evaluate")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    limit_threads(4)
    set_seed(args.seed)
    run = Run("eval_generalization" + (f"_{args.tag}" if args.tag else ""), args.seed, cfg)
    S = Setup(cfg, args.seed)
    rows = S.rows["test"] if args.n_users is None else S.rows["test"][:args.n_users]
    lp = S.logging_policy()
    out = {}
    for kind, m in POLICIES:
        agent = load_agent(S, kind, m, args.seed, args.tag)
        if agent is None:
            print("missing", kind, m)
            continue
        out[f"{kind}/{m}"] = {}
        for sname, ov in SETTINGS.items():
            env = S.env("test", min(cfg["eval"]["n_envs"], len(rows)), seed_offset=11, **ov)
            ev = evaluate(env, agent, rows, np.random.default_rng(0), S.item_tail, lp.probs_np_obs)
            ev.pop("returns")
            out[f"{kind}/{m}"][sname] = ev
        print(f"{kind}/{m:22s} " + "  ".join(f"{k} {v['sim_return']:.2f}" for k, v in out[f'{kind}/{m}'].items()))
    run.save_metrics({"settings": SETTINGS, "results": out,
                      "note": "Simulated returns on test users; drift/cold/noise/fatigue are synthetic stress tests."})


if __name__ == "__main__":
    main()
