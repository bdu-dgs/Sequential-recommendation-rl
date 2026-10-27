"""Characterise the simulator and the logging policy before any RL (step 4).

Reports SIMULATED returns of reference policies on validation users and coverage statistics of the generated
offline dataset. Oracle policies read the true response table (cheating); they are reference points only.

Usage: python scripts/characterize_env.py --config configs/experiment/base.yaml --seed 0 [key=value ...]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from seqrec_rl.agents.base import Agent
from seqrec_rl.agents.q_agents import PopularityAgent, RandomAgent
from seqrec_rl.environments.base import obs_to_torch
from seqrec_rl.evaluation.rollout import evaluate
from seqrec_rl.training.runs import Run, limit_threads, load_config, set_seed
from seqrec_rl.training.setup import Setup


class OracleGreedy(Agent):
    """Myopic oracle: picks the item with the highest true effective response at this step (reads the env)."""
    name = "oracle_greedy"

    def __init__(self, env, fatigue_aware=True):
        super().__init__(env.spec)
        self.env, self.fatigue_aware = env, fatigue_aware

    def scores(self, obs_t):
        s = self.env.response_all() if self.fatigue_aware else self.env.W[self.env.u]
        return torch.as_tensor(s)


class LoggingAgent(Agent):
    name = "logging_policy"

    def __init__(self, lp, greedy):
        super().__init__(lp.spec, lp.device)
        self.lp, self.greedy = lp, greedy

    def act(self, obs, rng, explore=False):
        if self.greedy:
            return self.lp.probs_t(obs_to_torch(obs, self.device)).argmax(1).cpu().numpy()
        return self.lp.sample(obs, rng)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment/base.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="characterize_env")
    ap.add_argument("--n_users", type=int, default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    limit_threads(4)
    set_seed(args.seed)
    run = Run(args.name, args.seed, cfg)
    S = Setup(cfg, args.seed)
    rows = S.rows["val"] if args.n_users is None else S.rows["val"][:args.n_users]
    env = S.env("val", min(128, len(rows)), seed_offset=7)
    lp = S.logging_policy()
    rng = np.random.default_rng(0)
    out = {"policies": {}}
    pop_counts = S.item_pop[:S.data.n_items]
    policies = {
        "random": RandomAgent(S.spec),
        "popularity_real": PopularityAgent(S.spec, pop_counts),
        "logging_stochastic": LoggingAgent(lp, greedy=False),
        "logging_greedy (response-model ranker)": LoggingAgent(lp, greedy=True),
        "oracle_greedy_no_fatigue": OracleGreedy(env, fatigue_aware=False),
        "oracle_greedy": OracleGreedy(env, fatigue_aware=True),
    }
    for name, ag in policies.items():
        ev = evaluate(env, ag, rows, rng, S.item_tail, lp.probs_np_obs)
        ev.pop("returns")
        out["policies"][name] = ev
        print(f"{name:42s} return {ev['sim_return']:.3f}  len {ev['length']:.1f}  leave {ev['leave_rate']:.2f}  "
              f"cov {ev['coverage']:.3f}  tail {ev['long_tail_share']:.3f}")

    # --- logging policy and dataset statistics -----------------------------------------------------
    buf = S.offline_buffer()
    n = buf.size
    acts = buf.data["action"][:n]
    counts = np.bincount(acts, minlength=S.data.n_items)
    props = buf.data["prop"][:n]
    ent = []
    idx = rng.choice(n, size=min(4096, n), replace=False)
    with torch.no_grad():
        for s in range(0, len(idx), 512):
            b = buf.batch(idx[s:s + 512], S.dev)
            from seqrec_rl.replay.buffer import obs_from_batch
            p = lp.probs_t(obs_from_batch(b)).clamp(min=1e-12)
            ent.append((-(p * p.log()).sum(1)).cpu().numpy())
    ent = np.concatenate(ent)
    ep = buf.data["episode"][:n]
    finished = np.isin(ep, ep[buf.data["done"][:n]])  # only episodes that ended inside the dataset
    ep_ret = buf.data["true_reward"][:n][finished].sum() / max(1, buf.data["done"][:n].sum())
    out["dataset"] = {
        "name": S.dataset_name(), "transitions": int(n), "episodes": int(buf.data["done"][:n].sum()),
        "mean_episode_true_return": float(ep_ret),
        "items_logged_at_least_once": int((counts > 0).sum()),
        "items_logged_10plus": int((counts >= 10).sum()),
        "top100_item_share": float(np.sort(counts)[::-1][:100].sum() / n),
        "logging_entropy_nats_mean": float(ent.mean()),
        "logging_effective_support_exp_entropy": float(np.exp(ent).mean()),
        "logged_prop_quantiles": {str(q): float(np.quantile(props, q)) for q in (0.01, 0.1, 0.5, 0.9)},
    }
    print(json.dumps(out["dataset"], indent=1))
    run.save_metrics(out)


if __name__ == "__main__":
    main()
