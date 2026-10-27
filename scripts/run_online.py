"""Online learning / offline-to-online fine-tuning in the simulator (SIMULATED returns).

Methods
  scratch_dqn          Double DQN from random init, online data only
  scratch_rlpd         Double DQN from random init, 50/50 offline/online batches (RLPD-style symmetric sampling;
                       no LayerNorm/ensemble, so "RLPD-style", not RLPD)
  ft_dqn               init from offline CQL, then unregularised Double DQN updates (naive fine-tuning)
  ft_cql               init from offline CQL, keep the uniform CQL penalty
  ft_cql_decay         init from offline CQL, CQL alpha decays linearly to 0 over the budget (SPOT-style schedule)
  ft_calql             init from offline Cal-QL, keep the Cal-QL penalty
  ft_support_<src>     init from offline CQL, support-weighted CQL penalty; src in
                       oracle | bc | uniform (matched mean strength) | shuffled | inverted
  bandit               NeuralLinear-UCB on top of the offline reward-regression ranker (gamma = 0)

Inheritance (configs: online.keep_target, online.keep_optimizer, online.offline_ratio) is recorded in the run config.
Usage: python scripts/run_online.py --method ft_support_oracle --seed 0 [key=value ...]
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from seqrec_rl import RUNS_DIR
from seqrec_rl.agents.q_agents import BCAgent
from seqrec_rl.replay.buffer import ReplayBuffer, obs_from_batch
from seqrec_rl.training.loops import Inherit, train_online
from seqrec_rl.training.runs import Run, limit_threads, load_config, set_seed
from seqrec_rl.training.setup import Setup


INIT_TAG = ""


def init_path(method_dir: str, seed: int):
    method_dir = method_dir + (f"_{INIT_TAG}" if INIT_TAG else "")
    p = RUNS_DIR / "offline" / method_dir / f"seed{seed}" / "final_agent.pt"
    if not p.exists():
        raise FileNotFoundError(f"{p} — run scripts/run_offline.py for '{method_dir}' first")
    return p


@torch.no_grad()
def mean_weight(agent, support, buf, rng, n_batches=20, bs=256) -> float:
    """E_s[sum_a mu(a|s) c(s,a)] under the inherited Q — used to give the uniform control the same mean strength."""
    from seqrec_rl.environments.base import action_mask
    vals = []
    for _ in range(n_batches):
        b = buf.batch(buf.sample_idx(bs, rng), agent.device)
        obs = obs_from_batch(b)
        mask = action_mask(obs["ep_items"], agent.spec.n_items)
        mu = torch.softmax(agent.q(obs).masked_fill(~mask, -torch.inf), 1)
        vals.append((mu * support(obs)).sum(1).mean().item())
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--config", default="configs/experiment/base.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--init_tag", default="", help="tag of the offline runs to initialise from")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    global INIT_TAG
    INIT_TAG = args.init_tag
    cfg = load_config(args.config, args.overrides)
    limit_threads(4)
    set_seed(args.seed)
    m = args.method
    name = f"online/{m}" + (f"_{args.tag}" if args.tag else "")
    oc = cfg["online"]
    run = Run(name, args.seed, cfg)
    S = Setup(cfg, args.seed)
    offline = S.offline_buffer()
    rng = np.random.default_rng(args.seed)
    env = S.env("train", oc["n_envs"], seed_offset=3)
    qkw = dict(eps_start=oc["eps_start"], eps_end=oc["eps_end"], eps_decay_steps=oc["eps_decay_steps"])
    inherit = Inherit(keep_target=oc["keep_target"], keep_optimizer=oc["keep_optimizer"],
                      offline_ratio=oc["offline_ratio"])
    alpha_schedule, on_step, updates = None, None, oc["updates_per_collect"]
    from seqrec_rl.agents.q_agents import QAgent

    if m in ("scratch_dqn", "scratch_rlpd"):
        agent = QAgent(S.spec, S.q_config(**qkw), S.dev)
        inherit = Inherit(keep_target=True, keep_optimizer=True, offline_ratio=0.0 if m == "scratch_dqn" else 0.5)
        # from scratch, explore broadly at first
        agent.cfg.eps_start = max(agent.cfg.eps_start, 1.0)
    elif m == "bandit":
        ranker = S.make_agent("ranker")
        ranker.load(init_path("ranker", args.seed))
        agent = S.bandit(ranker)
        updates = 0
        inherit = Inherit(offline_ratio=0.0)
    elif m.startswith("ft_"):
        variant = m[3:]
        src_offline = "calql" if variant == "calql" else "cql"
        alpha = cfg["cql"]["alpha"]
        support = None
        if variant.startswith("support_"):
            source = variant.split("_", 1)[1]
            if source == "bc":
                bc = BCAgent(S.spec, emb_dim=cfg["agent"]["emb_dim"], hidden=cfg["agent"]["hidden"], device=S.dev)
                bc.load(init_path("bc", args.seed))
                support = S.support("bc", prob_fn=bc.probs)
            elif source in ("uniform", "uniform_tracked"):
                support = S.support(source)
                if source == "uniform_tracked":
                    support.reference = S.support("oracle")
            else:
                support = S.support(source)
        agent = QAgent(S.spec, S.q_config(cql_alpha=0.0 if variant == "dqn" else alpha,
                                          calql=variant == "calql", **qkw), S.dev, support)
        agent.load(init_path(src_offline, args.seed), keys=("q", "q_target", "opt"))
        if variant.startswith("support_") and support.source in ("uniform", "uniform_tracked"):
            oracle = S.support("oracle")
            support.const = mean_weight(agent, oracle, offline, np.random.default_rng(0))
            run.log(phase="setup", matched_uniform_const=support.const)
        if variant.startswith("support_"):
            run.log(phase="setup", mean_weight_start=mean_weight(agent, support, offline, np.random.default_rng(0)))
            on_step = support.observe_online
        if variant == "cql_decay":
            alpha_schedule = lambda used: max(0.0, 1.0 - used / oc["budget"])  # noqa: E731
    else:
        raise ValueError(m)

    online = ReplayBuffer(S.spec, oc["budget"] + 4 * oc["n_envs"])
    history = train_online(agent, env, online, offline, oc["budget"], rng, inherit,
                           batch_size=cfg["agent"]["batch_size"], updates_per_collect=updates, run=run,
                           eval_fn=S.eval_fn("val"), eval_every=oc["eval_every"], alpha_schedule=alpha_schedule,
                           on_step=on_step, gamma=S.sim_cfg.gamma)
    test = S.eval_fn("test")(agent, keep_trajectories=20)
    metrics = {"method": m, "dataset": S.dataset_name(), "inherit": inherit.__dict__, "history": history,
               "test": test, "eval_sim": {**S.sim_cfg.to_dict(), **cfg.get("eval", {})},
               "note": "All returns are simulated returns in KuaiRecFO-v1."}
    if S.sim_cfg.response == "learned":  # exploitation study: also score the policy inside the training sim
        metrics["test_in_training_sim"] = S.eval_fn("test", response="learned",
                                                    learned_response_path=S.sim_cfg.learned_response_path)(agent)
        metrics["test_in_training_sim"].pop("returns", None)
    run.save_metrics(metrics)
    h0, hT = history[0]["sim_return"], history[-1]["sim_return"]
    dip = min(h["sim_return"] for h in history) - h0
    print(f"{name} seed {args.seed}: val start {h0:.3f} end {hT:.3f} worst dip {dip:+.3f} | "
          f"test {test['sim_return']:.3f}")


if __name__ == "__main__":
    main()
