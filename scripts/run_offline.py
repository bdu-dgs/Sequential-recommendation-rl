"""Train one offline method on the simulated logs and evaluate it (SIMULATED returns).

Methods: random, popularity, behavior (the logging policy itself), ranker (reward regression), bc,
dqn (offline Double DQN, no conservatism), cql, bcq.

Usage: python scripts/run_offline.py --method cql --config configs/experiment/base.yaml --seed 0 [key=value ...]
Outputs: experiments/runs/offline/<method>[_<tag>]/seed<k>/{metrics.json, ckpt/, log.jsonl, config.yaml}
"""

from __future__ import annotations

import argparse

import numpy as np

from seqrec_rl.agents.base import Agent
from seqrec_rl.training.loops import train_offline
from seqrec_rl.training.runs import Run, limit_threads, load_config, set_seed
from seqrec_rl.training.setup import Setup

LEARNED = {"ranker", "bc", "dqn", "cql", "calql", "bcq"}


class BehaviorAgent(Agent):
    """The logging policy itself, acting stochastically (its true behaviour)."""
    name = "behavior"

    def __init__(self, lp):
        super().__init__(lp.spec, lp.device)
        self.lp = lp

    def act(self, obs, rng, explore=False):
        return self.lp.sample(obs, rng)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--config", default="configs/experiment/base.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    limit_threads(4)
    set_seed(args.seed)
    name = f"offline/{args.method}" + (f"_{args.tag}" if args.tag else "")
    run = Run(name, args.seed, cfg)
    S = Setup(cfg, args.seed)
    buf = S.offline_buffer()
    rng = np.random.default_rng(args.seed)
    val_eval = S.eval_fn("val")
    if args.method == "behavior":
        agent = BehaviorAgent(S.logging_policy())
    else:
        agent = S.make_agent(args.method, buf)
    history = []
    selection = None
    if args.method in LEARNED:
        history = train_offline(agent, buf, cfg["offline"]["steps"], cfg["agent"]["batch_size"], rng, run,
                                eval_fn=val_eval, eval_every=cfg["offline"]["eval_every"])
        agent.save(run.dir / "last_agent.pt")
        last_test = S.eval_fn("test")(agent)
        last_test.pop("returns", None)
        best = max(history, key=lambda h: h["sim_return"])
        agent.load(run.dir / "best_agent.pt")
        selection = {"rule": "max validation simulated return over checkpoints every offline.eval_every steps",
                     "selected_step": best["step"], "last_step": history[-1]["step"],
                     "last_step_val": history[-1]["sim_return"], "last_step_test": last_test}
        agent.save(run.dir / "final_agent.pt")  # = the selected checkpoint (used to initialise fine-tuning)
    test = S.eval_fn("test")(agent, keep_trajectories=20)
    val = val_eval(agent)
    metrics = {"method": args.method, "dataset": S.dataset_name(), "val": val, "test": test, "history": history,
               "selection": selection, "note": "All returns are simulated returns in KuaiRecFO-v1."}
    run.save_metrics(metrics)
    print(f"{name} seed {args.seed}: val return {val['sim_return']:.3f}  test return {test['sim_return']:.3f}  "
          f"test len {test['length']:.1f}  coverage {test['coverage']:.3f}")


if __name__ == "__main__":
    main()
