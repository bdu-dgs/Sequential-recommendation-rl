"""Write the job lists for steps 5-8 (one shell command per line) to experiments/jobs/<stage>.txt.

Stages (each depends on the previous ones — see ris/submit_all.sh):
  s5a_data      offline datasets for the base config (seeds 0-4)
  s5b_offline   offline baselines (step 5)
  s6_online     online / offline-to-online methods (step 6)
  s7a_data      datasets for coverage and environment-mechanism ablations
  s7b_offline   offline initialisations for those ablations
  s7c_online    fine-tuning ablations (step 7)
  s8a_online    simulator-exploitation runs: train in the learned simulator (step 8)
  s8b_eval      evaluation-only: generalisation stress tests and OPE (step 8)
  diag          mechanism analysis: Q over-estimation by support during fine-tuning (seeds 0-2)

Usage: python scripts/make_jobs.py [--seeds 0 1 2 3 4] [--py python]
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "experiments" / "jobs"

OFFLINE_METHODS = ["random", "popularity", "behavior", "ranker", "bc", "dqn", "cql", "calql", "bcq"]
ONLINE_METHODS = ["scratch_dqn", "scratch_rlpd", "bandit", "ft_dqn", "ft_cql", "ft_cql_decay", "ft_calql",
                  "ft_support_oracle", "ft_support_bc", "ft_support_uniform", "ft_support_uniform_tracked",
                  "ft_support_shuffled", "ft_support_inverted"]

# datasets that change the logging policy or the environment: tag -> overrides
COVERAGE = {
    "narrow": ["logging.eps=0.0", "logging.head_size=500"],  # zero support outside a 500-item head
    "broad": ["logging.eps=0.2", "logging.temperature=0.3"],
}
MECHANISM = {
    "fat0": ["sim.fatigue_beta=0.0"],
    "fat06": ["sim.fatigue_beta=0.6"],
    "pat2": ["sim.patience=2"],
    "pat5": ["sim.patience=5"],
    "noise03": ["sim.noise_std=0.3"],
}
ABLATION_ONLINE = ["ft_dqn", "ft_cql", "ft_cql_decay", "ft_support_oracle", "ft_support_uniform",
                   "ft_support_uniform_tracked", "ft_support_bc"]
MECHANISM_ONLINE = ["ft_dqn", "ft_cql", "ft_support_oracle", "ft_support_uniform_tracked"]


def cmd(py, script, seed, args=(), overrides=()):
    parts = [py, f"scripts/{script}", f"--seed {seed}", *args, *overrides]
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--mech_seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--py", default="python")
    a = ap.parse_args()
    py, S, MS = a.py, a.seeds, a.mech_seeds
    jobs: dict[str, list[str]] = {k: [] for k in
                                  ("s5a_data", "s5b_offline", "s6_online", "s7a_data", "s7b_offline", "s7c_online",
                                   "s8a_online", "s8b_eval", "diag")}
    # ---- step 5 / 6: base config
    for s in S:
        jobs["s5a_data"].append(cmd(py, "generate_data.py", s))
        for m in OFFLINE_METHODS:
            jobs["s5b_offline"].append(cmd(py, "run_offline.py", s, [f"--method {m}"]))
        for m in ONLINE_METHODS:
            jobs["s6_online"].append(cmd(py, "run_online.py", s, [f"--method {m}"]))

    # ---- step 7a: coverage regimes (5 seeds)
    for tag, ov in COVERAGE.items():
        for s in S:
            jobs["s7a_data"].append(cmd(py, "generate_data.py", s, [], ov))
            for m in ("cql", "bc", "behavior", "bcq"):
                jobs["s7b_offline"].append(cmd(py, "run_offline.py", s, [f"--method {m}", f"--tag {tag}"], ov))
            for m in ABLATION_ONLINE:
                jobs["s7c_online"].append(cmd(py, "run_online.py", s,
                                              [f"--method {m}", f"--tag {tag}", f"--init_tag {tag}"], ov))
    # ---- step 7b: environment mechanisms (3 seeds)
    for tag, ov in MECHANISM.items():
        for s in MS:
            jobs["s7a_data"].append(cmd(py, "generate_data.py", s, [], ov))
            for m in ("cql", "behavior"):
                jobs["s7b_offline"].append(cmd(py, "run_offline.py", s, [f"--method {m}", f"--tag {tag}"], ov))
            for m in MECHANISM_ONLINE:
                jobs["s7c_online"].append(cmd(py, "run_online.py", s,
                                              [f"--method {m}", f"--tag {tag}", f"--init_tag {tag}"], ov))
    # ---- step 7c: conservatism strength, support temperature, online counts, inheritance (base data)
    for s in S:
        for alpha in (0.3, 3.0):
            for m in ("ft_cql", "ft_support_oracle", "ft_support_uniform_tracked"):
                jobs["s7c_online"].append(cmd(py, "run_online.py", s, [f"--method {m}", f"--tag alpha{alpha}"],
                                              [f"cql.alpha={alpha}"]))
        for kappa in (0.3, 3.0):
            jobs["s7c_online"].append(cmd(py, "run_online.py", s, ["--method ft_support_oracle",
                                                                  f"--tag kappa{kappa}"], [f"support.kappa={kappa}"]))
        for m in ("ft_support_oracle", "ft_support_uniform", "ft_support_shuffled"):
            jobs["s7c_online"].append(cmd(py, "run_online.py", s, [f"--method {m}", "--tag lam0"],
                                          ["support.lam=0.0"]))
        for tag, ov in {"keepopt": ["online.keep_optimizer=true"], "resettarget": ["online.keep_target=false"],
                        "offr0": ["online.offline_ratio=0.0"], "offr025": ["online.offline_ratio=0.25"]}.items():
            for m in ("ft_dqn", "ft_support_oracle"):
                jobs["s7c_online"].append(cmd(py, "run_online.py", s, [f"--method {m}", f"--tag {tag}"], ov))

    # ---- step 8a: simulator exploitation — interact with the learned simulator, evaluate in the FO simulator
    learned = ["sim.response=learned", "sim.learned_response_path=experiments/runs/response_model_wd/seed0/fo_pred_wr.npy",
               "eval.response=fo"]
    for s in S:
        for m in ("scratch_dqn", "ft_dqn", "ft_cql", "ft_support_oracle"):
            jobs["s8a_online"].append(cmd(py, "run_online.py", s, [f"--method {m}", "--tag learnedsim"], learned))
    # ---- step 8b: evaluation only
    for s in S:
        jobs["s8b_eval"].append(cmd(py, "evaluate_checkpoints.py", s))
        jobs["s8b_eval"].append(cmd(py, "run_ope.py", s))

    # ---- mechanism analysis (re-runs with the q_diag evaluation; tables via plot_overestimation.py)
    for s in MS:
        for m in MECHANISM_ONLINE:
            jobs["diag"].append(cmd(py, "run_online.py", s, [f"--method {m}", "--tag diag"]))

    JOBS.mkdir(parents=True, exist_ok=True)
    for k, v in jobs.items():
        (JOBS / f"{k}.txt").write_text("\n".join(v) + "\n")
        print(f"{k:12s} {len(v):4d} commands")


if __name__ == "__main__":
    main()
