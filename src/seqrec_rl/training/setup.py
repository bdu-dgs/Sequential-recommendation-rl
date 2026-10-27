"""Builds environments, the logging policy, datasets and agents from an experiment config."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .. import DATA_DIR, PROJECT_ROOT, PROCESSED_DIR
from ..agents.bandit import NeuralLinearUCB
from ..agents.q_agents import BCAgent, PopularityAgent, QAgent, QConfig, RandomAgent, RewardRegressionAgent
from ..agents.support import SupportWeights
from ..environments.kuairec_sim import KuaiRecSim, SimConfig, build_sim_data
from ..evaluation.rollout import collect, evaluate
from ..replay.buffer import ReplayBuffer
from ..replay.logging_policy import LoggingConfig, LoggingPolicy, load_response_model

OFFLINE_DIR = DATA_DIR / "offline"


def device() -> str:
    return "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"


class Setup:
    def __init__(self, cfg: dict, seed: int):
        self.cfg, self.seed = cfg, seed
        self.dev = device()
        self.sim_cfg = SimConfig(**cfg["sim"])
        self.data = build_sim_data(self.sim_cfg.hist_len, self.sim_cfg.reward_cap)
        self.rows = self.data.env_split_rows
        self.item_pop = np.load(PROCESSED_DIR / "item_train_pop.npy")
        cat_pop = self.item_pop[:self.data.n_items]
        self.item_tail = (cat_pop < np.quantile(cat_pop, 0.8)).astype(np.float32)  # outside the top-20% head
        self._logging = None

    def env(self, split: str, n_envs: int, seed_offset: int = 0, **overrides) -> KuaiRecSim:
        sc = SimConfig(**{**self.sim_cfg.to_dict(), **overrides})
        if sc.response == "learned" and sc.learned_response_path:
            sc.learned_response_path = str(PROJECT_ROOT / sc.learned_response_path)
        return KuaiRecSim(self.data, sc, self.rows[split], n_envs, seed=self.seed * 1000 + seed_offset)

    @property
    def spec(self):
        spec = self.env("train", 1).spec
        if self.cfg["agent"].get("history_repr", "tags") == "ids":
            spec.item_tags = None  # per-item ID embeddings for every history item (v0 representation)
        return spec

    def logging_policy(self) -> LoggingPolicy:
        if self._logging is None:
            m = load_response_model(PROJECT_ROOT / self.cfg["response_model"], self.data.n_hist_items,
                                    device=self.dev)
            self._logging = LoggingPolicy(m, self.spec, LoggingConfig(**self.cfg["logging"]), self.item_pop,
                                          self.dev)
        return self._logging

    # ------------------------------------------------------------------------------------------------
    def dataset_name(self) -> str:
        """Readable prefix + hash of everything that changes the generated data. The response source of the
        *training* simulator is excluded on purpose: offline logs are always generated in the FO simulator."""
        import hashlib
        import json
        lg, sc = self.cfg["logging"], self.cfg["sim"]
        key = {"sim": {k: v for k, v in sc.items() if k not in ("response", "learned_response_path")},
               "logging": lg, "offline_data": self.cfg["offline_data"], "scorer": self.cfg["response_model"]}
        h = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:8]
        return (f"T{lg['temperature']}_eps{lg['eps']}_head{lg['head_size']}_fat{sc['fatigue_beta']}_"
                f"pat{sc['patience']}_noise{sc['noise_std']}_n{self.cfg['offline_data']['transitions']}_"
                f"{h}_s{self.seed}")

    def offline_buffer(self, generate: bool = True) -> ReplayBuffer:
        path = OFFLINE_DIR / f"{self.dataset_name()}.npz"
        if path.exists():
            return ReplayBuffer.load(path, self.spec)
        if not generate:
            raise FileNotFoundError(path)
        n = self.cfg["offline_data"]["transitions"]
        env = self.env("train", self.cfg["offline_data"]["n_envs"], seed_offset=1, response="fo",
                       learned_response_path=None)
        buf = ReplayBuffer(env.spec, n + env.n_envs)
        rng = np.random.default_rng(self.seed + 12345)
        lp = self.logging_policy()
        collect(env, lambda o, r: lp.sample(o, r), n, rng, buf, source=0, gamma=self.sim_cfg.gamma)
        OFFLINE_DIR.mkdir(parents=True, exist_ok=True)
        buf.save(path)
        return buf

    # ------------------------------------------------------------------------------------------------
    def q_config(self, **kw) -> QConfig:
        a = self.cfg["agent"]
        base = dict(lr=a["lr"], gamma=self.sim_cfg.gamma, emb_dim=a["emb_dim"], hidden=a["hidden"],
                    dropout=a.get("dropout", 0.0), weight_decay=a.get("weight_decay", 0.0))
        base.update(kw)
        return QConfig(**base)

    def make_agent(self, method: str, offline_buf: ReplayBuffer | None = None, support_fn=None):
        a, spec = self.cfg["agent"], self.spec
        if method == "random":
            return RandomAgent(spec, self.dev)
        if method == "popularity":
            counts = np.bincount(offline_buf.data["action"][:offline_buf.size], minlength=spec.n_items)
            return PopularityAgent(spec, counts, self.dev)
        if method == "bc":
            return BCAgent(spec, a["lr"], a["emb_dim"], a["hidden"], self.dev, a.get("dropout", 0.0),
                           a.get("weight_decay", 0.0))
        if method == "ranker":
            return RewardRegressionAgent(spec, a["lr"], a["emb_dim"], a["hidden"], self.dev, a.get("dropout", 0.0),
                                         a.get("weight_decay", 0.0))
        if method == "dqn":
            return QAgent(spec, self.q_config(), self.dev)
        if method == "cql":
            return QAgent(spec, self.q_config(cql_alpha=self.cfg["cql"]["alpha"]), self.dev, support_fn)
        if method == "calql":
            return QAgent(spec, self.q_config(cql_alpha=self.cfg["cql"]["alpha"], calql=True), self.dev)
        if method == "bcq":
            return QAgent(spec, self.q_config(bcq_tau=self.cfg["bcq"]["tau"]), self.dev)
        raise ValueError(method)

    def bandit(self, ranker: RewardRegressionAgent) -> NeuralLinearUCB:
        b = self.cfg["bandit"]
        return NeuralLinearUCB(ranker, lam=b["lam"], beta=b["beta"])

    def support(self, source: str, prob_fn=None, const: float = 0.5) -> SupportWeights:
        s = self.cfg["support"]
        if prob_fn is None and source in ("oracle", "shuffled", "inverted"):
            prob_fn = self.logging_policy().probs_t
        return SupportWeights(self.spec.n_items, source, prob_fn, kappa=s["kappa"], lam=s["lam"], const=const,
                              device=self.dev, seed=self.seed)

    # ------------------------------------------------------------------------------------------------
    def eval_fn(self, split: str = "val", n_users: int | None = None, **env_overrides):
        """Evaluation env = training sim config, overridden by any SimConfig keys under cfg['eval'] (e.g.
        eval.response=fo evaluates in the real FO simulator while training interacts with the learned one),
        then by explicit ``env_overrides``."""
        sim_keys = set(SimConfig.__dataclass_fields__)
        env_overrides = {**{k: v for k, v in self.cfg.get("eval", {}).items() if k in sim_keys}, **env_overrides}
        rows = self.rows[split] if n_users is None else self.rows[split][:n_users]
        env = self.env(split, min(self.cfg["eval"]["n_envs"], len(rows)), seed_offset=7, **env_overrides)
        lp_probs = self.logging_policy().probs_np_obs

        def f(agent, keep_trajectories: int = 0):
            return evaluate(env, agent, rows, np.random.default_rng(0), self.item_tail, lp_probs,
                            keep_trajectories=keep_trajectories)
        return f
