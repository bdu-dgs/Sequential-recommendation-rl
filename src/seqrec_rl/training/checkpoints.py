"""Load trained agents from run directories for evaluation-only scripts."""

from __future__ import annotations

from .. import RUNS_DIR
from ..agents.base import Agent
from ..environments.base import obs_to_torch

Q_ONLINE = {"scratch_dqn", "scratch_rlpd", "ft_dqn", "ft_cql", "ft_cql_decay", "ft_calql", "ft_support_oracle",
            "ft_support_bc", "ft_support_uniform", "ft_support_uniform_tracked", "ft_support_shuffled", "ft_support_inverted"}


class LoggingAgent(Agent):
    name = "behavior"

    def __init__(self, lp, greedy=False):
        super().__init__(lp.spec, lp.device)
        self.lp, self.greedy = lp, greedy

    def act(self, obs, rng, explore=False):
        if self.greedy:
            return self.lp.probs_t(obs_to_torch(obs, self.device)).argmax(1).cpu().numpy()
        return self.lp.sample(obs, rng)[0]


def load_agent(S, kind: str, method: str, seed: int, tag: str = ""):
    """kind = 'offline' | 'online'. Returns None if the run does not exist."""
    d = RUNS_DIR / kind / (method + (f"_{tag}" if tag else "")) / f"seed{seed}"
    if method == "behavior":
        return LoggingAgent(S.logging_policy())
    if method == "random":
        return S.make_agent("random")
    if method == "popularity":
        return S.make_agent("popularity", S.offline_buffer(generate=False))
    path = d / ("final_agent.pt" if kind == "offline" else "ckpt/agent.pt")
    if not path.exists():
        return None
    if kind == "offline":
        agent = S.make_agent(method)
        agent.load(path)
        return agent
    if method == "bandit":
        agent = S.bandit(S.make_agent("ranker"))
        agent.load(path)
        return agent
    if method in Q_ONLINE:
        agent = S.make_agent("dqn")
        agent.load(path, keys=("q", "q_target"))
        return agent
    raise ValueError(method)
