"""Tiny, fully known recommendation MDP used to sanity-check RL updates and return computation.

- 8 items in two categories (A = items 0-3, B = items 4-7); items cannot be repeated within an episode.
- Two hidden user types with mirrored preferences. Type 0 values A items at ``hi`` and B items at ``lo``;
  type 1 the opposite. Every first reward therefore reveals the type.
- Fatigue/quit rule: if the recommended item has the same category as the previous two recommendations, the user
  leaves before watching it (reward 0, episode ends).
- Episodes also end after ``horizon`` steps. ``step`` is part of the observation, so the time limit is a true
  terminal, not a truncation.

A greedy (bandit) policy takes the top three items of the preferred category and is cut off; the optimal policy
interleaves categories two at a time. ``optimal_value`` computes the exact optimum by dynamic programming.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .base import EnvSpec

HI = np.array([1.0, 0.9, 0.8, 0.7])
LO = np.array([0.6, 0.5, 0.4, 0.3])
N_ITEMS = 8
CAT = np.array([0, 0, 0, 0, 1, 1, 1, 1])


def reward_table() -> np.ndarray:
    """[2 types, 8 items] immediate rewards."""
    return np.stack([np.concatenate([HI, LO]), np.concatenate([LO, HI])])


class ToyRecEnv:
    def __init__(self, n_envs: int, horizon: int = 8, hist_len: int = 4, gamma: float = 0.9, seed: int = 0):
        self.spec = EnvSpec(n_items=N_ITEMS, hist_len=hist_len, horizon=horizon, user_feat_dim=0, gamma=gamma)
        self.n_envs = n_envs
        self.rng = np.random.default_rng(seed)
        self.R = reward_table()

    def reset(self) -> dict:
        B, H, pad = self.n_envs, self.spec.horizon, self.spec.pad
        self.types = np.zeros(B, dtype=np.int64)
        self.t = np.zeros(B, dtype=np.int64)
        self.ep_items = np.full((B, H), pad, dtype=np.int64)
        self.ep_rewards = np.zeros((B, H), dtype=np.float32)
        self.done = np.zeros(B, dtype=bool)
        self.reset_slots(np.arange(B))
        return self._obs()

    def _obs(self) -> dict:
        B, K = self.n_envs, self.spec.hist_len
        hist_items = np.full((B, K), self.spec.pad, dtype=np.int64)
        hist_rewards = np.zeros((B, K), dtype=np.float32)
        for b in range(B):
            n = min(self.t[b], K)
            if n:
                hist_items[b, K - n:] = self.ep_items[b, self.t[b] - n:self.t[b]]
                hist_rewards[b, K - n:] = self.ep_rewards[b, self.t[b] - n:self.t[b]]
        return {
            "hist_items": hist_items,
            "hist_rewards": hist_rewards,
            "user_feat": np.zeros((B, 0), dtype=np.float32),
            "step": self.t.copy(),
            "ep_items": self.ep_items.copy(),
        }

    def reset_slots(self, idx: np.ndarray) -> None:
        idx = np.asarray(idx)
        self.types[idx] = self.rng.integers(0, 2, size=len(idx))
        self.t[idx] = 0
        self.ep_items[idx] = self.spec.pad
        self.ep_rewards[idx] = 0.0
        self.done[idx] = False

    def obs(self) -> dict:
        return self._obs()

    def step(self, actions: np.ndarray):
        """Returns (next_obs, reward, done, info). Finished slots are left untouched (reward 0)."""
        actions = np.asarray(actions, dtype=np.int64)
        B = self.n_envs
        act = np.flatnonzero(~self.done)
        assert (self.ep_items[act] != actions[act, None]).all(), "repeated item"
        r = np.zeros(B, dtype=np.float32)
        r[act] = self.R[self.types[act], actions[act]]
        quit_ = np.zeros(B, dtype=bool)
        for b in act:
            t = self.t[b]
            if t >= 2 and CAT[self.ep_items[b, t - 1]] == CAT[self.ep_items[b, t - 2]] == CAT[actions[b]]:
                quit_[b] = True
        r[quit_] = 0.0
        self.ep_items[act, self.t[act]] = actions[act]
        self.ep_rewards[act, self.t[act]] = r[act]
        self.t[act] += 1
        self.done[act] = quit_[act] | (self.t[act] >= self.spec.horizon)
        return self._obs(), r, self.done.copy(), {"quit": quit_, "true_reward": r, "active": act}


@lru_cache(maxsize=None)
def _v(user_type: int, used: int, c1: int, c2: int, t: int, horizon: int, gamma: float) -> float:
    """Optimal value with known user type. used = bitmask of used items, c1/c2 = last two categories (-1 none)."""
    if t >= horizon:
        return 0.0
    R = reward_table()
    best = 0.0 if used == (1 << N_ITEMS) - 1 else -np.inf
    for a in range(N_ITEMS):
        if used >> a & 1:
            continue
        c = int(CAT[a])
        if c1 == c2 == c and c1 >= 0:
            q = 0.0  # quit, terminal
        else:
            q = R[user_type, a] + gamma * _v(user_type, used | (1 << a), c, c1, t + 1, horizon, gamma)
        best = max(best, q)
    return best


def optimal_value(horizon: int = 8, gamma: float = 0.9) -> float:
    """Exact optimal expected return from the initial state (type ~ Uniform{0,1}, unobserved at t=0)."""
    R = reward_table()
    best = -np.inf
    for a in range(N_ITEMS):
        c = int(CAT[a])
        q = np.mean([R[u, a] + gamma * _v(u, 1 << a, c, -1, 1, horizon, gamma) for u in (0, 1)])
        best = max(best, q)
    return float(best)


def greedy_value(horizon: int = 8, gamma: float = 0.9) -> float:
    """Return of the type-aware myopic policy (always the best remaining item): 1.0, 0.9, then quit."""
    return float(HI[0] + gamma * HI[1])
