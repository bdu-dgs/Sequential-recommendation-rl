"""Transition storage for offline logs and online interaction.

``next_obs`` is not stored: in both environments the next observation is a deterministic function of
(obs, action, observed reward) — the history window shifts left and appends (action, reward), the action is written
into ``ep_items`` at position ``step`` and ``step`` increments. ``next_obs_torch`` rebuilds it exactly
(``tests/test_replay.py`` checks this against the environment).

Each transition also stores the behaviour probability of the logged action (``prop``; NaN when unknown), the true
(noise-free) simulated reward, the Monte-Carlo discounted return-to-go (filled when the episode ends; used by
Cal-QL and by analysis) and a ``source`` flag (0 = offline log, 1 = online interaction).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..environments.base import EnvSpec

FIELDS = {
    "hist_items": np.int32, "hist_rewards": np.float32, "step": np.int16, "ep_items": np.int16,
    "action": np.int32, "reward": np.float32, "true_reward": np.float32, "done": np.bool_,
    "prop": np.float32, "rtg": np.float32, "source": np.uint8, "user_row": np.int32, "episode": np.int64,
}


class ReplayBuffer:
    def __init__(self, spec: EnvSpec, capacity: int):
        self.spec, self.capacity = spec, capacity
        K, H = spec.hist_len, spec.horizon
        shapes = {"hist_items": (K,), "hist_rewards": (K,), "ep_items": (H,)}
        self.data = {k: np.zeros((capacity,) + shapes.get(k, ()), dtype=t) for k, t in FIELDS.items()}
        self.size = 0
        self.ptr = 0
        self.n_episodes = 0
        self._open: dict[int, list[int]] = {}  # env slot -> buffer indices of the running episode
        self._ep_id: dict[int, int] = {}  # env slot -> id of its running episode

    def __len__(self) -> int:
        return self.size

    def add_batch(self, obs: dict, actions, rewards, true_rewards, dones, props, source: int, slots, user_rows,
                  gamma: float) -> None:
        """Add one transition per slot in ``slots`` (indices into the batch arrays)."""
        for j in slots:
            i = self.ptr
            d = self.data
            d["hist_items"][i] = obs["hist_items"][j]
            d["hist_rewards"][i] = obs["hist_rewards"][j]
            d["step"][i] = obs["step"][j]
            d["ep_items"][i] = obs["ep_items"][j]
            d["action"][i] = actions[j]
            d["reward"][i] = rewards[j]
            d["true_reward"][i] = true_rewards[j]
            d["done"][i] = dones[j]
            d["prop"][i] = props[j] if props is not None else np.nan
            d["rtg"][i] = np.nan
            d["source"][i] = source
            d["user_row"][i] = user_rows[j]
            ep = self._open.setdefault(int(j), [])
            if not ep:
                self.n_episodes += 1
                self._ep_id[int(j)] = self.n_episodes
            d["episode"][i] = self._ep_id[int(j)]
            ep.append(i)
            if dones[j]:
                g = 0.0
                for k in reversed(ep):
                    g = d["reward"][k] + gamma * g
                    d["rtg"][k] = g
                self._open[int(j)] = []
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample_idx(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(0, self.size, size=n)

    def batch(self, idx: np.ndarray, device) -> dict:
        d = self.data
        out = {k: torch.as_tensor(d[k][idx], device=device) for k in d}
        for k in ("hist_items", "step", "ep_items", "action"):
            out[k] = out[k].long()
        return out

    def save(self, path: Path) -> None:
        np.savez_compressed(path, size=self.size, ptr=self.ptr, n_episodes=self.n_episodes, capacity=self.capacity,
                            **{k: v[:self.size] for k, v in self.data.items()})

    @classmethod
    def load(cls, path: Path, spec: EnvSpec, capacity: int | None = None) -> "ReplayBuffer":
        z = np.load(path)
        size = int(z["size"])
        buf = cls(spec, max(capacity or size, size))
        for k in FIELDS:
            buf.data[k][:size] = z[k][:size]
        full = "capacity" in z and size == int(z["capacity"])  # the saved ring had filled up (maybe wrapped)
        buf.size, buf.n_episodes = size, int(z["n_episodes"])
        buf.ptr = (int(z["ptr"]) if full else size) % buf.capacity
        return buf


def obs_from_batch(b: dict) -> dict:
    B = b["action"].shape[0]
    return {"hist_items": b["hist_items"], "hist_rewards": b["hist_rewards"],
            "user_feat": torch.zeros(B, 0, device=b["action"].device), "step": b["step"], "ep_items": b["ep_items"]}


def next_obs_torch(b: dict, spec: EnvSpec) -> dict:
    a = b["action"].long()
    hist_items = torch.cat([b["hist_items"][:, 1:], a[:, None]], dim=1)
    hist_rewards = torch.cat([b["hist_rewards"][:, 1:], b["reward"][:, None]], dim=1)
    step = b["step"] + 1
    ep_items = b["ep_items"].clone()
    pos = b["step"].clamp(max=spec.horizon - 1)
    ep_items.scatter_(1, pos[:, None], a[:, None])
    return {"hist_items": hist_items, "hist_rewards": hist_rewards,
            "user_feat": torch.zeros(a.shape[0], 0, device=a.device), "step": step, "ep_items": ep_items}
