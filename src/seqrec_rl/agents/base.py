"""Uniform agent interface used by every training and evaluation script."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from ..environments.base import EnvSpec, action_mask, obs_to_torch


class Agent:
    """act() never updates parameters; update() is the only method that changes the model."""

    name = "agent"

    def __init__(self, spec: EnvSpec, device: str = "cpu"):
        self.spec = spec
        self.device = torch.device(device)

    # --- acting -----------------------------------------------------------------------------------
    def scores(self, obs_t: dict) -> torch.Tensor:
        """Unmasked per-item scores [B, A] used for greedy action selection."""
        raise NotImplementedError

    @torch.no_grad()
    def act(self, obs: dict, rng: np.random.Generator, explore: bool = False) -> np.ndarray:
        """Acting always uses eval mode (no dropout); exploration comes only from ``explore``."""
        obs_t = obs_to_torch(obs, self.device)
        training = [m.training for m in self.modules().values() if isinstance(m, torch.nn.Module)]
        self.train_mode(False)
        s = self.scores(obs_t)
        for m, t in zip([m for m in self.modules().values() if isinstance(m, torch.nn.Module)], training):
            m.train(t)
        mask = action_mask(obs_t["ep_items"], self.spec.n_items)
        s = s.masked_fill(~mask, -torch.inf)
        a = s.argmax(dim=1).cpu().numpy()
        if explore:
            a = self.explore(a, mask.cpu().numpy(), rng)
        return a

    def explore(self, greedy_a: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return greedy_a

    # --- learning ---------------------------------------------------------------------------------
    def update(self, batch: dict) -> dict:
        return {}

    # --- persistence ------------------------------------------------------------------------------
    def modules(self) -> dict:
        """name -> nn.Module or torch.optim.Optimizer to checkpoint."""
        return {}

    def state_dict(self) -> dict:
        return {k: v.state_dict() for k, v in self.modules().items()}

    def load_state_dict(self, sd: dict, strict_keys: tuple | None = None) -> None:
        """Load modules from ``sd``; with ``strict_keys`` exactly those keys are loaded and must be present."""
        mods = self.modules()
        missing = [k for k in (strict_keys or ()) if k not in sd or k not in mods]
        if missing:
            raise KeyError(f"checkpoint/agent is missing {missing}")
        for k, v in sd.items():
            if k in mods and (strict_keys is None or k in strict_keys):
                mods[k].load_state_dict(v)

    def save(self, path: Path) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: Path, keys: tuple | None = None) -> None:
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=False), keys)

    def train_mode(self, flag: bool) -> None:
        for m in self.modules().values():
            if isinstance(m, torch.nn.Module):
                m.train(flag)


def epsilon_greedy(greedy_a: np.ndarray, mask: np.ndarray, eps: float, rng: np.random.Generator) -> np.ndarray:
    a = greedy_a.copy()
    flip = rng.random(len(a)) < eps
    for b in np.flatnonzero(flip):
        allowed = np.flatnonzero(mask[b])
        a[b] = rng.choice(allowed)
    return a


def hard_update(target: torch.nn.Module, source: torch.nn.Module) -> None:
    target.load_state_dict(source.state_dict())


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.mul_(1 - tau).add_(sp, alpha=tau)


def clone_frozen(m: torch.nn.Module) -> torch.nn.Module:
    t = copy.deepcopy(m)
    for p in t.parameters():
        p.requires_grad_(False)
    return t
