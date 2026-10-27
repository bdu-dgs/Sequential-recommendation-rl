"""Simulated logging (behaviour) policy with exact, recorded propensities.

pi_b(a|s) = (1 - eps) * softmax_{a in M(s) & S}( f(s, a) / temperature ) + eps * Uniform(M(s))

- f(s, a): predicted capped watch ratio from the supervised response model trained on REAL KuaiRec logs, applied
  to the observable history in s (a realistic "production ranker").
- M(s): items not yet recommended in the episode.
- S: support set of the softmax part — all items, or only the `head_size` most popular catalog items in the real
  train window (models a production system that rarely shows the long tail).
- eps: uniform exploration mass; eps = 0 with a head-only S creates actions with zero support.

Because the policy is a deterministic function of the observation, pi_b(.|s) can be recomputed exactly for any
logged state ("oracle support").
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from ..environments.base import EnvSpec, action_mask, obs_to_torch
from ..user_models.response_model import ResponseModel


@dataclass
class LoggingConfig:
    temperature: float = 0.1
    eps: float = 0.05
    head_size: int | None = None  # None = all items

    def to_dict(self):
        return asdict(self)


class LoggingPolicy:
    def __init__(self, model: ResponseModel, spec: EnvSpec, cfg: LoggingConfig, item_pop: np.ndarray,
                 device="cpu"):
        self.model, self.spec, self.cfg = model.eval(), spec, cfg
        self.device = torch.device(device)
        head = np.ones(spec.n_items, dtype=bool)
        if cfg.head_size is not None:
            head[:] = False
            head[np.argsort(-item_pop[:spec.n_items], kind="stable")[:cfg.head_size]] = True
        self.head = torch.as_tensor(head, device=self.device)

    @torch.no_grad()
    def probs_t(self, obs_t: dict) -> torch.Tensor:
        mask = action_mask(obs_t["ep_items"], self.spec.n_items)
        wr, _ = self.model.score_all(obs_t["hist_items"].long(), obs_t["hist_rewards"].float(), self.spec.n_items)
        soft_mask = mask & self.head
        logits = (wr / self.cfg.temperature).masked_fill(~soft_mask, -torch.inf)
        p_soft = torch.softmax(logits, dim=1)
        p_soft = torch.nan_to_num(p_soft, nan=0.0)  # rows where the head is exhausted
        n_allowed = mask.sum(1, keepdim=True).clamp(min=1)
        p_unif = mask.float() / n_allowed
        no_head = ~soft_mask.any(1, keepdim=True)
        eps = torch.where(no_head, torch.ones_like(n_allowed, dtype=torch.float32),
                          torch.full_like(n_allowed, self.cfg.eps, dtype=torch.float32))
        return (1 - eps) * p_soft + eps * p_unif

    def probs_np_obs(self, obs: dict) -> torch.Tensor:
        return self.probs_t(obs_to_torch(obs, self.device))

    def sample(self, obs: dict, rng: np.random.Generator):
        p = self.probs_t(obs_to_torch(obs, self.device)).double().cpu().numpy()
        p /= p.sum(1, keepdims=True)
        u = rng.random(len(p))[:, None]
        a = (p.cumsum(1) < u).sum(1).clip(max=p.shape[1] - 1)
        return a, p[np.arange(len(a)), a].astype(np.float32)


def load_response_model(path, n_items_all: int, emb_dim=64, hidden=128, device="cpu") -> ResponseModel:
    m = ResponseModel(n_items_all, emb_dim, hidden)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return m.to(device).eval()
