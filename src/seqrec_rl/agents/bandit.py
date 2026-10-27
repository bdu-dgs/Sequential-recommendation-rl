"""NeuralLinear-UCB contextual bandit (Riquelme et al. 2018 NeuralLinear; LinUCB-style bonus, Li et al. 2010).

Starts from the offline reward-regression ranker r(s, a) = phi(s) . w_a + b_a. With a shared linear model
    r(s, a) = theta^T x(s, a) + b_a,   x(s, a) = phi(s) * w_a  (elementwise),
and prior theta ~ N(1, (1/lam) I) (so the prior mean reproduces the pretrained ranker), the posterior after online
observations is A = lam I + sum x x^T,  theta_hat = A^-1 (lam 1 + sum x (r - b_a)).
Acting: argmax theta_hat^T x + b_a + beta sqrt(x^T A^-1 x)   (explore=True)  /  posterior mean (explore=False).
The bandit is myopic (gamma = 0); features phi and item vectors w are frozen; each online transition updates
the posterior exactly once (no replay).
"""

from __future__ import annotations

import numpy as np
import torch

from ..environments.base import action_mask, obs_to_torch
from .base import Agent
from .q_agents import RewardRegressionAgent


class NeuralLinearUCB(Agent):
    name = "bandit"

    def __init__(self, ranker: RewardRegressionAgent, lam: float = 1.0, beta: float = 1.0):
        super().__init__(ranker.spec, ranker.device)
        self.ranker = ranker
        self.W = ranker.net.head.weight.detach()  # [A, d]
        self.b = ranker.net.head.bias.detach()  # [A]
        d = self.W.shape[1]
        self.lam, self.beta = lam, beta
        self.A = lam * torch.eye(d, device=self.device, dtype=torch.float64)
        self.rhs = lam * torch.ones(d, device=self.device, dtype=torch.float64)
        self._refresh()

    def _refresh(self):
        self.A_inv = torch.linalg.inv(self.A)
        self.theta = (self.A_inv @ self.rhs).float()
        self.A_inv_f = self.A_inv.float()

    def modules(self):
        return {"ranker_net": self.ranker.net}

    def state_dict(self):
        return {"ranker_net": self.ranker.net.state_dict(), "A": self.A.cpu(), "rhs": self.rhs.cpu()}

    def load_state_dict(self, sd, strict_keys=None):
        self.ranker.net.load_state_dict(sd["ranker_net"])
        self.A, self.rhs = sd["A"].to(self.device), sd["rhs"].to(self.device)
        self._refresh()

    @torch.no_grad()
    def _scores(self, obs_t, ucb: bool):
        phi = self.ranker.features(obs_t)  # [B, d]
        mean = (phi * self.theta) @ self.W.T + self.b  # [B, A]
        if not ucb:
            return mean
        X = phi[:, None, :] * self.W[None, :, :]  # [B, A, d]
        var = ((X @ self.A_inv_f) * X).sum(-1).clamp(min=0)
        return mean + self.beta * var.sqrt()

    def scores(self, obs_t):
        return self._scores(obs_t, ucb=False)

    @torch.no_grad()
    def act(self, obs, rng, explore=False):
        obs_t = obs_to_torch(obs, self.device)
        s = self._scores(obs_t, ucb=explore)
        mask = action_mask(obs_t["ep_items"], self.spec.n_items)
        return s.masked_fill(~mask, -torch.inf).argmax(1).cpu().numpy()

    @torch.no_grad()
    def observe(self, obs, actions, rewards):
        obs_t = obs_to_torch(obs, self.device)
        a = torch.as_tensor(actions, device=self.device)
        x = (self.ranker.features(obs_t) * self.W[a]).double()
        r = torch.as_tensor(rewards, device=self.device, dtype=torch.float64) - self.b[a].double()
        self.A += x.T @ x
        self.rhs += x.T @ r
        self._refresh()
