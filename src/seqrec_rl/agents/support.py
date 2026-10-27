"""Support signals c(s, a) in [0, 1] for the support-weighted conservative penalty (see q_agents.QAgent).

Support ratio rho(s, a) = p(a|s) * |M(s)|: the probability of a under the support model relative to uniform over
the allowed items. Conservatism weight c(s, a) = 1 / (1 + rho_total(s, a) / kappa), so well-supported actions get
little push-down and unsupported actions get the full CQL push-down.

During online fine-tuning, items the agent has actually tried accumulate online support:
    rho_total(s, a) = rho_offline(s, a) + lam * rho_online(a),  rho_online(a) = n_on(a) * A / max(N_on, 1)
(an item-level count ratio, since states are continuous).

Sources of p(a|s): "oracle" (the true logging policy), "bc" (a behaviour-cloning estimate),
"uniform" (constant c = c_bar matched to the oracle's mean strength at the start of fine-tuning),
"uniform_tracked" (constant across actions, but re-matched at every update to the oracle's current mean strength
E_s[sum_a mu(a|s) c_oracle(s,a)] — set by QAgent.update via ``reference``), "shuffled" (oracle weights read
through a FIXED random permutation of the items: same per-state weight distribution, consistently assigned to the
wrong items), "inverted" (c -> 1 - c).
"""

from __future__ import annotations

import numpy as np
import torch

from ..environments.base import action_mask


class SupportWeights:
    def __init__(self, n_items: int, source: str, prob_fn=None, kappa: float = 1.0, lam: float = 0.0,
                 const: float = 0.5, device="cpu", seed: int = 0):
        self.n_items, self.source, self.prob_fn = n_items, source, prob_fn
        self.kappa, self.lam, self.const = kappa, lam, const
        self.device = torch.device(device)
        self.online_counts = torch.zeros(n_items, device=self.device)
        gen = torch.Generator().manual_seed(seed)
        self.perm = torch.randperm(n_items, generator=gen).to(self.device)
        self.reference: "SupportWeights | None" = None  # oracle weights tracked by "uniform_tracked"

    def observe_online(self, actions: np.ndarray) -> None:
        self.online_counts += torch.bincount(torch.as_tensor(actions, device=self.device),
                                             minlength=self.n_items).float()
        if self.reference is not None:
            self.reference.observe_online(actions)

    @torch.no_grad()
    def __call__(self, obs_t: dict) -> torch.Tensor:
        mask = action_mask(obs_t["ep_items"], self.n_items)
        B = mask.shape[0]
        if self.source in ("uniform", "uniform_tracked"):
            return torch.full((B, self.n_items), self.const, device=self.device)
        p = self.prob_fn(obs_t)
        rho = p * mask.sum(1, keepdim=True)
        if self.lam > 0:
            n_on = self.online_counts
            rho = rho + self.lam * (n_on * self.n_items / n_on.sum().clamp(min=1.0))[None, :]
        c = 1.0 / (1.0 + rho / self.kappa)
        if self.source == "inverted":
            c = 1.0 - c
        elif self.source == "shuffled":
            c = c[:, self.perm]
        return c
