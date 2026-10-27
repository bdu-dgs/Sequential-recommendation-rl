"""Value-based agents sharing one implementation: Double DQN, discrete CQL, discrete BCQ, and their fine-tuning
variants (Cal-QL-style calibrated penalty, support-weighted penalty).

TD loss (all variants): Huber(Q(s,a) - y), Double-DQN target
    y = r + gamma * (1 - done) * Q_target(s', argmax_{a' in M(s')} Q(s', a'))
where M(s') = allowed (not yet recommended) items, further restricted by the BCQ filter when bcq_tau > 0:
    M_bcq(s') = {a : G(a|s') / max_a G(a|s') > tau}   (Fujimoto et al. 2019, discrete BCQ).

Conservative penalty (CQL(H), Kumar et al. 2020), generalised with a per-(s,a) weight c(s,a) in [0, 1]:
    L_pen = alpha * E_s[ sum_a sg(c(s,a) mu(a|s)) Q(s,a) - sg(sum_a c(s,a) mu(a|s)) Q(s, a_D) ],
    mu(.|s) = softmax over allowed items of Q(s,.)
With c == 1 the gradient equals that of alpha * (logsumexp_a Q(s,a) - Q(s,a_D)), i.e. standard discrete CQL.
Cal-QL (Nakamoto et al. 2023) variant: Q(s,a) inside the push-down term is replaced by max(Q(s,a), V_ref(s)) where
V_ref is the Monte-Carlo return-to-go of the logged trajectory, so values are not pushed below a reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..environments.base import EnvSpec, action_mask
from ..replay.buffer import next_obs_torch, obs_from_batch
from .base import Agent, clone_frozen, epsilon_greedy, soft_update
from .networks import ItemHeadNet


@dataclass
class QConfig:
    lr: float = 3e-4
    gamma: float = 0.95
    tau: float = 0.005  # Polyak target update
    huber_delta: float = 1.0
    cql_alpha: float = 0.0  # 0 => no conservative penalty (DQN)
    calql: bool = False
    bcq_tau: float = 0.0  # > 0 enables the discrete-BCQ action filter (requires a BC head)
    bc_weight: float = 1.0  # weight of the imitation (BC head) loss when BCQ is on
    emb_dim: int = 64
    hidden: int = 256
    dropout: float = 0.0
    weight_decay: float = 0.0
    grad_clip: float = 10.0
    eps_start: float = 0.2
    eps_end: float = 0.02
    eps_decay_steps: int = 20000


class QAgent(Agent):
    name = "q"

    def __init__(self, spec: EnvSpec, cfg: QConfig, device: str = "cpu", support_fn=None):
        super().__init__(spec, device)
        self.cfg = cfg
        self.q = ItemHeadNet(spec, cfg.emb_dim, cfg.hidden, cfg.dropout).to(self.device)
        self.q_target = clone_frozen(self.q).eval()
        self.bc = ItemHeadNet(spec, cfg.emb_dim, cfg.hidden, cfg.dropout).to(self.device) if cfg.bcq_tau > 0 else None
        self.reset_optimizer()
        # support_fn(obs_t, actions_or_None) -> c(s, .) [B, A] in [0, 1]; None => c == 1
        self.support_fn = support_fn
        self.alpha_scale = 1.0  # multiplies cfg.cql_alpha (used by schedules during fine-tuning)
        self.env_steps = 0

    # ----------------------------------------------------------------------------------------------
    def modules(self) -> dict:
        m = {"q": self.q, "q_target": self.q_target, "opt": self.opt}
        if self.bc is not None:
            m["bc"] = self.bc
        return m

    def allowed(self, obs_t: dict) -> torch.Tensor:
        mask = action_mask(obs_t["ep_items"], self.spec.n_items)
        if self.bc is not None and self.cfg.bcq_tau > 0:
            logits = self.bc(obs_t).masked_fill(~mask, -torch.inf)
            p = torch.softmax(logits, dim=1)
            keep = p / p.max(dim=1, keepdim=True).values > self.cfg.bcq_tau
            mask = mask & keep
        return mask

    def scores(self, obs_t: dict) -> torch.Tensor:
        q = self.q(obs_t)
        if self.bc is not None and self.cfg.bcq_tau > 0:
            q = q.masked_fill(~self.allowed(obs_t), -1e9)
        return q

    def epsilon(self) -> float:
        c = self.cfg
        frac = min(1.0, self.env_steps / max(1, c.eps_decay_steps))
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    def explore(self, greedy_a, mask, rng):
        return epsilon_greedy(greedy_a, mask, self.epsilon(), rng)

    # ----------------------------------------------------------------------------------------------
    def update(self, b: dict) -> dict:
        c = self.cfg
        obs = obs_from_batch(b)
        nobs = next_obs_torch(b, self.spec)
        a = b["action"]
        q_all = self.q(obs)
        q_sa = q_all.gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            nmask = self.allowed(nobs)
            # terminal transitions may have an empty next mask; their target ignores the bootstrap term
            empty = ~nmask.any(dim=1)
            nmask[empty, 0] = True
            na = self.q(nobs).masked_fill(~nmask, -torch.inf).argmax(dim=1)
            nq = self.q_target(nobs).gather(1, na[:, None]).squeeze(1)
            y = b["reward"] + c.gamma * (1.0 - b["done"].float()) * nq
        td = F.huber_loss(q_sa, y, delta=c.huber_delta)
        loss = td
        info = {"td": td.item(), "q_mean": q_sa.mean().item()}

        alpha = c.cql_alpha * self.alpha_scale
        if alpha > 0:
            mask = action_mask(obs["ep_items"], self.spec.n_items)
            q_pen = q_all
            if c.calql:
                vref = torch.nan_to_num(b["rtg"], nan=-1e9)[:, None]
                q_pen = torch.maximum(q_all, vref)
            qm = q_pen.masked_fill(~mask, -torch.inf)
            mu = torch.softmax(qm.detach(), dim=1)
            ref = getattr(self.support_fn, "reference", None)
            if ref is not None:  # "uniform_tracked": match the oracle's current mean strength on this batch
                self.support_fn.const = (mu * ref(obs)).sum(1).mean().item()
            w = mu if self.support_fn is None else mu * self.support_fn(obs).detach()
            push_down = (w * qm.masked_fill(~mask, 0.0)).sum(1)
            push_up = w.sum(1) * q_sa
            pen = (push_down - push_up).mean()
            loss = loss + alpha * pen
            info.update(cql_pen=pen.item(), cql_w=w.sum(1).mean().item())

        if self.bc is not None:
            mask = action_mask(obs["ep_items"], self.spec.n_items)
            logits = self.bc(obs).masked_fill(~mask, -1e9)
            bc_loss = F.cross_entropy(logits, a)
            loss = loss + c.bc_weight * bc_loss
            info["bc"] = bc_loss.item()

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        params = [p for g in self.opt.param_groups for p in g["params"]]
        torch.nn.utils.clip_grad_norm_(params, c.grad_clip)
        self.opt.step()
        soft_update(self.q_target, self.q, c.tau)
        return info

    def reset_optimizer(self) -> None:
        params = list(self.q.parameters()) + (list(self.bc.parameters()) if self.bc is not None else [])
        self.opt = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

    def train_mode(self, flag: bool) -> None:
        super().train_mode(flag)
        self.q_target.eval()  # bootstrap targets never use dropout


class BCAgent(Agent):
    """Behaviour cloning: softmax cross-entropy over allowed items."""

    name = "bc"

    def __init__(self, spec: EnvSpec, lr: float = 3e-4, emb_dim: int = 64, hidden: int = 256, device="cpu",
                 dropout: float = 0.0, weight_decay: float = 0.0):
        super().__init__(spec, device)
        self.net = ItemHeadNet(spec, emb_dim, hidden, dropout).to(self.device)
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)

    def modules(self):
        return {"net": self.net, "opt": self.opt}

    def scores(self, obs_t):
        return self.net(obs_t)

    def probs(self, obs_t) -> torch.Tensor:
        mask = action_mask(obs_t["ep_items"], self.spec.n_items)
        return torch.softmax(self.net(obs_t).masked_fill(~mask, -torch.inf), dim=1)

    def update(self, b):
        obs = obs_from_batch(b)
        mask = action_mask(obs["ep_items"], self.spec.n_items)
        loss = F.cross_entropy(self.net(obs).masked_fill(~mask, -1e9), b["action"])
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return {"bc": loss.item()}


class RewardRegressionAgent(Agent):
    """Supervised myopic ranker trained on the logs: regress the observed reward of the logged item
    (equivalently a Q-function with gamma = 0) and recommend the argmax."""

    name = "ranker"

    def __init__(self, spec: EnvSpec, lr: float = 3e-4, emb_dim: int = 64, hidden: int = 256, device="cpu",
                 dropout: float = 0.0, weight_decay: float = 0.0):
        super().__init__(spec, device)
        self.net = ItemHeadNet(spec, emb_dim, hidden, dropout).to(self.device)
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)

    def modules(self):
        return {"net": self.net, "opt": self.opt}

    def scores(self, obs_t):
        return self.net(obs_t)

    def features(self, obs_t) -> torch.Tensor:
        return self.net.encoder(obs_t)

    def update(self, b):
        obs = obs_from_batch(b)
        pred = self.net(obs).gather(1, b["action"][:, None]).squeeze(1)
        loss = F.mse_loss(pred, b["reward"])
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return {"mse": loss.item()}


class PopularityAgent(Agent):
    """Recommend the most frequently logged item not yet shown in the episode."""

    name = "popularity"

    def __init__(self, spec: EnvSpec, counts: np.ndarray, device="cpu"):
        super().__init__(spec, device)
        self.counts = torch.as_tensor(np.log1p(counts), dtype=torch.float32, device=self.device)

    def scores(self, obs_t):
        return self.counts.expand(obs_t["step"].shape[0], -1)


class RandomAgent(Agent):
    name = "random"

    def scores(self, obs_t):
        B = obs_t["step"].shape[0]
        return torch.rand(B, self.spec.n_items, device=self.device)
