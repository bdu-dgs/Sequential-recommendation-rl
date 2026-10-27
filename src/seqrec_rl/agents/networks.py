"""Shared state encoder and output heads.

State encoder (GRU4Rec-style): each history position is embedded as item_embedding + reward_projection; a GRU
reads the last K positions; the final hidden state is concatenated with a step embedding and user features and
passed through an MLP. Heads produce one output per item (Q-values, logits, or predicted rewards).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..environments.base import EnvSpec, action_mask


class StateEncoder(nn.Module):
    def __init__(self, spec: EnvSpec, emb_dim: int = 64, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.spec = spec
        self.use_tags = spec.item_tags is not None
        if self.use_tags:
            # ID embeddings for catalog (action) items only; every history item also gets a tag embedding
            self.item_emb = nn.Embedding(spec.n_items + 1, emb_dim, padding_idx=spec.n_items)
            tags = torch.as_tensor(np.asarray(spec.item_tags), dtype=torch.float32)
            tags = torch.cat([tags, torch.zeros(1, tags.shape[1])])  # padding row
            self.register_buffer("tags", tags, persistent=False)
            self.tag_proj = nn.Linear(tags.shape[1], emb_dim, bias=False)
        else:
            self.item_emb = nn.Embedding(spec.n_hist_items + 1, emb_dim, padding_idx=spec.hist_pad)
        self.rew_proj = nn.Linear(1, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True)
        self.step_emb = nn.Embedding(spec.horizon + 1, 16)
        in_dim = hidden + 16 + spec.user_feat_dim
        self.drop = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.out_dim = hidden

    def forward(self, obs: dict) -> torch.Tensor:
        items = obs["hist_items"].long()
        rew = obs["hist_rewards"].float().unsqueeze(-1)
        is_pad = (items == self.spec.hist_pad).unsqueeze(-1)
        if self.use_tags:
            cat_ids = torch.where(items < self.spec.n_items, items, torch.full_like(items, self.spec.n_items))
            emb = self.item_emb(cat_ids) + self.tag_proj(self.tags[items])
        else:
            emb = self.item_emb(items)
        x = self.drop(emb) + self.rew_proj(rew).masked_fill(is_pad, 0.0)
        _, h = self.gru(x)
        step = obs["step"].long().clamp(max=self.spec.horizon)
        z = torch.cat([h[-1], self.step_emb(step), obs["user_feat"].float()], dim=-1)
        return self.mlp(z)


class ItemHeadNet(nn.Module):
    """Encoder + linear head over all items. ``n_heads`` > 1 gives an ensemble of heads sharing nothing
    (each head has its own encoder) — used for bootstrapped / ensemble critics."""

    def __init__(self, spec: EnvSpec, emb_dim: int = 64, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.spec = spec
        self.encoder = StateEncoder(spec, emb_dim, hidden, dropout)
        self.head = nn.Linear(hidden, spec.n_items)

    def forward(self, obs: dict) -> torch.Tensor:
        return self.head(self.encoder(obs))


class EnsembleNet(nn.Module):
    def __init__(self, spec: EnvSpec, n_members: int, emb_dim: int = 64, hidden: int = 256):
        super().__init__()
        self.members = nn.ModuleList([ItemHeadNet(spec, emb_dim, hidden) for _ in range(n_members)])

    def forward(self, obs: dict) -> torch.Tensor:
        """[M, B, A]"""
        return torch.stack([m(obs) for m in self.members], dim=0)


def masked(values: torch.Tensor, obs: dict, n_items: int, fill: float = -1e9) -> torch.Tensor:
    mask = action_mask(obs["ep_items"], n_items)
    if values.dim() == 3:
        mask = mask.unsqueeze(0)
    return values.masked_fill(~mask, fill)
