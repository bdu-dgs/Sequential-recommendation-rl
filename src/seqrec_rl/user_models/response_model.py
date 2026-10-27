"""Supervised user-response model trained on real KuaiRec logs.

Given the user's last K (item, capped watch ratio) pairs, predict the response to a candidate item:
  - ``wr``: capped watch ratio (regression, Huber loss)
  - ``eng``: engagement probability, engaged := watch_ratio >= 1 (complete play)
Two-tower scoring (user vector . item vector + item bias) so all catalog items can be scored in one matmul; the
same model serves as (i) the supervised ranker, (ii) the scorer of the simulated logging policy, and (iii) the
learned-response simulator used in the simulator-exploitation study.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResponseModel(nn.Module):
    def __init__(self, n_items_all: int, emb_dim: int = 64, hidden: int = 128, wr_cap: float = 2.0):
        super().__init__()
        self.pad = n_items_all
        self.wr_cap = wr_cap
        self.hist_emb = nn.Embedding(n_items_all + 1, emb_dim, padding_idx=n_items_all)
        self.rew_proj = nn.Linear(1, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True)
        self.user_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2 * emb_dim))
        self.item_vec = nn.Embedding(n_items_all, 2 * emb_dim)
        self.item_bias = nn.Embedding(n_items_all, 2)
        self.glob = nn.Parameter(torch.zeros(2))
        nn.init.normal_(self.item_vec.weight, std=0.01)
        nn.init.zeros_(self.item_bias.weight)
        self.emb_dim = emb_dim

    def user_vec(self, hist_items: torch.Tensor, hist_rewards: torch.Tensor) -> torch.Tensor:
        is_pad = (hist_items == self.pad).unsqueeze(-1)
        x = self.hist_emb(hist_items) + self.rew_proj(hist_rewards.unsqueeze(-1)).masked_fill(is_pad, 0.0)
        _, h = self.gru(x)
        return self.user_mlp(h[-1])  # [B, 2d]

    def _heads(self, u: torch.Tensor, iv: torch.Tensor, ib: torch.Tensor):
        d = self.emb_dim
        wr = (u[..., :d] * iv[..., :d]).sum(-1) + ib[..., 0] + self.glob[0]
        eng = (u[..., d:] * iv[..., d:]).sum(-1) + ib[..., 1] + self.glob[1]
        return wr, eng

    def forward(self, hist_items, hist_rewards, target_items):
        u = self.user_vec(hist_items, hist_rewards)
        return self._heads(u, self.item_vec(target_items), self.item_bias(target_items))

    @torch.no_grad()
    def score_all(self, hist_items, hist_rewards, n_catalog: int):
        """[B, n_catalog] predicted capped watch ratio and engagement probability for catalog items."""
        u = self.user_vec(hist_items, hist_rewards)
        d = self.emb_dim
        iv, ib = self.item_vec.weight[:n_catalog], self.item_bias.weight[:n_catalog]
        wr = u[:, :d] @ iv[:, :d].T + ib[:, 0] + self.glob[0]
        eng = u[:, d:] @ iv[:, d:].T + ib[:, 1] + self.glob[1]
        return wr, torch.sigmoid(eng)
