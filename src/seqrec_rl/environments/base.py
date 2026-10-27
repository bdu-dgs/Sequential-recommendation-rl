"""Common batched observation format shared by all environments and agents.

An observation is a dict of numpy arrays with a leading batch dimension B:

- ``hist_items``  int64   [B, K]  last K items shown to the user (most recent last), pad = ``hist_pad``.
                                  History items live in a vocabulary of ``n_hist_items`` >= ``n_items``; the first
                                  ``n_items`` indices are the actions, so a recommended item keeps its index.
- ``hist_rewards`` float32 [B, K] the feedback the user gave to those items (0 for padding)
- ``user_feat``   float32 [B, F]  static user features observable before the session (F may be 0)
- ``step``        int64   [B]     index of the current decision inside the episode (0-based)
- ``ep_items``    int64   [B, H]  items already recommended in this episode (pad = n_items); used for the
                                  no-repeat action mask

Everything in an observation is known *before* the current recommendation is made. Feedback for an item only
enters ``hist_rewards`` after that item was recommended.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

OBS_KEYS = ("hist_items", "hist_rewards", "user_feat", "step", "ep_items")


@dataclass
class EnvSpec:
    n_items: int  # number of actions
    hist_len: int  # K
    horizon: int  # H, maximum episode length
    user_feat_dim: int
    gamma: float = 0.9
    n_hist_items: int | None = None  # history vocabulary size (defaults to n_items)
    # [n_hist_items, T] tag multi-hot for every history item. When set, encoders represent history items by tags,
    # with ID embeddings only for action (catalog) items, so rare non-catalog warm-start items cannot act as
    # user identifiers.
    item_tags: object = None

    def __post_init__(self):
        if self.n_hist_items is None:
            self.n_hist_items = self.n_items

    @property
    def pad(self) -> int:
        """Padding index of ``ep_items``."""
        return self.n_items

    @property
    def hist_pad(self) -> int:
        """Padding index of ``hist_items``."""
        return self.n_hist_items


def obs_to_torch(obs: dict, device) -> dict:
    return {k: torch.as_tensor(obs[k], device=device) for k in OBS_KEYS}


def action_mask(ep_items: torch.Tensor, n_items: int) -> torch.Tensor:
    """Boolean [B, n_items] mask of allowed actions: items not yet recommended in this episode."""
    B = ep_items.shape[0]
    mask = torch.ones(B, n_items + 1, dtype=torch.bool, device=ep_items.device)
    mask.scatter_(1, ep_items.long(), False)
    return mask[:, :n_items]


def action_mask_np(ep_items: np.ndarray, n_items: int) -> np.ndarray:
    B = ep_items.shape[0]
    mask = np.ones((B, n_items + 1), dtype=bool)
    np.put_along_axis(mask, ep_items.astype(np.int64), False, axis=1)
    return mask[:, :n_items]


def slice_obs(obs: dict, idx) -> dict:
    return {k: v[idx] for k, v in obs.items()}


def stack_obs(obs_list: list[dict]) -> dict:
    return {k: np.concatenate([o[k] for o in obs_list], axis=0) for k in OBS_KEYS}
