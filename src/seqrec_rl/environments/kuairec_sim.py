"""KuaiRec-FO simulated session environment (``KuaiRecFO-v1``). All returns from it are SIMULATED returns.

Provenance of each component
----------------------------
REAL (from KuaiRec logs)
  - users: the 1,411 users of the fully-observed (FO) small matrix
  - actions: the 3,327 FO items
  - base response W[u, a] = min(watch_ratio, cap) of user u on item a, read from the FO matrix (no model error).
    Missing (blocked) pairs have response 0.
  - warm-start history: the user's last K real big-matrix interactions (items + capped watch ratio) before the end
    of the train window (2020-08-10 local). It contains no FO pairs (big and small matrices share no pairs).
SYNTHETIC (simulation rules, NOT identified by the data — see docs/environment_assumptions.md)
  - fatigue: response is scaled by (1 - beta * f), where f is the fraction of the last `fatigue_window` in-episode
    recommendations that share at least one tag with the item
  - patience/leave: a recommendation with effective response < skip_threshold costs one unit of patience; the
    user leaves when patience reaches 0
  - preference drift: W is multiplied per (user, tag) by exp(delta * z), z ~ N(0, 1) (fixed per drift_seed)
  - feedback noise: the *observed* reward is effective response + N(0, noise_std^2), clipped to [0, cap];
    the true effective response is reported in info["true_reward"]
LEARNED (only when response="learned")
  - W is replaced by the supervised response model's prediction from the warm-start history (used to measure
    how much policies exploit simulator error)

Episode: at most `horizon` steps, no repeated items (action mask), ends early when patience is exhausted. The step
index is part of the observation, so the horizon is a true terminal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .. import PROCESSED_DIR
from .base import EnvSpec


@dataclass
class SimConfig:
    horizon: int = 30
    hist_len: int = 20
    reward_cap: float = 2.0
    gamma: float = 0.95
    fatigue_beta: float = 0.3
    fatigue_window: int = 5
    patience: int = 3
    skip_threshold: float = 0.5
    noise_std: float = 0.0
    drift_delta: float = 0.0
    drift_seed: int = 0
    cold_start_hist: int | None = None  # keep only this many warm-start history items (None = all K)
    response: str = "fo"  # "fo" or "learned"
    learned_response_path: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SimData:
    W: np.ndarray  # [U, A] float32 capped FO watch ratio (NaN -> 0)
    tags: np.ndarray  # [A, T] bool
    warm_items: np.ndarray  # [U, K] int64 (pad = n_hist_items)
    warm_rewards: np.ndarray  # [U, K] float32
    fo_users: np.ndarray  # [U] original user ids in FO row order
    n_hist_items: int
    env_split_rows: dict = field(default_factory=dict)  # split -> FO row indices
    tags_all: np.ndarray | None = None  # [n_hist_items, T] tags of every history-vocabulary item

    @property
    def n_items(self) -> int:
        return self.W.shape[1]


def build_sim_data(hist_len: int = 20, cap: float = 2.0, processed_dir: Path = PROCESSED_DIR,
                   cache: bool = True) -> SimData:
    cache_path = processed_dir / f"sim_data_K{hist_len}_cap{cap}.npz"
    from ..data.prepare import load_splits
    splits = load_splits(processed_dir)
    fo_users = np.array(splits["small_users_in_fo_order"])
    if cache and cache_path.exists():
        z = np.load(cache_path)
        W, tags, wi, wr = z["W"], z["tags"], z["warm_items"], z["warm_rewards"]
    else:
        from ..data.sequences import SequenceExamples, load_interactions
        raw = np.load(processed_dir / "fo_watch_ratio.npy")
        W = np.nan_to_num(np.minimum(raw, cap), nan=0.0).astype(np.float32)
        n_cat = splits["n_catalog"]
        tags = np.load(processed_dir / "item_tags.npy")[:n_cat] > 0
        inter = load_interactions(processed_dir)
        inter = inter[inter["user_group"] == "small"]
        ex = SequenceExamples(inter, hist_len, pad=splits["n_items_all"], wr_cap=cap)
        # end of train window: 2020-08-10 23:59:59 local (UTC+8)
        h = ex.last_history(fo_users, before_ts=1597075199.0)
        wi, wr = h["hist_items"], h["hist_rewards"]
        if cache:
            np.savez_compressed(cache_path, W=W, tags=tags, warm_items=wi, warm_rewards=wr)
    rows = {k: np.flatnonzero(np.isin(fo_users, v)) for k, v in splits["env_users"].items()}
    tags_all = np.load(processed_dir / "item_tags.npy") > 0
    return SimData(W=W, tags=tags, warm_items=wi, warm_rewards=wr, fo_users=fo_users,
                   n_hist_items=splits["n_items_all"], env_split_rows=rows, tags_all=tags_all)


class KuaiRecSim:
    def __init__(self, data: SimData, cfg: SimConfig, user_rows: np.ndarray, n_envs: int, seed: int = 0):
        self.data, self.cfg = data, cfg
        self.user_rows = np.asarray(user_rows)
        self.n_envs = n_envs
        self.rng = np.random.default_rng(seed)
        self.spec = EnvSpec(n_items=data.n_items, hist_len=cfg.hist_len, horizon=cfg.horizon, user_feat_dim=0,
                            gamma=cfg.gamma, n_hist_items=data.n_hist_items, item_tags=data.tags_all)
        if cfg.hist_len != data.warm_items.shape[1]:
            raise ValueError("cfg.hist_len must match the SimData history length")
        if cfg.cold_start_hist is not None and not 0 <= cfg.cold_start_hist <= cfg.hist_len:
            raise ValueError("cold_start_hist must be in [0, hist_len]")
        W = data.W
        if cfg.response == "learned":
            W = np.clip(np.load(cfg.learned_response_path), 0.0, cfg.reward_cap).astype(np.float32)
        if cfg.drift_delta > 0:
            drng = np.random.default_rng(cfg.drift_seed)
            z = drng.standard_normal((W.shape[0], data.tags.shape[1])).astype(np.float32)
            ntag = np.maximum(data.tags.sum(1), 1)
            item_z = (z @ data.tags.T.astype(np.float32)) / ntag  # [U, A] mean z over the item's tags
            W = np.minimum(W * np.exp(cfg.drift_delta * item_z), cfg.reward_cap).astype(np.float32)
        self.W = W
        self.tags = data.tags
        self._alloc()

    # ------------------------------------------------------------------------------------------------
    def _alloc(self):
        B, K, H = self.n_envs, self.cfg.hist_len, self.cfg.horizon
        self.u = np.zeros(B, dtype=np.int64)
        self.t = np.zeros(B, dtype=np.int64)
        self.patience = np.zeros(B, dtype=np.int64)
        self.hist_items = np.full((B, K), self.spec.hist_pad, dtype=np.int64)
        self.hist_rewards = np.zeros((B, K), dtype=np.float32)
        self.ep_items = np.full((B, H), self.spec.pad, dtype=np.int64)
        self.done = np.ones(B, dtype=bool)

    def reset(self, user_rows: np.ndarray | None = None) -> dict:
        self.reset_slots(np.arange(self.n_envs), user_rows)
        return self.obs()

    def reset_slots(self, idx: np.ndarray, user_rows: np.ndarray | None = None) -> None:
        idx = np.asarray(idx)
        if len(idx) == 0:
            return
        u = self.rng.choice(self.user_rows, size=len(idx)) if user_rows is None else np.asarray(user_rows)
        self.u[idx] = u
        self.t[idx] = 0
        self.patience[idx] = self.cfg.patience
        hi, hr = self.data.warm_items[u].copy(), self.data.warm_rewards[u].copy()
        if self.cfg.cold_start_hist is not None:
            keep = self.cfg.cold_start_hist
            K = self.cfg.hist_len
            hi[:, :K - keep] = self.spec.hist_pad
            hr[:, :K - keep] = 0.0
        self.hist_items[idx], self.hist_rewards[idx] = hi, hr
        self.ep_items[idx] = self.spec.pad
        self.done[idx] = False

    def obs(self) -> dict:
        B = self.n_envs
        return {
            "hist_items": self.hist_items.copy(),
            "hist_rewards": self.hist_rewards.copy(),
            "user_feat": np.zeros((B, 0), dtype=np.float32),
            "step": self.t.copy(),
            "ep_items": self.ep_items.copy(),
        }

    def fatigue(self, actions: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
        """Fraction of the last `fatigue_window` in-episode recommendations sharing a tag with each action."""
        rows = np.arange(self.n_envs) if rows is None else rows
        w = self.cfg.fatigue_window
        H = self.cfg.horizon
        t = self.t[rows]
        cols = t[:, None] - np.arange(1, w + 1)[None, :]  # [B, w]
        valid = cols >= 0
        prev = np.take_along_axis(self.ep_items[rows], np.clip(cols, 0, H - 1), axis=1)
        prev = np.where(valid, prev, self.spec.pad)
        tags_pad = np.vstack([self.tags, np.zeros((1, self.tags.shape[1]), bool)])
        share = (tags_pad[prev] & self.tags[actions][:, None, :]).any(-1)  # [B, w]
        return share.sum(1) / w

    def response_all(self) -> np.ndarray:
        """[B, A] effective response of every item for every slot at the current step (oracle access;
        used only by reference policies and analysis, never by learning agents)."""
        B, A = self.n_envs, self.spec.n_items
        w = self.cfg.fatigue_window
        cols = self.t[:, None] - np.arange(1, w + 1)[None, :]
        prev = np.where(cols >= 0, np.take_along_axis(self.ep_items, np.clip(cols, 0, self.cfg.horizon - 1), 1),
                        self.spec.pad)
        tags_pad = np.vstack([self.tags, np.zeros((1, self.tags.shape[1]), bool)]).astype(np.float32)
        recent = tags_pad[prev]  # [B, w, T]
        share = (np.einsum("bwt,at->bwa", recent, self.tags.astype(np.float32)) > 0).sum(1) / w  # [B, A]
        return (self.W[self.u] * (1.0 - self.cfg.fatigue_beta * share)).astype(np.float32)

    def step(self, actions: np.ndarray):
        """Advance active slots. Slots that are already finished are left untouched (reward 0, done True);
        the caller resets them with reset_slots() when it wants a new episode there."""
        actions = np.asarray(actions, dtype=np.int64)
        B = self.n_envs
        act = np.flatnonzero(~self.done)
        a = actions[act]
        if (self.ep_items[act] == a[:, None]).any():
            raise ValueError("repeated item within an episode")
        if (a < 0).any() or (a >= self.spec.n_items).any():
            raise ValueError("action out of range")
        eff = np.zeros(B, dtype=np.float32)
        base = self.W[self.u[act], a]
        eff[act] = base * (1.0 - self.cfg.fatigue_beta * self.fatigue(a, act))
        obs_r = eff.copy()
        if self.cfg.noise_std > 0:
            noisy = eff[act] + self.rng.normal(0, self.cfg.noise_std, len(act))
            obs_r[act] = np.clip(noisy, 0.0, self.cfg.reward_cap)
        skip = np.zeros(B, dtype=bool)
        skip[act] = eff[act] < self.cfg.skip_threshold
        self.patience[act] -= skip[act].astype(np.int64)
        self.ep_items[act, self.t[act]] = a
        self.hist_items[act] = np.concatenate([self.hist_items[act, 1:], a[:, None]], axis=1)
        self.hist_rewards[act] = np.concatenate([self.hist_rewards[act, 1:], obs_r[act, None]], axis=1)
        self.t[act] += 1
        left = np.zeros(B, dtype=bool)
        left[act] = self.patience[act] <= 0
        self.done[act] = left[act] | (self.t[act] >= self.cfg.horizon)
        info = {"true_reward": eff, "skip": skip, "left": left, "user_row": self.u.copy(), "active": act}
        return self.obs(), obs_r, self.done.copy(), info
