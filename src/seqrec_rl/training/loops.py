"""Offline and online training loops with checkpoint/resume.

Checkpoint = agent modules (networks, target net, optimizer) + replay buffers + RNG states + loop counters.
Online fine-tuning inherits from an offline checkpoint according to ``Inherit``:
  weights (always), target network (copy of the inherited target, or re-synced to the online net), optimizer state
  (keep or reset), offline replay (kept and mixed at ``offline_ratio``, or dropped).
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ..agents.base import hard_update
from ..evaluation.rollout import collect


@dataclass
class Inherit:
    keep_target: bool = True
    keep_optimizer: bool = False
    offline_ratio: float = 0.5  # fraction of each batch drawn from the offline buffer (0 = online only)


def _rng_state(rng: np.random.Generator) -> dict:
    return {"np_gen": rng.bit_generator.state, "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(rng: np.random.Generator, st: dict) -> None:
    rng.bit_generator.state = st["np_gen"]
    torch.set_rng_state(st["torch"])
    if st["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def save_ckpt(path: Path, agent, rng, counters: dict, buffers: dict | None = None, extra: dict | None = None):
    """Crash-safe: DONE is removed first and written last, every file goes through a tmp + rename, so a
    preempted save leaves either the previous complete checkpoint invalidated (no DONE) or a complete new one."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "DONE").unlink(missing_ok=True)
    torch.save(agent.state_dict(), path / "agent.pt.tmp")
    with open(path / "state.pkl.tmp", "wb") as f:
        pickle.dump({"rng": _rng_state(rng), "counters": counters, "extra": extra or {}}, f)
    for name, buf in (buffers or {}).items():
        buf.save(path / f"buffer_{name}.tmp.npz")
    (path / "agent.pt.tmp").replace(path / "agent.pt")
    (path / "state.pkl.tmp").replace(path / "state.pkl")
    for name in (buffers or {}):
        (path / f"buffer_{name}.tmp.npz").replace(path / f"buffer_{name}.npz")
    (path / "DONE").write_text(json.dumps(counters))


def load_ckpt(path: Path, agent, rng) -> dict | None:
    if not (path / "DONE").exists():
        return None
    agent.load(path / "agent.pt")
    with open(path / "state.pkl", "rb") as f:
        st = pickle.load(f)
    _set_rng_state(rng, st["rng"])
    return st


def train_offline(agent, buffer, n_steps: int, batch_size: int, rng: np.random.Generator, run=None,
                  eval_fn=None, eval_every: int = 5000, ckpt_every: int = 5000, log_every: int = 500) -> list:
    """Gradient steps on a fixed buffer. Resumes from run.dir/ckpt if present."""
    history, start = [], 0
    ckpt = run.dir / "ckpt" if run is not None else None
    if ckpt is not None:
        st = load_ckpt(ckpt, agent, rng)
        if st is not None:
            start = st["counters"]["step"]
            history = st["extra"].get("history", [])
    agent.train_mode(True)
    acc: dict = {}
    for step in range(start + 1, n_steps + 1):
        b = buffer.batch(buffer.sample_idx(batch_size, rng), agent.device)
        info = agent.update(b)
        for k, v in info.items():
            acc[k] = acc.get(k, 0.0) + v
        if run is not None and step % log_every == 0:
            run.log(phase="offline", step=step, **{k: v / log_every for k, v in acc.items()})
            acc = {}
        if eval_fn is not None and (step % eval_every == 0 or step == n_steps):
            ev = eval_fn(agent)
            ev["step"] = step
            ev.pop("returns", None)
            history.append(ev)
            if run is not None:
                run.log(phase="offline_eval", **{k: v for k, v in ev.items() if np.isscalar(v)})
                # validation-based checkpoint selection (same rule and budget for every method)
                if ev["sim_return"] >= max(h["sim_return"] for h in history):
                    agent.save(run.dir / "best_agent.pt")
        if ckpt is not None and (step % ckpt_every == 0 or step == n_steps):
            save_ckpt(ckpt, agent, rng, {"step": step}, extra={"history": history})
    return history


def train_online(agent, env, online_buf, offline_buf, budget: int, rng: np.random.Generator, inherit: Inherit,
                 batch_size: int = 256, updates_per_collect: int = 4, collect_per_iter: int | None = None,
                 warmup: int = 0, run=None, eval_fn=None, eval_every: int = 2000, ckpt_every: int = 10000,
                 alpha_schedule=None, on_step=None, gamma: float = 0.95) -> list:
    """Interact for ``budget`` transitions with exploration; after each collection of ``collect_per_iter``
    transitions (default env.n_envs) do ``updates_per_collect`` gradient steps. Evaluation (greedy, no updates)
    every ``eval_every`` transitions, including at 0 to measure the starting point / initial drop."""
    collect_per_iter = collect_per_iter or env.n_envs
    history, used = [], 0
    ckpt = run.dir / "ckpt" if run is not None else None
    st = load_ckpt(ckpt, agent, rng) if ckpt is not None else None
    if st is not None:
        used = st["counters"]["used"]
        agent.env_steps = st["counters"]["env_steps"]
        history = st["extra"]["history"]
        from ..replay.buffer import ReplayBuffer
        online_buf.__dict__.update(ReplayBuffer.load(ckpt / "buffer_online.npz", online_buf.spec,
                                                     online_buf.capacity).__dict__)
        if on_step is not None and len(online_buf):
            on_step(online_buf.data["action"][:online_buf.size])
    else:
        if not inherit.keep_target and hasattr(agent, "q_target"):
            hard_update(agent.q_target, agent.q)
        if not inherit.keep_optimizer and hasattr(agent, "reset_optimizer"):
            agent.reset_optimizer()
        if eval_fn is not None:
            ev = eval_fn(agent)
            ev["used"] = 0
            history.append(ev)
            if run is not None:
                run.log(phase="online_eval", **{k: v for k, v in ev.items() if np.isscalar(v)})
    act = lambda obs, r: (agent.act(obs, r, explore=True), None)  # noqa: E731
    next_eval = (used // eval_every + 1) * eval_every
    next_ckpt = (used // ckpt_every + 1) * ckpt_every
    acc, n_acc, coll = {}, 0, {}
    while used < budget:
        coll = collect(env, act, collect_per_iter, rng, online_buf, source=1, gamma=gamma, on_step=on_step,
                       on_transition=getattr(agent, "observe", None))
        used += coll["n"]
        agent.env_steps = used
        if alpha_schedule is not None and hasattr(agent, "alpha_scale"):
            agent.alpha_scale = alpha_schedule(used)
        if used >= warmup and len(online_buf) >= batch_size:
            for _ in range(updates_per_collect):
                n_off = int(round(batch_size * inherit.offline_ratio)) if offline_buf is not None else 0
                parts = []
                if n_off:
                    parts.append(offline_buf.batch(offline_buf.sample_idx(n_off, rng), agent.device))
                parts.append(online_buf.batch(online_buf.sample_idx(batch_size - n_off, rng), agent.device))
                b = parts[0] if len(parts) == 1 else {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
                info = agent.update(b)
                for k, v in info.items():
                    acc[k] = acc.get(k, 0.0) + v
                n_acc += 1
        if used >= next_eval or used >= budget:
            next_eval += eval_every
            if run is not None and n_acc:
                run.log(phase="online", used=used, eps=getattr(agent, "epsilon", lambda: 0)(),
                        train_return=coll.get("true_return"), **{k: v / n_acc for k, v in acc.items()})
            acc, n_acc = {}, 0
            if eval_fn is not None:
                ev = eval_fn(agent)
                ev["used"] = used
                history.append(ev)
                if run is not None:
                    run.log(phase="online_eval", **{k: v for k, v in ev.items() if np.isscalar(v)})
        if ckpt is not None and (used >= next_ckpt or used >= budget):
            next_ckpt += ckpt_every
            save_ckpt(ckpt, agent, rng, {"used": used, "env_steps": agent.env_steps},
                      {"online": online_buf}, extra={"history": history})
    return history


def inherit_dict(i: Inherit) -> dict:
    return asdict(i)
