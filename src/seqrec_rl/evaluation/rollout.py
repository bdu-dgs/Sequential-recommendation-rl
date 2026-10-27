"""Data collection and policy evaluation. Evaluation never calls Agent.update() (see tests/test_eval_no_update.py).

All returns here are SIMULATED returns of the environment that produced them.
"""

from __future__ import annotations

import numpy as np
import torch

from ..environments.base import obs_to_torch


def collect(env, act_fn, n_transitions: int, rng: np.random.Generator, buffer=None, source: int = 1,
            gamma: float = 0.95, on_step=None, on_transition=None) -> dict:
    """Run ``act_fn(obs, rng) -> (actions, props|None)`` on an auto-resetting vector env until
    ``n_transitions`` transitions were added. Returns episode statistics of episodes that finished."""
    if not hasattr(env, "_collect_started"):
        env.reset()
        env._collect_started = True
    added, ep_ret, ep_true, ep_len = 0, [], [], []
    run_ret = getattr(env, "_run_ret", np.zeros(env.n_envs))
    run_true = getattr(env, "_run_true", np.zeros(env.n_envs))
    while added < n_transitions:
        obs = env.obs()
        actions, props = act_fn(obs, rng)
        nobs, r, done, info = env.step(actions)
        active = info["active"]
        if buffer is not None:
            buffer.add_batch(obs, actions, r, info["true_reward"], done, props, source, active,
                             info.get("user_row", np.zeros(env.n_envs, dtype=np.int64)), gamma)
        if on_step is not None:
            on_step(actions[active])
        if on_transition is not None and len(active):
            on_transition({k: v[active] for k, v in obs.items()}, actions[active], r[active])
        run_ret[active] += r[active]
        run_true[active] += info["true_reward"][active]
        added += len(active)
        fin = active[done[active]]
        for j in fin:
            ep_ret.append(run_ret[j])
            ep_true.append(run_true[j])
            ep_len.append(int(env.t[j]))
        run_ret[fin] = 0.0
        run_true[fin] = 0.0
        env.reset_slots(fin)
    env._run_ret, env._run_true = run_ret, run_true
    return {"n": added, "episodes": len(ep_ret), "return": float(np.mean(ep_ret)) if ep_ret else np.nan,
            "true_return": float(np.mean(ep_true)) if ep_true else np.nan,
            "length": float(np.mean(ep_len)) if ep_len else np.nan}


def agent_act_fn(agent, explore: bool):
    def f(obs, rng):
        return agent.act(obs, rng, explore=explore), None
    return f


@torch.no_grad()
def evaluate(env, agent, user_rows: np.ndarray, rng: np.random.Generator, item_tail: np.ndarray | None = None,
             support_prob_fn=None, keep_trajectories: int = 0) -> dict:
    """Greedy rollout, one episode per user in ``user_rows`` (in chunks of env.n_envs)."""
    agent.train_mode(False)
    if hasattr(env, "rng"):  # identical noise / sampling stream for every evaluation call
        env.rng = np.random.default_rng(20201024)
    rets, trues, lens, left, rec_items, tag_div, tail_share, sup = [], [], [], [], [], [], [], []
    trajs, q_rows = [], []
    tags = getattr(env, "tags", None)
    for s in range(0, len(user_rows), env.n_envs):
        chunk = user_rows[s:s + env.n_envs]
        n = len(chunk)
        pad_rows = np.r_[chunk, np.repeat(chunk[-1:], env.n_envs - n)]
        if hasattr(env, "data"):
            env.reset_slots(np.arange(env.n_envs), pad_rows)
        else:
            env.reset()
        ret = np.zeros(env.n_envs)
        tru = np.zeros(env.n_envs)
        user_left = np.zeros(env.n_envs, dtype=bool)
        steps = [[] for _ in range(env.n_envs)]
        while not env.done[:n].all():
            obs = env.obs()
            a = agent.act(obs, rng, explore=False)
            if support_prob_fn is not None:
                p = support_prob_fn(obs)  # numpy obs -> [B, A] tensor on the support model's device
                pa = p.gather(1, torch.as_tensor(a, device=p.device)[:, None]).squeeze(1).cpu().numpy()
            else:
                pa = None
            qa = None
            if hasattr(agent, "q"):  # predicted Q of the chosen action (for the over-estimation diagnostic)
                ot = obs_to_torch(obs, agent.device)
                qa = agent.q(ot).gather(1, torch.as_tensor(a, device=agent.device)[:, None]).squeeze(1).cpu().numpy()
            _, r, done, info = env.step(a)
            act = info["active"]
            for j in act[act < n]:
                steps[j].append((int(a[j]), float(info["true_reward"][j]), float(r[j]),
                                 None if pa is None else float(pa[j]), None if qa is None else float(qa[j])))
            ret[act] += r[act]
            tru[act] += info["true_reward"][act]
            user_left |= info.get("left", info.get("quit", np.zeros(env.n_envs, dtype=bool)))
        for j in range(n):
            items = [x[0] for x in steps[j]]
            rets.append(ret[j])
            trues.append(tru[j])
            lens.append(len(items))
            left.append(bool(user_left[j]))
            rec_items.extend(items)
            if tags is not None and items:
                tag_div.append(len(set(np.flatnonzero(tags[items].any(0)))) / tags.shape[1])
            if item_tail is not None and items:
                tail_share.append(float(np.mean(item_tail[items])))
            if pa is not None:
                sup.extend([x[3] for x in steps[j]])
            if steps[j] and steps[j][0][4] is not None and pa is not None:
                # true value of (s_t, a_t) under the greedy policy = discounted return actually collected
                # (dynamics are deterministic when noise_std = 0 and the policy is greedy)
                g, mc = 0.0, []
                for x in reversed(steps[j]):
                    g = x[1] + env.spec.gamma * g
                    mc.append(g)
                q_rows.extend((x[4], m, x[3]) for x, m in zip(steps[j], reversed(mc)))
            if len(trajs) < keep_trajectories:
                trajs.append({"user_row": int(chunk[j]), "steps": steps[j]})
    rec = np.array(rec_items)
    out = {
        "n_users": len(rets),
        "sim_return": float(np.mean(trues)),
        "sim_return_std": float(np.std(trues)),
        "observed_return": float(np.mean(rets)),
        "length": float(np.mean(lens)),
        "leave_rate": float(np.mean(left)),
        "per_step_reward": float(np.sum(trues) / max(1, np.sum(lens))),
        "coverage": float(len(np.unique(rec)) / env.spec.n_items) if len(rec) else 0.0,
        "tag_coverage_per_episode": float(np.mean(tag_div)) if tag_div else np.nan,
        "long_tail_share": float(np.mean(tail_share)) if tail_share else np.nan,
        "returns": [float(x) for x in trues],
    }
    if sup:
        sup = np.array(sup)
        out["mean_logging_prob_of_chosen"] = float(sup.mean())
        out["frac_chosen_below_1e-4"] = float((sup < 1e-4).mean())
    if q_rows:
        q, mc, p = (np.array(c) for c in zip(*q_rows))
        diag = {"all": {"n": int(len(q)), "q": float(q.mean()), "true": float(mc.mean()),
                        "overestimation": float((q - mc).mean())}}
        for name, lo, hi in (("p<1e-4", 0, 1e-4), ("1e-4<=p<1e-2", 1e-4, 1e-2), ("p>=1e-2", 1e-2, 2)):
            m = (p >= lo) & (p < hi)
            diag[name] = {"share": float(m.mean()), "n": int(m.sum()),
                          "q": float(q[m].mean()) if m.any() else np.nan,
                          "true": float(mc[m].mean()) if m.any() else np.nan,
                          "overestimation": float((q[m] - mc[m]).mean()) if m.any() else np.nan}
        out["q_diag"] = diag
    if keep_trajectories:
        out["trajectories"] = trajs
    agent.train_mode(True)
    return out
