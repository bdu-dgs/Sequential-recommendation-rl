import numpy as np
import torch

from seqrec_rl.agents.q_agents import QAgent, QConfig
from seqrec_rl.environments.toy import ToyRecEnv
from seqrec_rl.evaluation.rollout import collect, evaluate
from seqrec_rl.replay.buffer import ReplayBuffer, next_obs_torch


def random_act(obs, rng):
    a = np.zeros(len(obs["step"]), dtype=np.int64)
    for b in range(len(a)):
        a[b] = rng.choice([i for i in range(8) if i not in obs["ep_items"][b]])
    return a, None


def test_next_obs_reconstruction_matches_env():
    rng = np.random.default_rng(0)
    env = ToyRecEnv(n_envs=4, seed=0)
    env.reset()
    buf = ReplayBuffer(env.spec, 1000)
    collect(env, random_act, 300, rng, buf, gamma=0.9)
    b = buf.batch(np.arange(buf.size - 1), "cpu")
    nxt = next_obs_torch(b, env.spec)
    # consecutive rows from the same slot are interleaved; compare each non-terminal row to the next row
    # of the same episode
    ep = buf.data["episode"][:buf.size]
    for i in range(buf.size - 1):
        if buf.data["done"][i]:
            continue
        nxt_rows = np.flatnonzero(ep[i + 1:] == ep[i])
        if len(nxt_rows) == 0:  # episode still open when collection stopped
            continue
        j = i + 1 + nxt_rows[0]
        for k in ("hist_items", "hist_rewards", "step", "ep_items"):
            assert np.allclose(nxt[k][i].numpy(), buf.data[k][j]), (k, i, j)


def test_return_to_go_and_terminal_target():
    rng = np.random.default_rng(0)
    env = ToyRecEnv(n_envs=2, seed=0)
    buf = ReplayBuffer(env.spec, 1000)
    collect(env, random_act, 100, rng, buf, gamma=0.9)
    d = buf.data
    for e in np.unique(d["episode"][:buf.size]):
        idx = np.flatnonzero(d["episode"][:buf.size] == e)
        if not d["done"][idx[-1]]:
            continue
        g = 0.0
        for i in idx[::-1]:
            g = d["reward"][i] + 0.9 * g
            assert np.isclose(d["rtg"][i], g)
    # terminal transitions: the TD target equals the reward (no bootstrap)
    agent = QAgent(env.spec, QConfig(gamma=0.9, emb_dim=8, hidden=16))
    term = np.flatnonzero(d["done"][:buf.size])
    b = buf.batch(term, "cpu")
    nobs = next_obs_torch(b, env.spec)
    with torch.no_grad():
        nq = agent.q_target(nobs).max(1).values
        y = b["reward"] + 0.9 * (1 - b["done"].float()) * nq
    assert torch.allclose(y, b["reward"])


def test_evaluation_does_not_update_model():
    env = ToyRecEnv(n_envs=8, seed=0)
    agent = QAgent(env.spec, QConfig(emb_dim=8, hidden=16))
    before = {k: v.clone() for k, v in agent.q.state_dict().items()}
    evaluate(env, agent, np.arange(16), np.random.default_rng(0))
    after = agent.q.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
