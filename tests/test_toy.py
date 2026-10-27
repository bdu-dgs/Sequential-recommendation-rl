import numpy as np
import pytest
import torch

from seqrec_rl.agents.q_agents import QAgent, QConfig
from seqrec_rl.environments.toy import ToyRecEnv, greedy_value, optimal_value
from seqrec_rl.evaluation.rollout import collect, evaluate
from seqrec_rl.replay.buffer import ReplayBuffer
from seqrec_rl.training.loops import Inherit, train_offline, train_online


def test_dp_values():
    assert optimal_value(8, 1.0) == pytest.approx(5.2)  # AABB interleaving collects every item
    assert optimal_value(8, 0.9) == pytest.approx(3.9322, abs=1e-4)
    assert greedy_value(8, 0.9) == pytest.approx(1.81)


def test_quit_rule_and_horizon():
    env = ToyRecEnv(n_envs=1, seed=0)
    env.reset()
    env.types[:] = 0
    _, r, d, _ = env.step(np.array([0]))
    _, r, d, _ = env.step(np.array([1]))
    assert not d[0]
    _, r, d, info = env.step(np.array([2]))  # third A in a row -> user quits, reward 0
    assert d[0] and info["quit"][0] and r[0] == 0.0


def discounted_eval(agent, gamma=0.9, n=400, seed=1):
    env = ToyRecEnv(n_envs=n, gamma=gamma, seed=seed)
    obs = env.reset()
    ret, disc = np.zeros(n), np.ones(n)
    rng = np.random.default_rng(0)
    while not env.done.all():
        a = agent.act(obs, rng)
        obs, r, d, info = env.step(a)
        act = info["active"]
        ret[act] += disc[act] * r[act]
        disc[act] *= gamma
    return ret.mean()


def test_dqn_learns_toy_optimum():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    env = ToyRecEnv(n_envs=16, gamma=0.9, seed=0)
    agent = QAgent(env.spec, QConfig(gamma=0.9, lr=1e-3, emb_dim=16, hidden=64, eps_start=1.0, eps_end=0.05,
                                     eps_decay_steps=8000, tau=0.01))
    buf = ReplayBuffer(env.spec, 50000)
    train_online(agent, env, buf, None, budget=20000, rng=rng, inherit=Inherit(offline_ratio=0.0),
                 batch_size=128, updates_per_collect=2, gamma=0.9)
    v = discounted_eval(agent)
    assert v > 0.9 * optimal_value(8, 0.9), v  # far above greedy (1.81)


def test_cql_on_narrow_logs_stays_in_support():
    """Logs from a policy that only plays category-A items: CQL should not choose unlogged B items greedily
    at the first step (they have no support), whereas the TD-only learner has no such guarantee."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    env = ToyRecEnv(n_envs=16, gamma=0.9, seed=0)
    buf = ReplayBuffer(env.spec, 20000)

    def behave(obs, r):
        a = np.zeros(len(obs["step"]), dtype=np.int64)
        for b in range(len(a)):
            allowed = [i for i in range(4) if i not in obs["ep_items"][b]] or \
                      [i for i in range(8) if i not in obs["ep_items"][b]]
            a[b] = r.choice(allowed)
        return a, None

    collect(env, behave, 5000, rng, buf, source=0, gamma=0.9)
    agent = QAgent(env.spec, QConfig(gamma=0.9, lr=1e-3, emb_dim=16, hidden=64, cql_alpha=1.0))
    train_offline(agent, buf, 3000, 128, rng)
    obs = ToyRecEnv(n_envs=4, seed=3).reset()
    a = agent.act(obs, rng)
    assert (a < 4).all()
