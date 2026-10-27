"""Regression tests for issues found in the pre-cluster code review."""

import numpy as np
import pytest
import torch

from seqrec_rl import PROCESSED_DIR
from seqrec_rl.agents.support import SupportWeights
from seqrec_rl.environments.toy import ToyRecEnv
from seqrec_rl.evaluation.rollout import collect
from seqrec_rl.replay.buffer import ReplayBuffer


def _random_act(obs, rng):
    a = np.array([rng.choice([i for i in range(8) if i not in obs["ep_items"][b]]) for b in range(len(obs["step"]))])
    return a, None


def test_buffer_append_after_load(tmp_path):
    rng = np.random.default_rng(0)
    env = ToyRecEnv(n_envs=2, seed=0)
    buf = ReplayBuffer(env.spec, 1000)
    collect(env, _random_act, 100, rng, buf, gamma=0.9)
    buf.save(tmp_path / "b.npz")
    for cap in (None, 500):
        b2 = ReplayBuffer.load(tmp_path / "b.npz", env.spec, cap)
        assert b2.ptr == buf.size % b2.capacity
        n0 = b2.size
        if b2.capacity > n0:
            collect(ToyRecEnv(n_envs=2, seed=1), _random_act, 10, rng, b2, gamma=0.9)
            assert b2.size == n0 + 10
        np.testing.assert_array_equal(b2.data["action"][:n0], buf.data["action"][:n0])


def _obs(B, A, H=4):
    return {"ep_items": torch.full((B, H), A), "step": torch.zeros(B, dtype=torch.long)}


def test_shuffled_support_uses_a_fixed_permutation():
    A = 10
    p = torch.softmax(torch.arange(A, dtype=torch.float32).repeat(3, 1), 1)
    sw = SupportWeights(A, "shuffled", prob_fn=lambda o: p, seed=0)
    c1, c2 = sw(_obs(3, A)), sw(_obs(3, A))
    assert torch.equal(c1, c2)
    oracle = SupportWeights(A, "oracle", prob_fn=lambda o: p)(_obs(3, A))
    assert torch.allclose(c1.sort(1).values, oracle.sort(1).values)
    assert not torch.allclose(c1, oracle)


def test_uniform_tracked_matches_oracle_strength_each_update():
    from seqrec_rl.agents.q_agents import QAgent, QConfig
    env = ToyRecEnv(n_envs=4, seed=0)
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(env.spec, 2000)
    collect(env, _random_act, 400, rng, buf, gamma=0.9)
    probs = lambda o: torch.full((o["ep_items"].shape[0], 8), 1 / 8)  # noqa: E731
    sw = SupportWeights(8, "uniform_tracked", const=0.123)
    sw.reference = SupportWeights(8, "oracle", prob_fn=probs, kappa=1.0)
    agent = QAgent(env.spec, QConfig(cql_alpha=1.0, emb_dim=8, hidden=16), support_fn=sw)
    agent.update(buf.batch(np.arange(64), "cpu"))
    # oracle weight with uniform p over 8 items, nothing masked at step>0 rows varies; const must be in (0, 1)
    assert sw.const != 0.123 and 0 < sw.const <= 1


@pytest.mark.data
@pytest.mark.skipif(not (PROCESSED_DIR / "fo_watch_ratio.npy").exists(), reason="processed data missing")
def test_eval_override_uses_fo_matrix(tmp_path):
    from seqrec_rl.training.runs import load_config
    from seqrec_rl.training.setup import Setup
    pred = tmp_path / "pred.npy"
    S0 = Setup(load_config("configs/experiment/base.yaml"), 0)
    np.save(pred, np.full_like(S0.data.W, 0.7))
    cfg = load_config("configs/experiment/base.yaml",
                      ["sim.response=learned", f"sim.learned_response_path={pred.as_posix()}", "eval.response=fo"])
    S = Setup(cfg, 0)
    train_env = S.env("train", 2)
    assert np.allclose(train_env.W, 0.7)
    from seqrec_rl.environments.kuairec_sim import KuaiRecSim
    f = S.eval_fn("val", n_users=2)
    eval_env = [c.cell_contents for c in f.__closure__ if isinstance(c.cell_contents, KuaiRecSim)][0]
    assert np.array_equal(eval_env.W, S.data.W)  # eval.response=fo is honoured by eval_fn
    assert S.dataset_name() == S0.dataset_name()  # logs are always generated in the FO simulator
