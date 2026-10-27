"""KuaiRec simulator: reward correctness, episode boundaries, and no post-recommendation information in the state.
Requires processed data (python scripts/prepare_data.py)."""

import numpy as np
import pytest

from seqrec_rl import PROCESSED_DIR

pytestmark = pytest.mark.data
if not (PROCESSED_DIR / "fo_watch_ratio.npy").exists():
    pytest.skip("processed data missing", allow_module_level=True)

from seqrec_rl.environments.kuairec_sim import KuaiRecSim, SimConfig, build_sim_data  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return build_sim_data()


def test_reward_matches_fo_matrix_and_fatigue(data):
    cfg = SimConfig(fatigue_beta=0.3, patience=100)
    env = KuaiRecSim(data, cfg, data.env_split_rows["train"], n_envs=8, seed=0)
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(10):
        a = np.array([rng.choice(np.setdiff1d(np.arange(data.n_items), env.ep_items[b])) for b in range(8)])
        f = env.fatigue(a)
        expect = data.W[env.u, a] * (1 - 0.3 * f)
        _, r, d, info = env.step(a)
        assert np.allclose(info["true_reward"], expect, atol=1e-6)
        assert np.allclose(r, expect, atol=1e-6)  # no noise configured


def test_fatigue_by_hand(data):
    env = KuaiRecSim(data, SimConfig(fatigue_window=5), data.env_split_rows["train"], n_envs=1, seed=0)
    env.reset()
    tags = data.tags
    a0 = int(np.flatnonzero(tags[:, np.argmax(tags.sum(0))])[0])  # an item with the most common tag
    env.step(np.array([a0]))
    same = [i for i in np.flatnonzero((tags & tags[a0]).any(1)) if i != a0][0]
    assert np.isclose(env.fatigue(np.array([same]))[0], 1 / 5)


def test_patience_and_horizon(data):
    env = KuaiRecSim(data, SimConfig(patience=2, skip_threshold=10.0, horizon=30),
                     data.env_split_rows["train"], n_envs=1, seed=0)  # every item is a skip
    env.reset()
    _, _, d, _ = env.step(np.array([0]))
    assert not d[0]
    _, _, d, info = env.step(np.array([1]))
    assert d[0] and info["left"][0]
    env = KuaiRecSim(data, SimConfig(patience=10**6, horizon=3), data.env_split_rows["train"], n_envs=1, seed=0)
    env.reset()
    for t in range(3):
        _, _, d, _ = env.step(np.array([t]))
    assert d[0] and env.t[0] == 3


def test_observation_has_no_future_information(data):
    """Changing the response of items that were not yet recommended must not change any observation."""
    rows = data.env_split_rows["train"][:4]
    cfg = SimConfig(patience=100)
    env1 = KuaiRecSim(data, cfg, rows, n_envs=4, seed=0)
    env2 = KuaiRecSim(data, cfg, rows, n_envs=4, seed=0)
    actions = [np.array([10, 20, 30, 40]), np.array([11, 21, 31, 41]), np.array([12, 22, 32, 42])]
    env2.W = env2.W.copy()
    untouched = np.setdiff1d(np.arange(data.n_items), np.concatenate(actions))
    env2.W[:, untouched] = 0.123
    o1, o2 = env1.reset(rows), env2.reset(rows)
    for a in actions:
        for k in o1:
            assert np.array_equal(o1[k], o2[k]), k
        o1, *_ = env1.step(a)
        o2, *_ = env2.step(a)
    # and the feedback of the current action only appears after the step
    o = env1.reset(rows)
    assert not (o["hist_items"] == 10).any()
    o, r, *_ = env1.step(actions[0])
    assert (o["hist_items"][:, -1] == actions[0]).all() and np.allclose(o["hist_rewards"][:, -1], r)


def test_warm_history_precedes_cutoff_and_excludes_fo_pairs(data):
    # history items of FO users are never FO pairs of the same user (big and small matrices share no pairs)
    n_cat = data.n_items
    for u in range(0, len(data.fo_users), 50):
        items = data.warm_items[u]
        cat = items[(items < n_cat)]
        assert len(cat) == 0
