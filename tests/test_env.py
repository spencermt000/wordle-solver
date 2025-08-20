import numpy as np
import pytest

from scripts.data_utils import load_answer_vocab
from scripts.sampler import WordSampler
from scripts.env import WordleEnv


@pytest.fixture
def env():
    vocab = load_answer_vocab("word_list.csv")
    sampler = WordSampler(vocab, seed=0)
    return WordleEnv(vocab, sampler)


def test_reset_returns_valid_mask(env):
    obs, info = env.reset()
    assert "action_mask" in info
    assert np.sum(info["action_mask"]) == len(env.vocab)


def test_step_reduces_candidates(env):
    _, info = env.reset()
    valid = np.nonzero(info["action_mask"])[0]
    guess_idx = int(valid[0])

    obs, reward, terminated, truncated, info = env.step(guess_idx)

    assert reward <= 0  # usually negative unless solved
    assert not terminated
    assert np.sum(info["action_mask"]) < len(env.vocab)


def test_solve_word_gives_positive_reward(env):
    # Force target to a known word
    target_word = env.vocab[0]
    env.target = target_word
    target_idx = env.vocab.index(target_word)

    obs, reward, terminated, truncated, info = env.step(target_idx)

    assert terminated
    assert reward > 0