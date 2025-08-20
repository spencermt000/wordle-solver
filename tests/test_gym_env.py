import numpy as np
from scripts.data_utils import load_answer_vocab
from scripts.sampler import WordSampler
from scripts.gym_env import GymWordleEnv


def test_gym_wordle_env_behavior():
    vocab = load_answer_vocab("word_list.csv")
    sampler = WordSampler(vocab, seed=42)
    env = GymWordleEnv(vocab, sampler, allow_probe_guesses=False)

    # Reset returns (obs, info) where info carries the action mask
    obs, info = env.reset()
    mask = info["action_mask"]

    # Basic checks on reset
    assert obs.shape == (158,)
    assert mask.shape[0] == len(vocab)
    assert np.sum(mask) > 0  # there must be valid actions

    # Take one valid action (first allowed index)
    valid = np.nonzero(mask)[0]
    assert valid.size > 0
    action = int(valid[0])

    obs2, reward, terminated, truncated, info2 = env.step(action)

    # Types and shapes
    assert obs2.shape == (158,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

    # Candidate set should not grow after a step (mask has same length but <= number of ones)
    new_mask = info2["action_mask"]
    assert new_mask.shape[0] == mask.shape[0]
    assert np.sum(new_mask) <= np.sum(mask)
