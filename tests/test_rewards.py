import numpy as np
from scripts.data_utils import load_answer_vocab
from scripts.sampler import WordSampler
from scripts.env import WordleEnv
from scripts.vocab import WordVocab

def _make_env():
    # Use your real answer vocab; fall back to a tiny manual vocab if needed.
    try:
        vocab = load_answer_vocab("word_list.csv")
    except Exception:
        vocab = WordVocab(["allot", "total", "stoal", "bleed"])
    sampler = WordSampler(vocab, seed=0)
    # default rewards: alpha=2, beta=1, step_penalty=1, success_bonus=10
    return WordleEnv(vocab, sampler, allow_probe_guesses=False)

def test_partial_match_reward_is_expected():
    env = _make_env()
    # force a known target if present; else skip this test
    try:
        target_idx = env.vocab.index_of("total")
        allot_idx  = env.vocab.index_of("allot")
    except Exception:
        # If your answer list lacks these, the fallback vocab will have them.
        env = WordleEnv(WordVocab(["allot", "total", "stoal", "bleed"]), WordSampler(WordVocab(["allot", "total", "stoal", "bleed"]), seed=0))
        target_idx = 1  # "total"
        allot_idx  = 0  # "allot"

    env.reset(target_idx=target_idx)
    obs, reward, done, info, mask = env.step(allot_idx)
    # "allot" vs "total" -> pattern [1,1,0,1,1] => yellows=4, greens=0
    assert reward == 3.0, f"expected 3.0, got {reward}"
    assert done is False

def test_exact_solution_reward_and_done():
    env = _make_env()
    # choose any target and guess it immediately
    obs, mask = env.reset()
    tgt_idx = env._target_idx  # internal, but fine for a test
    obs, reward, done, info, mask = env.step(tgt_idx)
    assert done is True
    assert reward == 19.0, f"expected 19.0 for immediate solve, got {reward}"