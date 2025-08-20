from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from scripts.env import WordleEnv
from scripts.vocab import WordVocab
from scripts.sampler import WordSampler


class GymWordleEnv(gym.Env):
    """
    Gymnasium wrapper around the custom WordleEnv.
    - Observation: 158-dim float32 vector (see WordleEnv._build_observation doc)
    - Action space: Discrete(len(vocab))
    - info contains an 'action_mask' (int8 array) for valid actions at each step.
    """

    metadata = {"render.modes": []}

    def __init__(self, vocab: WordVocab, sampler: WordSampler, **kwargs) -> None:
        if not isinstance(vocab, WordVocab):
            raise TypeError("vocab must be a WordVocab")
        if not isinstance(sampler, WordSampler):
            raise TypeError("sampler must be a WordSampler")

        self.env = WordleEnv(vocab, sampler, **kwargs)
        self.vocab = vocab
        self._last_mask = None  # caches latest action mask for action-masking wrappers

        # Observation: 158 floats (5*26 pos freqs + 26 global freqs + log_rem + step_scaled)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(158,), dtype=np.float32
        )
        # Actions: indices into the vocab
        self.action_space = spaces.Discrete(len(vocab))

        # For Gymnasium seeding API
        self.np_random, _ = gym.utils.seeding.np_random(None)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # Follow Gymnasium reset protocol
        super().reset(seed=seed)
        obs, mask = self.env.reset()
        obs = np.asarray(obs, dtype=np.float32)
        info = {"action_mask": np.asarray(mask, dtype=np.int8)}
        self._last_mask = info["action_mask"]
        return obs, info

    def step(self, action: int):
        # Ensure action is an int within the discrete space
        action = int(action)
        if not self.action_space.contains(action):
            raise gym.error.InvalidAction(f"Invalid action: {action}")

        obs, reward, done, info, mask = self.env.step(action)

        obs = np.asarray(obs, dtype=np.float32)
        reward = float(reward)
        terminated = bool(done)
        truncated = False

        # Provide the current valid action mask to the agent
        info = dict(info)  # copy to avoid accidental mutation
        info["action_mask"] = np.asarray(mask, dtype=np.int8)
        self._last_mask = info["action_mask"]

        return obs, reward, terminated, truncated, info

    def get_action_mask(self) -> np.ndarray:
        """
        Return the latest valid action mask as an int8 numpy array.
        This is useful for sb3-contrib's ActionMasker wrapper, which expects
        env.unwrapped.get_action_mask() to be available.
        """
        if self._last_mask is None:
            # If called before reset(), compute a fresh one by resetting.
            obs, info = self.reset()
            return info["action_mask"]
        return self._last_mask


if __name__ == "__main__":
    # Quick smoke test if run directly (requires word_list.csv and answer-only loader)
    try:
        from scripts.data_utils import load_answer_vocab

        vocab = load_answer_vocab("word_list.csv")
        sampler = WordSampler(vocab, seed=0)
        gym_env = GymWordleEnv(vocab, sampler, allow_probe_guesses=False)
        obs, info = gym_env.reset()
        assert gym_env.observation_space.contains(obs)
        assert "action_mask" in info and np.any(info["action_mask"] == 1)

        # Take the first valid action
        valid = np.nonzero(info["action_mask"])[0]
        obs, rew, term, trunc, info = gym_env.step(int(valid[0]))
        print("gym_env step ok -> reward:", rew, "terminated:", term, "remaining:", np.sum(info["action_mask"]))

        # Example (commented): using sb3-contrib's ActionMasker
        # from sb3_contrib.common.wrappers import ActionMasker
        # masked_env = ActionMasker(gym_env, lambda e: e.unwrapped.get_action_mask())
        # print("ActionMasker attached. Sample action space:", masked_env.action_space)
    except Exception as e:
        print("Smoke test failed:", repr(e))
