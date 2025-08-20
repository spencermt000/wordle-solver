"""
env.py

A minimal Wordle environment that you can plug into an RL loop.
- Discrete actions: indices into the WordVocab
- Observations: summary stats over the current candidate set
- Rewards: shaped by greens/yellows with a per-step penalty and success bonus
"""

from __future__ import annotations

import math
from typing import List, Tuple, Dict, Optional

from scripts.vocab import WordVocab
from scripts.sampler import WordSampler
from scripts.feedback import score_pattern
from scripts.constraints import filter_candidates


class WordleEnv:
    """
    Wordle environment.

    API
    ---
    reset(target_idx: Optional[int] = None) -> tuple[list[float], list[int]]
        Starts a new episode. If `target_idx` is provided, uses that word as target (useful for tests).
        Returns (observation, action_mask).

    step(action_idx: int) -> tuple[list[float], float, bool, dict, list[int]]
        Applies a guess. Returns (observation, reward, done, info, action_mask).

    Observation
    -----------
    A 1D vector composed of:
      - Positional letter frequencies over the remaining candidate set (5*26 = 130 floats, normalized)
      - Global letter frequencies over candidates (26 floats, normalized)
      - log(1 + #candidates) (1 float)
      - step index scaled by max steps (1 float)
    Total length = 130 + 26 + 1 + 1 = 158

    Action Mask
    -----------
    If `allow_probe_guesses` is False, mask allows only current candidates.
    If True, all vocab actions are allowed (probes permitted).
    """

    def __init__(
        self,
        vocab: WordVocab,
        sampler: WordSampler,
        *,
        max_guesses: int = 6,
        alpha: float = 2.0,         # reward per green
        beta: float = 1.0,          # reward per yellow
        step_penalty: float = 1.0,  # per-step cost
        success_bonus: float = 10.0,
        allow_probe_guesses: bool = False,
    ) -> None:
        if not isinstance(vocab, WordVocab):
            raise TypeError("vocab must be a WordVocab")
        if not isinstance(sampler, WordSampler):
            raise TypeError("sampler must be a WordSampler")
        if len(vocab) == 0:
            raise ValueError("empty vocabulary")

        self.vocab = vocab
        self.sampler = sampler
        self.max_guesses = int(max_guesses)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.step_penalty = float(step_penalty)
        self.success_bonus = float(success_bonus)
        self.allow_probe_guesses = bool(allow_probe_guesses)

        # Episode state
        self._target_idx: Optional[int] = None
        self._target_word: Optional[str] = None
        self._history: List[Tuple[str, List[int]]] = []
        self._candidates: List[int] = []
        self._step: int = 0

    # -------------------------
    # Core env API
    # -------------------------
    def reset(self, target_idx: Optional[int] = None) -> tuple[list[float], list[int]]:
        """Start a new episode and return (observation, action_mask)."""
        if target_idx is None:
            self._target_idx = self.sampler.choice_index()
        else:
            if target_idx < 0 or target_idx >= len(self.vocab):
                raise IndexError(f"target_idx out of range: {target_idx}")
            self._target_idx = target_idx
        self._target_word = self.vocab.word_at(self._target_idx)

        self._history = []
        self._step = 0
        self._candidates = list(range(len(self.vocab)))

        obs = self._build_observation()
        mask = self._build_action_mask()
        return obs, mask

    def step(self, action_idx: int) -> tuple[list[float], float, bool, dict, list[int]]:
        """
        Take an action (guess index).

        Returns
        -------
        observation: list[float]
        reward: float
        done: bool
        info: dict  (includes 'guess', 'pattern', 'remaining')
        action_mask: list[int]
        """
        if action_idx < 0 or action_idx >= len(self.vocab):
            raise IndexError(f"action index out of range: {action_idx}")
        if (not self.allow_probe_guesses) and (action_idx not in self._candidates):
            raise ValueError("action not allowed by current candidate set (set allow_probe_guesses=True to permit probes)")

        guess = self.vocab.word_at(action_idx)
        pattern = score_pattern(guess, self._target_word)  # type: ignore[arg-type]

        # Reward shaping
        greens = sum(1 for p in pattern if p == 2)
        yellows = sum(1 for p in pattern if p == 1)
        reward = self.alpha * greens + self.beta * yellows - self.step_penalty

        done = False
        if greens == 5:
            reward += self.success_bonus
            done = True

        # Update history and step count
        self._history.append((guess, pattern))
        self._step += 1

        # Terminal if max steps reached (and not solved)
        if not done and self._step >= self.max_guesses:
            done = True

        # Update candidates using all history so far
        if not done:
            words_full = self.vocab.words()
            filtered = filter_candidates(words_full, self._history)
            # Convert to indices
            self._candidates = [self.vocab.index_of(w) for w in filtered]
        else:
            # Episode over: freeze candidates to either [target] or current set
            self._candidates = [self._target_idx] if self._target_idx is not None else []

        obs = self._build_observation()
        mask = self._build_action_mask()

        info = {
            "guess": guess,
            "pattern": pattern,
            "remaining": len(self._candidates),
            "step": self._step,
            "solved": greens == 5,
            "target": self._target_word if done else None,
        }
        return obs, reward, done, info, mask

    # -------------------------
    # Helpers
    # -------------------------
    def _build_action_mask(self) -> list[int]:
        """Return a 0/1 mask over the entire vocab."""
        n = len(self.vocab)
        if self.allow_probe_guesses:
            return [1] * n
        allowed = set(self._candidates)
        return [1 if i in allowed else 0 for i in range(n)]

    def _build_observation(self) -> list[float]:
        """
        Build a fixed-length observation vector from the current candidate set.

        Layout:
          - pos_freqs: 5*26 normalized positional frequencies
          - global_freqs: 26 normalized letter frequencies
          - log_rem: log(1 + remaining)
          - step_scaled: step / max_guesses
        """
        n = len(self._candidates)
        # Avoid zero division; if no candidates (shouldn't happen mid-episode), fall back to vocab
        pool = self._candidates if n > 0 else list(range(len(self.vocab)))
        words = [self.vocab.word_at(i) for i in pool]
        n = len(words)

        # Positional frequencies (counts)
        pos_counts = [[0] * 26 for _ in range(5)]
        global_counts = [0] * 26

        for w in words:
            for pos, ch in enumerate(w):
                li = ord(ch) - 97
                pos_counts[pos][li] += 1
                global_counts[li] += 1

        # Normalize positional counts by n (number of words in pool)
        pos_freqs: List[float] = []
        for pos in range(5):
            for li in range(26):
                pos_freqs.append(pos_counts[pos][li] / n if n > 0 else 0.0)

        # Normalize global counts by (5 * n) to keep on the same 0..1 scale
        denom = 5 * n if n > 0 else 1
        global_freqs = [c / denom for c in global_counts]

        log_rem = math.log1p(len(self._candidates))
        step_scaled = self._step / max(1, self.max_guesses)

        obs = pos_freqs + global_freqs + [log_rem, step_scaled]
        return obs

    # -------------------------
    # Introspection helpers
    # -------------------------
    @property
    def history(self) -> List[Tuple[str, List[int]]]:
        return list(self._history)

    @property
    def target(self) -> Optional[str]:
        return self._target_word

    @property
    def remaining_candidates(self) -> int:
        return len(self._candidates)


if __name__ == "__main__":
    # Quick smoke test with a tiny vocab
    words = ["crane", "trace", "adieu", "cigar", "rebut", "total", "stoal", "allot"]
    vocab = WordVocab(words)
    sampler = WordSampler(vocab, seed=123)

    env = WordleEnv(vocab, sampler, allow_probe_guesses=False)
    obs, mask = env.reset(target_idx=vocab.index_of("total"))
    # Take a guess "allot" and expect the pattern [1,1,0,1,1], and candidate set to include "total" and maybe others
    guess_idx = vocab.index_of("allot")
    obs, r, done, info, mask = env.step(guess_idx)
    print("info:", info)
    print("remaining:", env.remaining_candidates)
    print("done:", done)
    # Continue randomly from allowed actions
    import random
    while not done:
        choices = [i for i, m in enumerate(mask) if m == 1]
        a = random.choice(choices)
        obs, r, done, info, mask = env.step(a)
        # print(info)
    print("target:", env.target, "solved:", info["solved"])
