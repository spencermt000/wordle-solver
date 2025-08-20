from __future__ import annotations

import random
from scripts.vocab import WordVocab


class WordSampler:
    def __init__(self, vocab: WordVocab, seed: int | None = None) -> None:
        # Validate vocab
        if not isinstance(vocab, WordVocab):
            raise TypeError("vocab must be a WordVocab")
        if len(vocab) == 0:
            raise ValueError("vocab is empty")

        # Store vocab
        self._vocab = vocab

        # Create RNG (deterministic if seed provided)
        self._rng = random.Random(seed)
        self._seed = seed

    def set_seed(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self._seed = seed

    def choice_index(self) -> int:
        n = len(self._vocab)
        if n == 0:
            raise ValueError("vocab is empty")
        return self._rng.randrange(n)

    def choice_word(self) -> str:
        return self._vocab.word_at(self.choice_index())

    def batch_indices(self, k: int) -> list[int]:
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        return [self.choice_index() for _ in range(k)]
