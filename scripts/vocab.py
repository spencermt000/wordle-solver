from __future__ import annotations
from typing import List
import pandas as pd


class WordVocab:
    def __init__(self, words: List[str]) -> None:
        if not isinstance(words, list):
            raise TypeError("`words` must be a list of strings")
        if not words:
            raise ValueError("no words provided")
        if not all(isinstance(w, str) for w in words):
            raise TypeError("all items in `words` must be str")

        # Enforce uniqueness (first occurrence policy should be handled by from_csv)
        if len(set(words)) != len(words):
            raise ValueError("duplicate words detected; input to WordVocab must be deduplicated")

        self._words: List[str] = list(words)  # make a defensive copy
        self._index = {w: i for i, w in enumerate(self._words)}

        # Invariant check
        for w, i in self._index.items():
            if self._words[i] != w:
                raise AssertionError("index invariant violated")

    # ---------- Construction helpers ----------

    @classmethod
    def from_csv(
        cls,
        path: str,
        column: str = "word",
        *,
        word_len: int = 5,
        lowercase: bool = True,
        dedupe: bool = True,
        alpha_only: bool = True,
    ) -> "WordVocab":
        """
        Load words from a CSV and build a WordVocab.

        Parameters
        ----------
        path : str
            Path to CSV file.
        column : str
            Column name containing words.
        word_len : int, default=5
            Required word length.
        lowercase : bool, default=True
            If True, lowercase words before validation.
        dedupe : bool, default=True
            If True, keep the first occurrence and drop later duplicates.
        alpha_only : bool, default=True
            If True, keep only alphabetic words (str.isalpha()).

        Raises
        ------
        FileNotFoundError, KeyError, ValueError, TypeError
        """
        df = pd.read_csv(path)
        if column not in df.columns:
            raise KeyError(f"column '{column}' not found in {path}")

        raw_iter = df[column].tolist()

        clean: List[str] = []
        seen = set()

        for val in raw_iter:
            if not isinstance(val, str):
                val = str(val) if val is not None else ""
            w = val.lower() if lowercase else val

            if len(w) != word_len:
                continue
            if alpha_only and not w.isalpha():
                continue

            if dedupe:
                if w in seen:
                    continue
                seen.add(w)

            clean.append(w)

        if not clean:
            raise ValueError("no valid words after filtering")

        return cls(clean)

    # ---------- Basic protocol ----------

    def __len__(self) -> int:
        """Number of words in the vocabulary."""
        return len(self._words)

    def words(self) -> List[str]:
        """Return a copy of the internal word list (to avoid external mutation)."""
        return list(self._words)

    def contains(self, word: str) -> bool:
        """Return True iff `word` exists in the vocabulary (case-sensitive)."""
        return word in self._index

    def index_of(self, word: str) -> int:
        """Return the index for `word`; raise KeyError if unknown."""
        try:
            return self._index[word]
        except KeyError:
            raise KeyError(f"unknown word: {word}") from None

    def word_at(self, idx: int) -> str:
        """Return the word at position `idx`; raise IndexError if out of bounds."""
        if idx < 0 or idx >= len(self._words):
            raise IndexError(f"index out of range: {idx}")
        return self._words[idx]

    def to_indices(self, words: List[str]) -> List[int]:
        """Convert a list of words to indices; raise KeyError on the first missing."""
        out: List[int] = []
        for w in words:
            out.append(self.index_of(w))
        return out

    def to_words(self, indices: List[int]) -> List[str]:
        """Convert a list of indices to words; raise IndexError on the first invalid index."""
        out: List[str] = []
        for i in indices:
            out.append(self.word_at(i))
        return out

if __name__ == "__main__":
    v = WordVocab.from_csv("word_list.csv", column="word")
    assert len(v) > 0
    w0 = v.word_at(0)
    assert v.index_of(w0) == 0
    assert v.contains(w0) is True
    assert v.to_words([0, 1]) == [v.word_at(0), v.word_at(1)]
    print("WordVocab sanity checks passed.")