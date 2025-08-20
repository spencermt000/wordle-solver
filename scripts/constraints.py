def _li(c: str) -> int:
    """Map a lowercase letter to 0..25."""
    return ord(c) - 97
"""
constraints.py

Keeps track of Wordle-style constraints and filters candidate words.
"""

from typing import List, Tuple
from scripts.feedback import score_pattern

class ConstraintState:
    def __init__(self, word_length: int = 5, alphabet_size: int = 26):
        # slot-level allowance: True means the letter is possible in that position
        self.word_length = word_length
        self.alphabet_size = alphabet_size
        self.pos_allowed = [[True] * alphabet_size for _ in range(word_length)]
        self.min_counts = [0] * alphabet_size
        self.max_counts = [word_length] * alphabet_size

    def reset(self):
        self.pos_allowed = [[True] * self.alphabet_size for _ in range(self.word_length)]
        self.min_counts = [0] * self.alphabet_size
        self.max_counts = [self.word_length] * self.alphabet_size

    def apply_feedback(self, guess: str, pattern: List[int]):
        """
        Update constraints based on feedback pattern.
        - Green = fix the letter at that slot
        - Yellow = letter must be included but not in that slot
        - Gray = letter is absent OR limited count
        """
        # Validate inputs
        if not isinstance(guess, str) or len(guess) != self.word_length:
            raise ValueError("guess must be a lowercase alphabetic string of length 5")
        if not guess.isalpha() or not guess.islower():
            raise ValueError("guess must be lowercase alphabetic")
        if not isinstance(pattern, list) or len(pattern) != self.word_length:
            raise ValueError("pattern must be a list of length 5")
        if any((p not in (0, 1, 2)) for p in pattern):
            raise ValueError("pattern elements must be in {0,1,2}")
        
        # Pass 1: apply slot-level constraints for greens and yellows
        # Also collect stats for per-letter counts
        gy_counts = [0] * self.alphabet_size   # number of green+yellow occurrences per letter
        saw_gray  = [False] * self.alphabet_size
        
        for i, (ch, p) in enumerate(zip(guess, pattern)):
            li = _li(ch)
            if p == 2:
                # Fix letter at this slot: only this letter allowed here
                for a in range(self.alphabet_size):
                    self.pos_allowed[i][a] = (a == li)
                gy_counts[li] += 1
            elif p == 1:
                # Letter exists but not in this slot
                self.pos_allowed[i][li] = False
                gy_counts[li] += 1
            else:
                # Gray: this slot is not this letter (and may imply global exclusion/upper bound)
                self.pos_allowed[i][li] = False
                saw_gray[li] = True
        
        # Pass 2: update global min/max counts based on duplicates and grays
        for li in range(self.alphabet_size):
            k = gy_counts[li]
            if k > 0:
                # we must have at least k occurrences of this letter somewhere
                if self.min_counts[li] < k:
                    self.min_counts[li] = k
                # if we also saw a gray of this same letter, that means overuse occurred
                # ==> cap the max to exactly k (can't be more than k)
                if saw_gray[li]:
                    if self.max_counts[li] > k:
                        self.max_counts[li] = k
            else:
                # no green/yellow for this letter
                if saw_gray[li]:
                    # only grays -> the letter does not appear in target
                    self.max_counts[li] = 0
                    # disallow this letter in all positions
                    for i in range(self.word_length):
                        self.pos_allowed[i][li] = False


def filter_candidates(words: List[str], history: List[Tuple[str, List[int]]]) -> List[str]:
    """
    Keep only candidates that match *all* (guess, pattern) pairs in history.
    Uses score_pattern for correctness.
    """
    candidates = []
    for w in words:
        ok = True
        for guess, patt in history:
            if score_pattern(guess, w) != patt:
                ok = False
                break
        if ok:
            candidates.append(w)
    return candidates