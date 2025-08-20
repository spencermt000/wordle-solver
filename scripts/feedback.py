"""
Feedback utilities for Wordle.

This module intentionally leaves implementations blank and provides
step-by-step guidance in comments so you can write the logic yourself.
"""

from collections import Counter


def score_pattern(guess: str, target: str) -> list[int]:
    """
    Compute the 5-position Wordle feedback for `guess` against `target`.

    Returns
    -------
    list[int]
        A list of length 5 with values in {0, 1, 2} where:
        - 0 = gray  (letter not present OR over-used relative to target counts)
        - 1 = yellow (letter present but in a different position)
        - 2 = green (letter matches the target at that position)

    Contract & Validation (implement these checks first)
    ----------------------------------------------------
    - Enforce lowercase alphabetic strings of length 5 for both `guess` and `target`.
      Hint: Use `.isalpha()` and `len(...) == 5`. Decide whether to `.lower()` or raise.
      Recommendation: **raise ValueError** rather than silently coercing.
    - Return must always be a list of exactly 5 ints in {0,1,2}.

    Correct Duplicate Handling (two-pass rule)
    ------------------------------------------
    1) GREENS PASS:
       - Initialize an output list `pattern = [0,0,0,0,0]`.
       - Build a letter-frequency counter of the `target` (e.g., dict or collections.Counter).
       - For each index i in 0..4:
           if guess[i] == target[i]:
               pattern[i] = 2
               decrement the count for that letter in the target counter.
         (Only greens are marked in this pass.)

    2) YELLOWS PASS:
       - For each index i where pattern[i] is still 0:
           if the letter guess[i] still has a positive remaining count in the target counter:
               pattern[i] = 1
               decrement the count
           else:
               keep pattern[i] = 0 (gray)

    Edge Cases to Test After Implementation
    ---------------------------------------
    - All greens:      guess == target
    - No overlap:      distinct letters
    - Duplicates:
        * 'allot' vs 'total'  -> classic tricky case
        * 'press' vs 'spree'  -> repeated letters on both sides
        * 'abbey' vs 'cabin'  -> overuse of 'b' in guess relative to target

    Performance
    -----------
    - This is O(5) with small constant factors; clarity > micro-optimizations.

    """
    # Validation
    if not isinstance(guess, str) or not isinstance(target, str):
        raise TypeError("guess and target must be strings")
    if len(guess) != 5 or len(target) != 5:
        raise ValueError("guess and target must be length 5")
    if not guess.isalpha() or not target.isalpha():
        raise ValueError("guess and target must be alphabetic")
    if not guess.islower() or not target.islower():
        raise ValueError("guess and target must be lowercase")
    
    pattern: list[int] = [0, 0, 0, 0, 0]
    remaining = Counter(target)
    
    # Pass 1: mark greens and decrement availability
    for i, (g, t) in enumerate(zip(guess, target)):
        if g == t:
            pattern[i] = 2
            remaining[g] -= 1
    
    # Pass 2: mark yellows where counts allow (else gray)
    for i, g in enumerate(guess):
        if pattern[i] == 0:
            if remaining[g] > 0:
                pattern[i] = 1
                remaining[g] -= 1
            # else leave as 0 (gray)
    
    return pattern


def pattern_to_int(pattern: list[int]) -> int:
    """
    Encode a 5-trit pattern [p0,p1,p2,p3,p4] (each in {0,1,2}) into a single integer in [0, 242].

    Validation (do this first)
    --------------------------
    - Ensure `pattern` is a sequence of length 5.
    - Ensure each element is exactly 0, 1, or 2. Raise ValueError otherwise.

    Encoding (base-3 positional)
    ----------------------------
    - Define value = 0.
    - For each element p in order:
        value = value * 3 + p
    - Return value.

    Optional extension
    ------------------
    - You may later implement the inverse `int_to_pattern(x: int) -> list[int]`
      by repeatedly dividing by 3 (mod 3) and reversing the result to length 5.
    """
    # Validate
    if not isinstance(pattern, (list, tuple)):
        raise TypeError("pattern must be a list or tuple of 5 integers in {0,1,2}")
    if len(pattern) != 5:
        raise ValueError("pattern must have length 5")
    value = 0
    for p in pattern:
        if not isinstance(p, int) or p not in (0, 1, 2):
            raise ValueError("pattern elements must be integers in {0,1,2}")
        value = value * 3 + p
    return value


def consistent_with(word: str, guess: str, pattern: list[int]) -> bool:
    """
    Check if `word` (as a hypothetical target) is consistent with the given `(guess, pattern)`.

    Principle
    ---------
    - This should **delegate** to `score_pattern(guess, word)` and compare with `pattern`.
      Do not re-implement the scoring logic here to avoid divergence.

    Validation
    ----------
    - Enforce the same input constraints as `score_pattern`: lowercase, alphabetic, length 5.
    - Validate `pattern` exactly like in `pattern_to_int` (length 5, values in {0,1,2}).

    Return
    ------
    - `True` if `score_pattern(guess, word) == pattern`, else `False`.

    Usage
    -----
    - This will be crucial for filtering candidate sets and building an action mask in the RL env.
    """
    # Validate word/guess similarly to score_pattern
    if not isinstance(word, str) or not isinstance(guess, str):
        raise TypeError("word and guess must be strings")
    if len(word) != 5 or len(guess) != 5:
        raise ValueError("word and guess must be length 5")
    if not word.isalpha() or not guess.isalpha():
        raise ValueError("word and guess must be alphabetic")
    if not word.islower() or not guess.islower():
        raise ValueError("word and guess must be lowercase")
    # Validate pattern using the same rules as pattern_to_int (reuse its checks by calling it)
    _ = pattern_to_int(pattern)  # will raise if invalid
    return score_pattern(guess, word) == list(pattern)


if __name__ == "__main__":
    # Quick sanity checks
    assert score_pattern("crane", "crane") == [2, 2, 2, 2, 2]
    assert score_pattern("allot", "total") == [1, 1, 0, 1, 1]
    assert score_pattern("abbey", "cabin") == [1, 0, 2, 0, 0]
    assert score_pattern("press", "spree") == [1, 1, 1, 1, 0]
    # pattern_to_int checks
    assert pattern_to_int([2, 2, 2, 2, 2]) == 242
    assert pattern_to_int([0, 0, 0, 0, 0]) == 0
    print("feedback.py sanity checks passed.")