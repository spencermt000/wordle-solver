"""
solver/solver_cli.py

Interactive Wordle helper (human-in-the-loop):
- YOU type the first guessed word (no defaults) and the feedback pattern you saw.
- Feedback accepted as: 'gybby', '21001', or a Python-like list '[0, 0, 2, 2, 2]'.
- The solver prunes candidates and shows the top 3 next guesses.
- You can enter any next guess you want and then the new feedback; repeat until solved.

Run:
  python -m solver.solver_cli --csv word_list.csv

Shortcuts:
  quit / q / exit  -> exit
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from math import log2
from typing import List, Dict

from scripts.data_utils import load_answer_vocab
from scripts.vocab import WordVocab
from scripts.feedback import score_pattern, pattern_to_int
from scripts.constraints import filter_candidates


def _load_vocab(csv_path: str) -> WordVocab:
    # Use official answers for interactive solving; switch to full list if you prefer.
    return load_answer_vocab(csv_path)


def parse_feedback(s: str) -> List[int]:
    """Parse a 5-char feedback into a list of ints [0/1/2].
    Accepted forms:
      - letters: g/y/b  (green/yellow/black)
      - digits:  2/1/0
      - list:   [0, 1, 2, 2, 0]
    Raises ValueError on invalid input.
    """
    s = s.strip().lower()
    # List-like form: [0,1,2,2,0]
    if s.startswith("[") and s.endswith("]"):
        nums = re.findall(r"[012]", s)
        if len(nums) != 5:
            raise ValueError("list form must contain exactly five 0/1/2 values")
        return [int(x) for x in nums]

    # letters or digits
    mapping = {"g": 2, "y": 1, "b": 0, "2": 2, "1": 1, "0": 0}
    if len(s) != 5:
        raise ValueError("feedback must be length 5 (gybgy / 21001 / [0,1,2,2,0])")
    try:
        return [mapping[ch] for ch in s]
    except KeyError as e:
        raise ValueError("feedback must use only g/y/b or 2/1/0") from e


def _pattern_histogram(guess: str, targets: List[str]) -> Dict[int, int]:
    counts: Dict[int, int] = defaultdict(int)
    for t in targets:
        code = pattern_to_int(score_pattern(guess, t))
        counts[code] += 1
    return counts


def score_guess_on_pool(guess: str, pool: List[str]) -> Dict[str, float]:
    """Compute heuristic metrics for a guess against the current candidate pool."""
    counts = _pattern_histogram(guess, pool)
    N = len(pool)
    exp_remaining = sum(c * c for c in counts.values()) / max(1, N)
    entropy = 0.0
    for c in counts.values():
        p = c / N
        if p > 0:
            entropy -= p * log2(p)
    worst_case = max(counts.values()) if counts else 0
    return {
        "exp_remaining": float(exp_remaining),
        "entropy": float(entropy),
        "worst_case": int(worst_case),
        "partitions": int(len(counts)),
    }


def rank_candidates(candidates: List[str]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for g in candidates:
        m = score_guess_on_pool(g, candidates)
        m["guess"] = g
        rows.append(m)
    rows.sort(key=lambda r: (r["exp_remaining"], r["worst_case"], -r["entropy"]))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Interactive Wordle solver (manual feedback)")
    ap.add_argument("--csv", default="word_list.csv", help="Path to word_list.csv")
    args = ap.parse_args()

    vocab = _load_vocab(args.csv)
    answers = vocab.words()

    history: List[tuple[str, List[int]]] = []
    candidates: List[str] = answers.copy()

    print("\nWordle helper — after EACH guess you make in the game, paste the feedback here.")
    print("Accepted: g/y/b, 2/1/0, or [0,1,2,2,0]. Type 'quit' to exit.\n")

    # ----- first move (you decide, no defaults) -----
    step = 1
    while True:
        guess = input("Enter your guess word: ").strip().lower()
        if guess in {"q", "quit", "exit"}:
            print("bye!")
            return
        if len(guess) != 5 or not guess.isalpha():
            print("Please enter a 5-letter alphabetic word.")
            continue
        break

    while True:
        fb = input("Feedback for that guess (g/y/b or 2/1/0 or [..]): ").strip()
        if fb.lower() in {"q", "quit", "exit"}:
            print("bye!")
            return
        try:
            patt = parse_feedback(fb)
            break
        except ValueError as e:
            print("Invalid feedback:", e)

    history.append((guess, patt))
    if patt == [2, 2, 2, 2, 2]:
        print("Solved! 🎉")
        return

    candidates = filter_candidates(candidates, history)
    print(f"Remaining candidates: {len(candidates)}")
    if len(candidates) <= 10:
        print("Candidates:", ", ".join(candidates))

    # ----- subsequent moves -----
    while True:
        ranked = rank_candidates(candidates)
        if not ranked:
            print("No candidates remain. Check your feedback inputs.")
            return
        top = ranked[:3]
        print("Top suggestions:")
        for i, r in enumerate(top, 1):
            print(f"  {i}. {r['guess']}  (exp_rem={r['exp_remaining']:.2f}, worst={int(r['worst_case'])}, H={r['entropy']:.3f})")

        nxt = input("Type your next guess (or press Enter to use #1): ").strip().lower()
        if nxt in {"q", "quit", "exit"}:
            print("bye!")
            return
        if not nxt:
            guess = top[0]["guess"]
        else:
            if len(nxt) != 5 or not nxt.isalpha():
                print("Please enter a valid 5-letter word.")
                continue
            guess = nxt

        # Get feedback and update
        while True:
            fb = input("Feedback (g/y/b or 2/1/0 or [..]): ").strip()
            if fb.lower() in {"q", "quit", "exit"}:
                print("bye!")
                return
            try:
                patt = parse_feedback(fb)
                break
            except ValueError as e:
                print("Invalid feedback:", e)

        history.append((guess, patt))
        if patt == [2, 2, 2, 2, 2]:
            print("Solved! 🎉")
            return

        candidates = filter_candidates(candidates, history)
        print(f"Remaining candidates: {len(candidates)}")
        if len(candidates) <= 10:
            print("Candidates:", ", ".join(candidates))
        if not candidates:
            print("No candidates remain. Check your feedback inputs.")
            return


if __name__ == "__main__":
    main()
