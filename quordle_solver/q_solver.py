"""
quordle_solver/q_solver.py

Interactive Quordle-style helper (4 Wordles at once).
- You enter a single guess; all four boards use that guess.
- Then you paste the feedback for EACH board (g/y/b, 2/1/0, or [0,1,2,2,0]).
- The solver prunes each board's candidate pool and recommends the next guess
  by minimizing the SUM of expected remaining across all *unsolved* boards.
- Repeats until all boards are solved or you quit.

Run:
  python -m quordle_solver.q_solver --csv word_list.csv

Shortcuts:
  quit / q / exit  -> exit at any prompt
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from math import log2
from typing import List, Dict, Tuple

from scripts.data_utils import load_answer_vocab
from scripts.vocab import WordVocab
from scripts.feedback import score_pattern, pattern_to_int
from scripts.constraints import filter_candidates


# ---------------------------
# Parsing & scoring helpers
# ---------------------------

def parse_feedback(s: str) -> List[int]:
    """
    Parse a 5-char feedback string into [0/1/2]*5.
    Accepted:
      - letters: g/y/b  (green/yellow/black)
      - digits:  2/1/0
      - list:   [0, 1, 2, 2, 0]
    """
    s = s.strip().lower()
    # List-like form: [0,1,2,2,0]
    if s.startswith("[") and s.endswith("]"):
        nums = re.findall(r"[012]", s)
        if len(nums) != 5:
            raise ValueError("list form must contain exactly five 0/1/2 values")
        return [int(x) for x in nums]
    mapping = {"g": 2, "y": 1, "b": 0, "2": 2, "1": 1, "0": 0}
    if len(s) != 5:
        raise ValueError("feedback must be 5 characters (gybgy / 21001 / [0,1,2,2,0])")
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
    """
    Compute heuristic metrics for a guess against the current candidate pool.
    Returns dict: exp_remaining, entropy, worst_case, partitions
    """
    if not pool:
        return {"exp_remaining": 0.0, "entropy": 0.0, "worst_case": 0, "partitions": 0}
    counts = _pattern_histogram(guess, pool)
    N = len(pool)
    exp_remaining = sum(c * c for c in counts.values()) / N
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


def rank_global_candidates(candidate_guess_pool: List[str], pools: List[List[str]]) -> List[Dict[str, float]]:
    """
    Rank guesses by summing expected-remaining over all *unsolved* boards.

    candidate_guess_pool : words we consider as guesses (e.g., union of pools)
    pools                : list of candidate pools per board (for solved boards, pass empty list)
    """
    rows: List[Dict[str, float]] = []
    # Precompute per-board metrics cache for speed
    for g in candidate_guess_pool:
        sum_exp = 0.0
        sum_worst = 0
        sum_entropy = 0.0
        for pool in pools:
            if not pool:
                continue  # solved board contributes 0
            m = score_guess_on_pool(g, pool)
            sum_exp += m["exp_remaining"]
            sum_worst += int(m["worst_case"])
            sum_entropy += m["entropy"]
        rows.append(
            {
                "guess": g,
                "sum_exp_remaining": sum_exp,
                "sum_worst_case": sum_worst,
                "sum_entropy": sum_entropy,
            }
        )
    # Sort: minimize sum_exp, then sum_worst; tie-break by -sum_entropy
    rows.sort(key=lambda r: (r["sum_exp_remaining"], r["sum_worst_case"], -r["sum_entropy"]))
    return rows


# ---------------------------
# Main interactive loop
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Interactive Quordle solver (4 simultaneous boards).")
    ap.add_argument("--csv", default="word_list.csv", help="Path to word_list.csv (answers list).")
    args = ap.parse_args()

    vocab = load_answer_vocab(args.csv)
    answers = vocab.words()

    # Four boards: each with its own history and candidate pool
    K = 4
    histories: List[List[Tuple[str, List[int]]]] = [[] for _ in range(K)]
    pools: List[List[str]] = [answers.copy() for _ in range(K)]
    solved: List[bool] = [False for _ in range(K)]

    print("\nQuordle helper — enter your guess once, then enter feedback for each board.\n"
          "Feedback accepted: g/y/b, 2/1/0, or [0,1,2,2,0]. Type 'quit' to exit.\n")

    step = 1
    while True:
        # Check if all solved
        if all(solved):
            print("All boards solved! 🎉")
            return

        # Decide guess: user types a word or presses Enter to use our suggestion
        # First compute a suggestion based on current pools (for convenience)
        # Candidate guess set = union of pools from unsolved boards; fallback to full answers
        unsolved_pools = [p if not solved[i] else [] for i, p in enumerate(pools)]
        guess_pool = sorted(set(w for p in unsolved_pools for w in p))
        if not guess_pool:
            guess_pool = answers

        suggestion_rows = rank_global_candidates(guess_pool, unsolved_pools)
        suggestion = suggestion_rows[0]["guess"] if suggestion_rows else (guess_pool[0] if guess_pool else "")

        print(f"Step {step}:")
        user_guess = input(f"Enter your guess word (Enter to use suggestion '{suggestion}'): ").strip().lower()
        if user_guess in {"q", "quit", "exit"}:
            print("bye!")
            return
        guess = user_guess if user_guess else suggestion

        if len(guess) != 5 or not guess.isalpha():
            print("Please enter a valid 5-letter word.")
            continue

        # Collect feedback for each board (skip boards already solved)
        new_feedbacks: List[List[int] | None] = [None] * K
        for b in range(K):
            if solved[b]:
                continue
            while True:
                fb = input(f"  Board {b+1} feedback (g/y/b or 2/1/0 or [..]): ").strip()
                if fb.lower() in {"q", "quit", "exit"}:
                    print("bye!")
                    return
                try:
                    patt = parse_feedback(fb)
                    new_feedbacks[b] = patt
                    break
                except ValueError as e:
                    print("Invalid feedback:", e)

        # Update each board's history/pool
        for b in range(K):
            if solved[b]:
                continue
            patt = new_feedbacks[b]
            assert patt is not None
            histories[b].append((guess, patt))
            if patt == [2, 2, 2, 2, 2]:
                solved[b] = True
                pools[b] = []
                print(f"  ✓ Board {b+1} solved!")
                continue
            pools[b] = filter_candidates(pools[b], histories[b])
            rem = len(pools[b])
            print(f"  Board {b+1}: remaining candidates = {rem}")
            if rem <= 10 and rem > 0:
                print("    Candidates:", ", ".join(pools[b]))

            if rem == 0:
                print(f"  ⚠️ Board {b+1} has no candidates left; check the feedback you entered.")

        # Show next suggestions (top 3) for the combined objective
        unsolved_pools = [p if not solved[i] else [] for i, p in enumerate(pools)]
        guess_pool = sorted(set(w for p in unsolved_pools for w in p)) or answers
        ranked = rank_global_candidates(guess_pool, unsolved_pools)
        if not ranked:
            print("No valid next guesses could be ranked. Try a common probe (e.g., 'crane').")
        else:
            top = ranked[:3]
            print("Top combined suggestions:")
            for i, r in enumerate(top, 1):
                print(f"  {i}. {r['guess']}  (Σexp_rem={r['sum_exp_remaining']:.2f}, Σworst={int(r['sum_worst_case'])}, ΣH={r['sum_entropy']:.3f})")

        step += 1


if __name__ == "__main__":
    main()
