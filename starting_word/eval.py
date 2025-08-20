"""
starting_word/eval.py

Score candidate first guesses by how well they split the answer set.

Metrics per guess:
- exp_remaining: expected remaining candidates after the first feedback
- entropy: information gain (higher is better)
- worst_case: size of the largest bucket (lower is better)
- partitions: number of distinct feedback patterns induced

Usage:
  python -m starting_word.eval
"""

from __future__ import annotations

from collections import defaultdict
from math import log2
from typing import Iterable, List, Dict, Tuple
import time
import csv

from scripts.data_utils import load_answer_vocab
from scripts.feedback import score_pattern, pattern_to_int


def _pattern_histogram(guess: str, targets: Iterable[str]) -> Dict[int, int]:
    """
    For a given guess, compute a histogram over feedback patterns across all targets.
    Returns a dict: pattern_code -> count
    """
    counts: Dict[int, int] = defaultdict(int)
    for t in targets:
        patt = score_pattern(guess, t)
        code = pattern_to_int(patt)
        counts[code] += 1
    return counts


def _metrics_from_counts(counts: Dict[int, int], total: int) -> Tuple[float, float, int, int]:
    """
    Given a histogram of bucket counts and the total number of targets,
    compute (exp_remaining, entropy, worst_case, partitions).
    """
    if total <= 0:
        raise ValueError("total must be positive")
    exp_remaining = sum(c * c for c in counts.values()) / total
    # entropy in bits; guard against log2(0)
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0.0:
            entropy -= p * log2(p)
    worst_case = max(counts.values()) if counts else 0
    partitions = len(counts)
    return exp_remaining, entropy, worst_case, partitions


def evaluate_first_guesses(
    answers: List[str],
    guesses: List[str] | None = None,
    *,
    progress: bool = False,
) -> List[Dict[str, float]]:
    """
    Evaluate each candidate first guess against the full answer set.

    Parameters
    ----------
    answers : list[str]
        The set of possible targets (official past/future Wordle answers).
    guesses : list[str] | None
        Candidate guesses to score. If None, uses `answers`.
    progress : bool
        If True, prints a tiny progress indicator every 100 guesses.

    Returns
    -------
    list[dict]
        Sorted list (best first) of records with keys:
        'guess', 'exp_remaining', 'entropy', 'worst_case', 'partitions'
    """
    if not answers:
        raise ValueError("answers must be non-empty")
    pool = guesses if guesses is not None else answers
    N = len(answers)

    results: List[Dict[str, float]] = []
    for i, g in enumerate(pool):
        if len(g) != 5 or not g.isalpha() or not g.islower():
            # Skip invalid guess strings quietly; keep evaluation robust.
            continue
        counts = _pattern_histogram(g, answers)
        exp_remaining, entropy, worst_case, partitions = _metrics_from_counts(counts, N)
        results.append(
            {
                "guess": g,
                "exp_remaining": float(exp_remaining),
                "entropy": float(entropy),
                "worst_case": int(worst_case),
                "partitions": int(partitions),
            }
        )
        if progress and (i + 1) % 50 == 0:
            print(f"Scored {i+1}/{len(pool)} guesses...", flush=True)

    # Sort: primary = exp_remaining asc, secondary = worst_case asc, tertiary = -entropy desc
    results.sort(key=lambda r: (r["exp_remaining"], r["worst_case"], -r["entropy"]))
    return results


def _print_top(results: List[Dict[str, float]], k: int = 20) -> None:
    print(f"\nTop {k} starting words by expected remaining:")
    print(f"{'rank':>4}  {'guess':<8}  {'exp_rem':>8}  {'entropy':>8}  {'worst':>5}  {'parts':>6}")
    for idx, r in enumerate(results[:k], start=1):
        print(
            f"{idx:>4}  {r['guess']:<8}  {r['exp_remaining']:>8.2f}  {r['entropy']:>8.3f}  {int(r['worst_case']):>5}  {int(r['partitions']):>6}"
        )


def _write_csv(results: List[Dict[str, float]], path: str) -> None:
    fieldnames = ["guess", "exp_remaining", "entropy", "worst_case", "partitions"]
    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


if __name__ == "__main__":
    # Load official answer-only vocab
    vocab = load_answer_vocab("word_list.csv")
    answers = vocab.words()
    # For now, evaluate answers as both guesses and targets (common practice)
    print(f"Scoring {len(answers)} guesses against {len(answers)} answers...", flush=True)
    t0 = time.perf_counter()
    results = evaluate_first_guesses(answers, answers, progress=True)
    dt = time.perf_counter() - t0
    print(f"Done in {dt:.2f}s", flush=True)
    _print_top(results, k=20)
    _write_csv(results, "starting_word_results.csv")
    print("Wrote results to starting_word_results.csv")