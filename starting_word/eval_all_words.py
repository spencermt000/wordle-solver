"""
starting_word/eval_all_words.py

Evaluate starting words using the ENTIRE word_list.csv as both targets and guesses.
This uses the analytical scoring (expected remaining, entropy, worst-case, partitions)
from starting_word.eval.evaluate_first_guesses.

Usage:
  python -m starting_word.eval_all_words
  python -m starting_word.eval_all_words --csv word_list.csv --out all_words_results.csv --top 30 --limit-guesses 500
"""

from __future__ import annotations

import argparse
import csv
import time
from typing import List, Dict

import pandas as pd

from scripts.vocab import WordVocab
from starting_word.eval import evaluate_first_guesses


def _load_all_words(csv_path: str) -> WordVocab:
    """
    Load ALL 5-letter alphabetic words from word_list.csv (deduped, lowercased),
    ignoring the 'day' filter entirely.
    """
    df = pd.read_csv(csv_path)
    if "word" not in df.columns:
        raise KeyError("CSV must contain a 'word' column")
    words: List[str] = []
    seen = set()
    for w in df["word"].astype(str).str.lower():
        if len(w) == 5 and w.isalpha() and w not in seen:
            seen.add(w)
            words.append(w)
    return WordVocab(words)


def _write_csv(rows: List[Dict[str, float]], path: str) -> None:
    if not rows:
        return
    fieldnames = ["guess", "exp_remaining", "entropy", "worst_case", "partitions"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})


def _print_top(results: List[Dict[str, float]], k: int = 20) -> None:
    print(f"\nTop {k} starting words by expected remaining:")
    print(f"{'rank':>4}  {'guess':<8}  {'exp_rem':>8}  {'entropy':>8}  {'worst':>5}  {'parts':>6}")
    for idx, r in enumerate(results[:k], start=1):
        print(
            f"{idx:>4}  {r['guess']:<8}  {r['exp_remaining']:>8.2f}  {r['entropy']:>8.3f}  {int(r['worst_case']):>5}  {int(r['partitions']):>6}"
        )


def main():
    ap = argparse.ArgumentParser(description="Evaluate starting words using the entire CSV word list.")
    ap.add_argument("--csv", default="word_list.csv", help="Path to word_list.csv")
    ap.add_argument("--out", default="all_words_results.csv", help="Output CSV filename")
    ap.add_argument("--top", type=int, default=15, help="How many top rows to print")
    ap.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show progress during evaluation (use --no-progress to disable)",
    )
    ap.add_argument("--limit-guesses", type=int, default=None, help="Evaluate only the first K guesses (for speed)")
    args = ap.parse_args()

    vocab = _load_all_words(args.csv)
    words = vocab.words()

    if args.limit_guesses is not None:
        words = words[: args.limit_guesses]

    print(f"Scoring {len(words)} guesses against {len(vocab)} targets (all words)...", flush=True)
    t0 = time.perf_counter()
    results = evaluate_first_guesses(answers=vocab.words(), guesses=words, progress=args.progress)
    dt = time.perf_counter() - t0
    print(f"Done in {dt:.2f}s", flush=True)

    _print_top(results, k=args.top)
    _write_csv(results, args.out)
    print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()