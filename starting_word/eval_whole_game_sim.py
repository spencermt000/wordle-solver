"""
starting_word/eval_whole_game_sim.py

Simulate *full games* to evaluate starting words.
For each starting word, we sample random targets and play until solved or max steps
using a simple policy after the first guess (heuristic or random).

Usage examples:
  python -m starting_word.eval_whole_game_sim --episodes 300 --answers-only --first raise irate slate crane
  python -m starting_word.eval_whole_game_sim --episodes 200 --targets all --guesses answers --limit-guesses 100
  python -m starting_word.eval_whole_game_sim --episodes 200 --answers-only --policy random

Outputs a CSV with metrics per starting word and prints the top K.
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics as stats
import time
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from scripts.vocab import WordVocab
from scripts.sampler import WordSampler
from scripts.env import WordleEnv
from scripts.data_utils import load_answer_vocab
from scripts.feedback import score_pattern


# -----------------------------
# Loading helpers
# -----------------------------

def _load_all_words(csv_path: str) -> WordVocab:
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


# -----------------------------
# Policy helpers
# -----------------------------

def _expected_step_reward(vocab: WordVocab, guess_idx: int, candidate_indices: List[int]) -> float:
    g = vocab.word_at(guess_idx)
    total = 0.0
    for t_idx in candidate_indices:
        t = vocab.word_at(t_idx)
        patt = score_pattern(g, t)
        greens = sum(1 for p in patt if p == 2)
        yellows = sum(1 for p in patt if p == 1)
        total += (2.0 * greens + 1.0 * yellows)
    return total / max(1, len(candidate_indices))


def _choose_action(vocab: WordVocab, mask: List[int], policy: str) -> int:
    allowed = [i for i, m in enumerate(mask) if m == 1]
    if not allowed:
        return 0
    if policy == "random":
        return random.choice(allowed)
    # heuristic
    return max(allowed, key=lambda i: _expected_step_reward(vocab, i, allowed))


# -----------------------------
# Simulation core
# -----------------------------

def _run_episode(env: WordleEnv, first_idx: int, target_idx: int, policy: str = "heuristic") -> Tuple[bool, int]:
    obs, mask = env.reset(target_idx=target_idx)
    # First guess is fixed
    obs, reward, done, info, mask = env.step(first_idx)
    steps = 1
    # Continue with policy
    while not done:
        action = _choose_action(env.vocab, mask, policy)
        obs, reward, done, info, mask = env.step(action)
        steps += 1
    return bool(info.get("solved", False)), steps


def evaluate_starting_words(
    vocab: WordVocab,
    targets: List[str],
    guesses: List[str],
    *,
    episodes: int = 200,
    policy: str = "heuristic",
    seed: int = 0,
    progress: bool = True,
) -> List[Dict[str, float]]:
    rng = random.Random(seed)
    target_indices = [vocab.index_of(w) for w in targets]
    results: List[Dict[str, float]] = []

    for gi, g in enumerate(guesses, start=1):
        try:
            first_idx = vocab.index_of(g)
        except KeyError:
            continue

        env = WordleEnv(vocab, WordSampler(vocab, seed=42), allow_probe_guesses=False)

        solved = 0
        steps_list: List[int] = []

        for ep in range(episodes):
            t_idx = rng.choice(target_indices)
            done, steps = _run_episode(env, first_idx, t_idx, policy=policy)
            if done:
                solved += 1
                steps_list.append(steps)

        solve_rate = solved / episodes if episodes > 0 else 0.0
        avg_steps = (sum(steps_list) / len(steps_list)) if steps_list else float("nan")
        median = stats.median(steps_list) if steps_list else float("nan")
        p90 = stats.quantiles(steps_list, n=10)[8] if len(steps_list) >= 10 else float("nan")
        p95 = stats.quantiles(steps_list, n=20)[18] if len(steps_list) >= 20 else float("nan")

        results.append(
            {
                "guess": g,
                "episodes": episodes,
                "solve_rate": round(solve_rate, 4),
                "avg_steps_solved": round(avg_steps, 3) if steps_list else float("nan"),
                "median_steps": median if steps_list else float("nan"),
                "p90_steps": p90,
                "p95_steps": p95,
                "solved": solved,
            }
        )
        if progress and gi % 10 == 0:
            print(f"Evaluated {gi}/{len(guesses)} starting words...", flush=True)

    results.sort(key=lambda r: (-r["solve_rate"], r["avg_steps_solved"] if r["avg_steps_solved"] == r["avg_steps_solved"] else 1e9))
    return results


def _write_csv(rows: List[Dict[str, float]], path: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Whole-game Monte Carlo evaluation of Wordle starting words.")
    ap.add_argument("--csv", default="word_list.csv", help="Path to word_list.csv")
    ap.add_argument("--episodes", type=int, default=200, help="Episodes per starting word")
    ap.add_argument("--targets", choices=["answers", "all"], default="answers", help="Target set")
    ap.add_argument("--guesses", choices=["answers", "all"], default="answers", help="Guess set to evaluate")
    ap.add_argument(
        "--first", nargs="*", default=None,
        help="Explicit list of starting guesses to test"
    )
    ap.add_argument(
        "--first-file", type=str, default=None,
        help="Path to a text file of starting guesses (one per line)"
    )
    ap.add_argument("--limit-guesses", type=int, default=None, help="Evaluate only the first K guesses (for speed)")
    ap.add_argument("--policy", choices=["heuristic", "random"], default="heuristic", help="Policy after first guess")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for target sampling")
    ap.add_argument("--out", default="whole_game_results.csv", help="Output CSV path")
    ap.add_argument("--top", type=int, default=10, help="How many top rows to print")
    ap.add_argument("--progress", action="store_true", default=True, help="Show progress (enabled by default)")
    args = ap.parse_args()

    # Load vocab/targets
    if args.targets == "answers":
        vocab = load_answer_vocab(args.csv)
        targets = vocab.words()
    else:
        vocab = _load_all_words(args.csv)
        targets = vocab.words()

    # Guesses pool
    if args.first:
        guesses = [w.lower() for w in args.first]
    elif args.first_file:
        with open(args.first_file, "r", encoding="utf-8") as f:
            guesses = [line.strip().lower() for line in f if line.strip()]
    elif args.guesses == "answers":
        guesses = load_answer_vocab(args.csv).words() if args.targets == "all" else targets
    else:
        guesses = vocab.words()

    if args.limit_guesses is not None:
        guesses = guesses[: args.limit_guesses]

    print(f"Evaluating {len(guesses)} starting words over {args.episodes} episodes each (targets={args.targets}, guesses={args.guesses}, policy={args.policy})", flush=True)
    t0 = time.perf_counter()
    rows = evaluate_starting_words(vocab, targets, guesses, episodes=args.episodes, policy=args.policy, seed=args.seed, progress=args.progress)
    dt = time.perf_counter() - t0
    print(f"Done in {dt:.2f}s", flush=True)

    # Print top K
    for r in rows[: args.top]:
        print(r)

    _write_csv(rows, args.out)
    print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()
