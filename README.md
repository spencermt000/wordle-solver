# Part 1: Starting Word Theory

## File-by-file
**Scripts**
- constraints: constraints and filtering, used for validation and conceptually
    - ConstraintState: tracks position allowances + min/max letter counts
    - apply_feedback: updates constraints based on the feedback
    - filter_candidates: uses score_pattern to keep the candidate pool of words consistent with past feedback/guesses
- feedback: the actual game logic/feedback for guesses 
    - score_pattern:
    - pattern_to_int:
    - consistent_ith:
- sampler: modular utility to sample targets deterministically 
    - WordSampler: 
- vocab: loads and holds the word list sourced from the github link mentioned elsewhere 
    - WordVocab:

- env: customer Wordle RL environment ()
- gym_env:

- run_smoke:
- data_utils:
- init: Allows the scripts to be treated as a module 

**Starting Word**
- eval: first-guess theory
    analytic scoring of a single first guess against all answers (no second move simulated)
## logic 
- For each guess, bucket all answers by feedback pattern; compute:
	•	exp_remaining = sum(count^2)/N (lower is better),
	•	entropy (higher is better),
	•	worst_case (lower is better),
	•	partitions = distinct patterns.
	•	Prints top-K and writes starting_word_results.csv
- eval_all_words.py
    same logic as above, only uses the entire CSV dataset for targets/guesses, ignoring previous actual wordle answers 

- eval_whole_game_sim: full game monte carlo evaluator for starting words
•	Targets from --targets {answers|all}.
	•	Guess set from --guesses {answers|all}.
	•	First guesses from --first or --first-file starting_word/candidate_words.txt.
	•	Policy after first move: heuristic (greedy expected info) or random.
	•	Outputs solve rate, avg steps (solved), median, p90/p95; saves CSV (you can point to starting_word/outputs/...).

**Tests**
- test_gym_env.py
    quick smoke test of the Gym wrapper 
- test_pruning.py
    verifies pruning behavior and functionality matches Wordle rules using known pairs
- test_rewards.py
    checks reward shaping for partial matches and exact solve

How this ties into your “Starting Word Theory”
	•	Theory metrics (entropy / expected remaining / worst-case / partitions) explain why certain openers (e.g., raise/irate/slate) tend to be strong—they split the answer set into many, well-balanced buckets on move 1.
	•	Monte Carlo whole-game sims validate theory in practice: run the environment end-to-end to see solve rate and steps when you actually play the game with a greedy policy after the opener.
	•	Your earlier results: analytics said raise/irate/… are top; whole-game sims favored slate on average steps—great example that “best first split” and “best full-game outcome” can differ slightly depending on second-move strategy.