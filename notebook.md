
https://github.com/steve-kasica/wordle-words/blob/master/wordle.csv
- source of words 
### Options for RL "flavor"
1. Q-learning with function approximation (DQN-style): state → Q(a). Good when you can action-mask.
2. Policy gradient (PPO/A2C) with action masking: outputs a softmax over words; mask invalid ones.
- DQN allows for action masking

1) environment contract (write this first)

Create a minimal class with:
	•	reset() → returns obs and action_mask
	•	step(action_index) → returns (obs, reward, done, info, action_mask)

Define the spaces
	•	Actions: index into your CSV’s word list (size = N).
	•	Observation (state): do NOT use raw strings. Encode constraints from previous feedback:
	•	5×26 positional letter feasibility (1 if letter can be in that slot; 0 otherwise).
	•	26 global must-include counts (lower bounds) and 26 max counts (upper bounds), updated after each feedback.
	•	one scalar: remaining_candidates_count / N (or log count).
	•	step index (0–5) one-hot or scalar.
	•	(optional) previous guess pattern as 5 trits one-hot (green/yellow/gray).
	•	Action mask: 1 for actions (words) consistent with current constraints; 0 otherwise. (You’ll get this by filtering the candidate set—see §3.)

Unit tests you should write
	•	Reset with a fixed seed returns consistent obs.
	•	After a known guess/feedback, action_mask zeroes out inconsistent words you can hand-check.

⸻

2) feedback function (the Wordle oracle)

You need a pure function:
	•	Input: (guess, target)
	•	Output: 5-length list with {0=gray, 1=yellow, 2=green}, using true Wordle duplicate rules:
	1.	Mark all greens; decrement availability counts.
	2.	For remaining positions, mark yellow only if the letter’s remaining count > 0; decrement.

Tests
	•	ALLOT vs TOTAL must produce the canonical tricky duplicates outcome.
	•	A few hand-constructed cases with repeated letters (e.g., BANAL vs CANAL).

⸻

3) constraint updater & candidate filter

Two pieces:

A. Constraint state updater
From cumulative history (guess_t, pattern_t) update:
	•	Per-position feasible letters (greens fix a slot; grays remove letter from slot; yellows remove letter from that slot).
	•	Global min/max letter counts (e.g., gray can mean max=0 unless some yellows/greens already force ≥1).

B. Candidate filter
Given constraints, filter your full list to those consistent with all past feedback.
This filtered set builds action_mask (1 if a word is in the filtered set; else 0).

Tests
	•	After one guess/pattern, your filtered set should match recomputing feedback word-by-word (use the oracle to verify).

⸻

4) reward shaping (your idea, tuned)

Your proposal (reward right letters and right spots) is workable, but protect against weird incentives:
	•	Per step reward:
r_t = α * (#greens) + β * (#yellows) - 1
with α > β > 0, e.g., α=2, β=1, and the -1 is a time penalty.
	•	Terminal bonus: if solved, add +S (e.g., S=10) on that step.
	•	Fail at step 6: zero bonus; episode ends.

Tips:
	•	If you see the agent exploiting yellows (stalling), increase α and the terminal bonus, or reduce β.
	•	Keep rewards clipped to a small range for stability (e.g., clip to [-2, +12]).

⸻

5) state featurization (for your NN)

Concatenate:
	•	Positional feasibility: 5×26 binary.
	•	Global min/max counts: 26 + 26 (could scale to [0,1] by /5).
	•	Step: scalar /6 or one-hot length 6.
	•	Log remaining candidates: scalar (normalize).
	•	(Optional) last pattern one-hot: 5×3.

Normalize/standardize scalars (e.g., z-score or min-max). Keep it small and fixed-size.

⸻

6) agent and training loop

If you do DQN-style:
	•	Network: MLP on the state (2–3 hidden layers). Output: Q over all N actions.
	•	Action masking: set Q of masked actions to a very negative number before argmax; during loss, zero out gradients for masked actions (or just never select them and compute loss only on the chosen action target).
	•	Target network + replay buffer: standard DQN bits.
	•	Exploration: ε-greedy with linear decay; sample targets uniformly from your word list each episode (so the environment’s “initial state” varies).
	•	Loss: TD loss on (s, a, r, s'), done handling.

If you do policy gradient (e.g., PPO):
	•	Output logits over actions; apply mask by adding -∞ to masked logits.
	•	Optimize clipped policy loss + value loss + entropy bonus (small entropy; mask already gives structure).

⸻

7) training curriculum (helps a ton)
	•	Phase 1: Limit the action space to a 1–2k curated answer list (Wordle’s official answers if you have them).
	•	Phase 2: Add the full allowed guess list as actions, but keep masking to candidates (or allow probes by relaxing mask to “allowed guesses” and use a small penalty if guess isn’t in current candidate set).
	•	Phase 3: Harder targets (words with duplicate letters, rare letters).

⸻

8) evaluation protocol

After every N training updates:
	•	Run greedy (ε=0) evaluation on:
	•	a fixed 200-word dev set (stratified: common/rare/dup letters)
	•	random 500 words
	•	Report: average guesses, % solved ≤6, distribution (min/median/95th).

Track also:
	•	average remaining candidate count after guess 1 and 2 (should drop fast if learning).

⸻

9) common pitfalls (avoid these)
	•	Noisy constraints: if your duplicate-letter logic is off, learning collapses. Unit test it thoroughly.
	•	Mask leakage: don’t let the agent “see” the target via a mask that only leaves 1 action immediately (unless that’s logically correct).
	•	Reward trap: too much yellow reward → the agent farms yellows. Adjust α/β and increase terminal bonus.
	•	Action space too big too soon: start smaller, then scale.

⸻

10) build order checklist (what to implement, in order)
	1.	CSV loader → list of words; a word_to_idx dict.
	2.	Feedback oracle (with duplicate rules) + tests.
	3.	Constraint updater + candidate filter + tests.
	4.	Environment skeleton (reset, step) + action mask logic + tests.
	5.	State featurization function + normalization.
	6.	Pick algorithm (DQN or PPO) and implement the minimal training loop.
	7.	Add logging of metrics from §8.
	8.	Tune rewards α, β, S.
	9.	Scale up the wordlist and curriculum.