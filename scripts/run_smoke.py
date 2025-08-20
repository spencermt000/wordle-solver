from scripts.data_utils import load_answer_vocab
from scripts.sampler import WordSampler
from scripts.env import WordleEnv

vocab = load_answer_vocab("word_list.csv")
sampler = WordSampler(vocab, seed=123)
env = WordleEnv(vocab, sampler, allow_probe_guesses=False)

print("Vocab size:", len(vocab))

import random

# Run one random episode to verify the env loop
obs, mask = env.reset()
done = False
steps = 0
while not done:
    choices = [i for i, m in enumerate(mask) if m == 1]
    action = random.choice(choices)
    obs, reward, done, info, mask = env.step(action)
    steps += 1
    print(f"Step {steps}: guess={info['guess']} pattern={info['pattern']} remaining={info['remaining']} reward={reward:.2f}")

print(f"FINAL: solved={info['solved']} target={info['target']} steps={steps}")

# ---- Random baseline over many episodes ----
N = 500
solved = 0
steps_solved = []
for _ in range(N):
    obs, mask = env.reset()
    done = False
    steps = 0
    while not done:
        choices = [i for i, m in enumerate(mask) if m == 1]
        action = random.choice(choices)
        obs, reward, done, info, mask = env.step(action)
        steps += 1
    if info["solved"]:
        solved += 1
        steps_solved.append(steps)

rate = solved / N
avg_steps = sum(steps_solved) / len(steps_solved) if steps_solved else float('nan')
steps_solved.sort()
def pct(p):
    if not steps_solved:
        return float('nan')
    k = int(p * (len(steps_solved)-1))
    return steps_solved[k]


print("\n=== Random Agent Baseline ===")
print(f"Episodes: {N}")
print(f"Solve rate (<= {env.max_guesses}): {rate:.3f}")
print(f"Avg steps (solved only): {avg_steps:.2f}")
print(f"Median steps: {pct(0.5)} | 90th: {pct(0.9)} | 95th: {pct(0.95)}")

# ---- One heuristic episode: pick the allowed guess that maximizes expected (2*greens + 1*yellows) over current candidates ----
from scripts.feedback import score_pattern

def expected_step_reward(guess_idx: int, candidate_indices: list[int]) -> float:
    g = vocab.word_at(guess_idx)
    total = 0.0
    for t_idx in candidate_indices:
        t = vocab.word_at(t_idx)
        patt = score_pattern(g, t)
        greens = sum(1 for p in patt if p == 2)
        yellows = sum(1 for p in patt if p == 1)
        total += (2.0 * greens + 1.0 * yellows)
    return total / max(1, len(candidate_indices))

print("\n=== Heuristic Episode (expected 2*greens + yellows) ===")
obs, mask = env.reset()
done = False
steps = 0
while not done:
    candidate_indices = [i for i, m in enumerate(mask) if m == 1]
    # Score each allowed action by expected reward across current candidates
    best_idx = max(candidate_indices, key=lambda i: expected_step_reward(i, candidate_indices))
    obs, reward, done, info, mask = env.step(best_idx)
    steps += 1
    print(f"H{steps}: guess={info['guess']} pattern={info['pattern']} remaining={info['remaining']} reward={reward:.2f}")
print(f"HEURISTIC FINAL: solved={info['solved']} target={info['target']} steps={steps}")