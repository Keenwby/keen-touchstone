# Case study: GSM8K on a local 7B model — what pass@1 hides

**The first non-synthetic data this tool ever ate.** Every demo in the repo uses designed
fault injection with known ground truth (so the estimators can be truth-recovery-tested).
This run is the opposite: a real model, real sampling randomness, a public benchmark,
and an Inspect log the tool's authors did not construct.

## Setup

| | |
|---|---|
| Task set | `inspect_evals/gsm8k` (grade-school math, exact-answer scoring), first 20 problems |
| Trials | 10 epochs per problem → 200 rollouts |
| Model | `qwen2.5:7b` on local Ollama (Apple M4 Pro) — zero API cost |
| Wall clock | ~26 minutes, 505k tokens |

```bash
OLLAMA_BASE_URL=http://localhost:11434/v1 OLLAMA_API_KEY=ollama \
uv run --with inspect-evals --with datasets --with openai \
  inspect eval inspect_evals/gsm8k --model openai-api/ollama/qwen2.5:7b \
  --limit 20 --epochs 10 --log-dir logs/

uv run touchstone analyze logs/*.eval --out out/gsm8k
```

## Headline

```
pass@1 83.0%  →  pass^5 58.1% [39.7%, 74.5%] 95% CI  →  pass^10 40.0% [20.0%, 60.0%]
20 tasks × 10 trials (n=200 rollouts)
```

The model is "83% accurate" by every leaderboard convention. Asked to solve the *same*
problem ten times in a row — the deployment question — it succeeds 40% of the time.

The full decay curve (suite mean, cluster-bootstrap + posterior-envelope CI):

| k | pass^k | 95% CI |
|---|--------|--------|
| 1 | 83.0% | [67.7%, 93.0%] |
| 2 | 75.0% | [57.4%, 87.3%] |
| 3 | 68.6% | [49.8%, 82.5%] |
| 5 | 58.1% | [39.7%, 74.5%] |
| 7 | 49.7% | [30.5%, 68.2%] |
| 10 | 40.0% | [20.0%, 60.0%] |

## What the averages hide

Per-task reliability is **bimodal**, not uniform-ish around 83%:

- **8 of 20 problems: 10/10** — genuinely solid.
- **One problem: 0/10.** The model *never* solves it. No amount of retrying helps; a
  pass@1 average will never tell you this task exists.
- **One problem: 2/10**, five more at 9/10 — the flaky middle that k-repetition exposes.

This heterogeneity is why the report's variance-decomposition "lever" note reads
*more tasks beats more reruns* for this suite — the between-task spread dominates the
within-task sampling noise.

## What this run verified about the tool

1. **The adapter consumed a foreign `.eval` log first try** — task grouping by sample id,
   token accounting, scorer resolution — zero errors, zero patches.
2. **The honesty features fired where they should**: `power_status:
   UNDERPOWERED_NEED_MORE_N` on the headline (20 tasks is thin — and the CI says so),
   wide per-task CIs at n=10 rather than suppressed error bars, a monotone CI band.
3. **The thesis sentence survives contact with reality**: the pass@1 → pass^k collapse is
   not an artifact of the synthetic demos. It is what real model randomness does.

Artifacts (raw Inspect log, `aggregate.json`, `report.html`) were produced by the exact
commands above; regenerate with any local model that fails occasionally — a model that
aces the suite produces a flat, uninformative curve.
