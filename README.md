# KeenTouchstone

**Agentic eval & reliability harness.** Traces in → `pass^k` ± confidence intervals + a reliability decay curve out — **no golden dataset authored**.

> A touchstone is the stone gold is rubbed against to test its purity. KeenTouchstone tests whether an agent is *genuinely reliable*, not just shiny at pass@1.

`pass@1` answers "can it work?". Deployment asks "does it work **every time**?" — that is `pass^k` = P(all k attempts succeed), and it collapses fast: a 70%-per-trial agent has pass@3 ≈ 97% but pass^3 ≈ 34%. Almost all public numbers are pass@1-shaped, overstating real reliability by 2–3×. KeenTouchstone makes the distributional question — `pass^k`, CIs, decay curves — a first-class, continuously computable primitive over the runs you already have.

## Status

**v0.1 (Phase 1 MLP) — built, local pre-release.** The Phase 0 spec (data model + schemas) lives in [`docs/spec/`](./docs/spec/) and [`src/keen_touchstone/schemas/`](./src/keen_touchstone/schemas/). License: MIT. Not yet published to PyPI/GitHub.

## Quickstart (zero API keys)

```bash
uv sync
uv run touchstone demo            # simulated flaky agent → Inspect runner (mockllm) → stats → report
uv run touchstone ingest --demo   # synthetic OTel gen_ai.* traces → the same stats, context=online
open out/demo/report.html
```

The demo's terminal summary, verbatim — this is the story the tool exists to tell:

```
pass@1 80.6%  →  pass^6 37.4% [12.6%, 66.8%] 95% CI, bootstrap
6 tasks × 12 trials (n=72 rollouts)
```

An agent that looks 80% reliable at pass@1 clears six-in-a-row barely a third of the time — and the report says so with confidence intervals, a decay curve, per-task Beta-Binomial CIs, cost columns, and a variance-decomposition "lever" note telling you whether more reruns or more tasks buys the most certainty. For your own agent:

```bash
uv run touchstone run your_task.py --model anthropic/claude-sonnet-5 --epochs 10
uv run touchstone analyze ~/path/to/inspect/logs/        # logs you already have
uv run touchstone ingest your_traces.jsonl               # OTel gen_ai.* spans (JSONL)
```

## Prior art — what we concede up front

- **[Inspect AI](https://inspect.aisi.org.uk/) (UK AISI, MIT) already owns offline reliability science**: `Epochs(count, reducers)` with `pass_at_{k}` / `pass_k_{k}` / `at_least_{k}` reducers, clustered `stderr`, `bootstrap_stderr`, hermetic sandboxing, 20+ providers. **We build ON Inspect — it is our offline run substrate, not our competitor.**
- What Inspect does *not* do (verified against v0.3.244): **CIs on the `pass^k` estimate itself**, the **k=1..N decay curve**, reliability over **live traffic / raw OTel traces**, judge-calibration gating, deterministic replay, model-vs-harness attribution. That's the surface KeenTouchstone adds, phase by phase.
- Full tool-by-tool audit: [`docs/spec/PRIOR-ART.md`](./docs/spec/PRIOR-ART.md).

## What v0.1 does

One stats core, two ingestion paths, one honest report:

```bash
touchstone demo                       # bundled flaky-agent task via Inspect (no API keys needed)
                                      #   → .eval log → pass^k ± CI + decay curve → report.html + aggregate.json
touchstone analyze <logdir>           # your existing Inspect .eval logs        (context: offline)
touchstone run <task.py> --epochs 10  # wrap Inspect's runner, then analyze
touchstone ingest <traces.jsonl>      # OTel gen_ai.* spans from production     (context: online)
                                      #   ← the headline: your traces ARE the dataset
```

The output is a `ReliabilityAggregate` (see [`SPEC.md`](./docs/spec/SPEC.md) §4): per-task and suite-level `pass^k` with bootstrap CIs, the k=1..N decay curve with a CI band, cost columns, and a variance decomposition that tells you *which lever to pull* (more rollouts per task vs more tasks).

### The stats contract

- Per task (n trials, c successes): `pass^k` estimator = **C(c,k)/C(n,k)** — the τ-bench formula, unbiased for pᵏ; identical to Inspect's `pass_k_{k}` reducer (guarded by a parity test). `pass@k = 1 − C(n−c,k)/C(n,k)`.
- Suite aggregate = mean of per-task estimates; decay curve runs k = 1..min(nᵢ) by default so the task set is constant across k (no composition drift).
- **CI on the aggregate = cluster bootstrap resampling tasks** — never pooled attempts (pooling understates variance). Per-task CI = Jeffreys Beta posterior with endpoints transformed x↦xᵏ. Small samples get loud warnings, not suppressed error bars: *a wide interval is the honest product.*

## Roadmap

| Phase | Deliverable | Status |
|---|---|---|
| 0 | 4-artifact open spec (`Span`/`Cassette`/`EvalVerdict`/`ReliabilityAggregate` on one `trace_id`) | ✅ `docs/spec/` |
| **1** | **trace-bootstrap → `pass^k` ± CI + decay curve (this MLP)** | 🚧 |
| 2 | judge calibration as a CI-blocking gate (κ + alt-test → `JUDGE_LICENSED`/`NEEDS_HUMAN`) | — |
| 3 | cassette + deterministic replay (harness-logic re-execution) | — |
| 4 | online layer + the unified offline↔online loop | — |
| 5 | model-vs-harness attribution (counterfactual adjudication) | — |

## Development

```bash
uv sync              # env + deps (Python ≥3.10)
uv run pytest        # tests (property-based stats tests included)
uv run ruff check .  # lint
```
