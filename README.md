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
| 1 | trace-bootstrap → `pass^k` ± CI + decay curve | ✅ |
| 2 | judge calibration as a CI-blocking gate (κ + alt-test → `JUDGE_LICENSED`/`NEEDS_HUMAN`) | ✅ |
| 3 | cassette + deterministic replay (harness-logic re-execution) | ✅ |
| 4 | online layer + the unified offline↔online loop | ✅ |
| **5** | **model-vs-harness attribution (counterfactual adjudication)** | ✅ |

**The roadmap is complete: all five phases of the Phase 0 spec are built.**

## The judge gate (Phase 2)

Everyone *computes* judge-vs-human agreement (Ragas, Vertex AI); **nobody blocks on it**. KeenTouchstone does:

```bash
uv run touchstone judge demo          # keyless: exam two judges — one licensed, one blocked
uv run touchstone judge calibrate anchors.jsonl --judge-labels answers.jsonl
                                      # your judge takes the exam → license.json + 体检单 report
uv run touchstone judge gate out/judge/license.json     # CI: exit 1 blocks the build
uv run touchstone ingest traces.jsonl --outcomes-from verdicts.jsonl --license license.json
                                      # unlabeled traffic + LICENSED judge → pass^k (the loop)
```

- **The exam:** Cohen's κ vs human-confirmed labels (never raw agreement — a judge that always says "pass" gets 80% agreement on an 80%-pass set while discriminating nothing), with item-bootstrap CIs; TPR and **FPR (the rubber-stamp direction)** with Jeffreys CIs; abstentions excluded and counted, never coerced; the κ paradox surfaced on imbalanced sets.
- **The alt-test** (ACL 2025): with ≥3 human annotators, a formal statistical answer to "may this judge replace your annotators?" — leave-one-annotator-out, Benjamini–Yekutieli FDR correction, winning rate ω ≥ 0.5. Fewer than 3 annotators → honestly "not applicable", never computed anyway.
- **Anti-circularity, structural:** anchor labels must be `label_source: "human"` — auto-generated labels can expand an eval set but can never certify the judge that made them.
- **The gate is in the data path, not just CI:** model-graded verdicts without a matching `JUDGE_LICENSED` license are refused at ingest; licenses are not transferable between judges or prompts (the judging prompt is hashed — a new prompt is a new judge).

## The flight recorder (Phase 3)

*"I can re-run your Tuesday incident on Wednesday and get the same failure."* No shipping tool can deterministically replay an agent run; KeenTouchstone can — with the limits stated out loud.

```bash
uv run touchstone replay-demo      # keyless: record a buggy agent live → pass^k report
                                   # → replay the failed run → "REPLAY FAITHFUL — same
                                   #    failure (ValueError: '1,130.00'), zero network"
uv run touchstone replay out/replay-demo/cassettes/<run>.cassette.jsonl \
    --entry your_module:your_agent
```

Your agent routes its nondeterminism through an explicit seam — `io.llm_call(...)`, `io.tool_call(...)`, `io.now()`, `io.decision(...)`. Recording runs it for real and tapes everything (including the crash) to an append-only, schema-validated cassette, co-emitting Spans on the same `trace_id` — so recorded runs feed straight into `touchstone ingest` and the reliability stats. Replay re-executes **your orchestration code** against the frozen outputs: per-kind cursors, input matching (a changed harness diverges loudly at the exact step, with a diff), model/tool identity checks, exhaustion that **never** silently falls back to live systems, the clock served from tape, and leftover-events reporting.

**Honest limits, verbatim from the design thesis:** replay reproduces the harness orchestration logic given frozen model/tool outputs — **not model reproducibility** (hosted-API nondeterminism is unfixable from outside). Only calls routed through the seam are taped; SDK-level zero-code-change interception is future work.

## The online loop (Phase 4)

The thesis sentence, executable: *the same `pass^k` definition runs over an offline benchmark AND continuously over live traffic.*

```bash
uv run touchstone online-demo                    # keyless: the whole loop in one command
uv run touchstone ingest traffic.jsonl --signature-strategy template
                                                 # unlabeled traffic → DERIVED task identity
uv run touchstone watch traffic.jsonl --window 30 --slo 0.6@4 [--follow]
                                                 # tumbling windows; confident breach = exit 1
uv run touchstone compare baseline/aggregate.json candidate/aggregate.json --at-k 4
                                                 # paired sign-flip test on shared tasks
uv run touchstone slo-gate out/aggregate.json --slo 0.6@4     # release gate for CI
```

- **Task identity for organic traffic** — the keystone problem no shipping tool has cracked. v0.4 ships two deterministic, *inspectable* strategies: masked-input **templates** (a human reads exactly why two runs grouped) and **tool sequences**. Every derived grouping carries a quality readout (exemplars, singleton rate, purity vs declared tags when present) and a non-optional caveat: **derived grouping is a hypothesis, not ground truth.** Embedding clustering is deliberately deferred — it fails the spec's "stable" and "inspectable" requirements today.
- **watch** uses tumbling (never sliding) windows and two alert levels: BREACH only when the whole CI sits below the SLO; a straddling CI is a warning. The repeated-look inflation is noted in the output, not hidden.
- **compare** is a paired sign-flip permutation test on shared tasks (exact to 2^12, scipy-cross-checked) with a bootstrap CI on the mean delta — and an UNDERPOWERED flag that says "absence of evidence" out loud when the shared-task count is small.
- Still **ingest-don't-collect**: no daemon, no collector, no dashboards — read the files you already have.

## Attribution (Phase 5)

*"Your agent failed 8% of the time — 5 points were the model, 3 points were your retry logic."* No shipping tool can produce that sentence; KeenTouchstone **measures** it:

```bash
uv run touchstone attribute-demo    # keyless: one agent, two switchable fault sources,
                                    # four cells → the sentence, with CIs and significance
uv run touchstone attribute --baseline b.json --model-swap m.json \
    --harness-swap h.json [--both-swap x.json]
uv run touchstone diagnose out/replay-demo/cassettes/<failed>.cassette.jsonl
```

- **The trustworthy instrument is a measured A/B**: run the same task suite under baseline / model-swap / harness-swap (optionally both-swap); shares are paired per-task failure deltas with bootstrap CIs and sign-flip significance. With the fourth cell the 2×2 identity holds exactly (asserted at runtime) and the **interaction term** is estimated: positive = some failures need both fixes; negative = the shares double-count. Negative shares ("the better model made it worse") are findings, reported as such. The literature's "harness fixes buy 15–50%" number is single-team and unreplicated — this tool never cites it; it measures *your* data.
- **The low-trust hint is caged**: `diagnose` ranks ETCLOVG harness-layer hypotheses from a failed cassette with the academic ceiling (**step-level attribution SOTA ≈ 14%**, agent-level ≈ 53%) printed as the output header — a reading list for a human, never a verdict.

## Development

```bash
uv sync              # env + deps (Python ≥3.10)
uv run pytest        # tests (property-based stats tests included)
uv run ruff check .  # lint
```
