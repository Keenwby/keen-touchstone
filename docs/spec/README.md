> **📌 Snapshot note (2026-07-02):** this directory is the frozen Phase 0 spec text, lifted verbatim from the research workspace. The canonical, packaged JSON Schemas now live at [`../../src/keen_touchstone/schemas/`](../../src/keen_touchstone/schemas/); relative `./schemas/` links below refer to the original Phase 0 layout.
# KeenTouchstone — an open spec for agentic eval & reliability

> **Name: KeenTouchstone** — package handle `keen-touchstone`. *A touchstone is the stone gold is rubbed against to test its purity; this tests whether an agent is genuinely reliable, not just shiny at pass@1. The `keen-` prefix is deliberate — bare `touchstone`, `touchstone-eval`, and the whole `assay*` family are already taken by adjacent LLM-eval tools. Working name; re-confirm the public handle at publish time. (Avoid "Veritas" from the design thesis — Veritas Technologies trademark.)*
>
> **Status: Phase 0 — RFC / spec-first.** This repo is a *specification*, not yet an implementation. The code, when it comes, will mostly **wrap existing OSS** (UK AISI Inspect AI, OpenTelemetry, DeepEval/Ragas). Publishing the schema *is* the first deliverable.
>
> **Date:** 2026-06-23 · **License (intended):** MIT or Apache-2.0 (truly OSS — not ELv2, not open-core).

---

## What this is

Observability tells you **what your agent did**. Eval libraries tell you **if one run was good**. Offline eval frameworks (Inspect AI) can already tell you **how reliable an agent is on a fixed dataset**.

**Nobody closes the loop.** KeenTouchstone specifies a data model where the *same* reliability metric — `pass^k` with confidence intervals — runs identically over an **offline benchmark** and over **live production traffic**; where every LLM-based grader ships its **own calibration evidence** (Cohen's κ vs human labels) so you can prove the *judge* works before you trust the *judgment*; and where a failed run can be **replayed deterministically** and **adjudicated** as model-fault vs harness-fault.

It is the **reliability layer over existing instrumentation**, and its product is **evidence that survives deployment**, not a score.

> The full reasoning, competitive analysis, and design rationale live in the design thesis: `../research/deep-research/agentic-eval-reliability-harness-thesis.md`.

## Prior art — what we concede up front

Conceding precisely is more credible than a moat that collapses on one `pip install`. So, before you find it yourself:

- **UK AISI [Inspect AI](https://inspect.aisi.org.uk/) already owns offline reliability science.** It ships a first-class `pass_k` / `pass_at` reducer, `Epochs(count, reducers)`, clustered `stderr` and `bootstrap_stderr`, hermetic per-sample sandboxing, and a 20+ provider model layer. **We build ON Inspect, we do not reinvent it.** What Inspect does *not* do: run over **live production traffic**, **deterministic replay**, **judge-calibration as a gate**, **model-vs-harness attribution**, CIs *on the `pass^k` estimate itself*, or a `pass^k` *decay curve* (k=1..N).
- **[Ragas](https://github.com/explodinggradients/ragas) `validate_alignment()` already auto-computes Cohen's κ** (via `sklearn.metrics.cohen_kappa_score`), and **Vertex AI AutoSxS** emits κ + a confusion matrix. So "nobody computes κ" is **false**. What's unowned: **wiring κ into a tracked, promotion-blocking gate**. Every tool stops at "report for human inspection."
- **[HAL](https://hal.cs.princeton.edu/) (Princeton)** does reliability-at-scale + cost + automated log inspection — but it's **offline-batch-only** and ships **no license** (all-rights-reserved → unusable as a build base).
- We **wrap** DeepEval/Ragas scorers, **ride** OpenTelemetry GenAI semantic conventions, and **reuse** Inspect's runner. The value is the *judgment + replay + calibration + attribution* model the standards omit — not a new span format.

## The three genuinely-unowned moves (independently stress-tested)

Each was adversarially verified (2026-06-23) against ~18–20 shipping tools by a skeptic prompted to *break* the claim. All three survived. See [`PRIOR-ART.md`](./PRIOR-ART.md) for the tool-by-tool audit.

1. **`pass^k` / reliability distributions over LIVE traffic.** All repeated-trial machinery is offline. The hard part isn't the math — it's **task identity**: grouping *organic* production runs into "the same task" to aggregate. *(Closest: Bedrock AgentCore, Maxim, Braintrust — all offline.)*
2. **Judge-vs-human calibration as a CI-blocking gate.** Ragas/Vertex compute κ; **none gates on it.** *(The differentiator is the gate, not the metric.)*
3. **Model-vs-harness failure attribution** via **counterfactual adjudication** — auto model/harness swap on the failed trace → a *measured* "X% model / Y% harness" verdict. No shipping tool does this; existing A/B-swap is manual. *(Closest: DeepEval manual swap, LangSmith Engine RCA, HarnessFix research.)*

## What this is NOT

- **Not** another tracer (we ride OTel/Langfuse).
- **Not** another metric library (we wrap DeepEval/Ragas).
- **Not** another LLM-judge wrapper (the named anti-pattern).
- **Not** an authoritative step-level failure attributor — SOTA is ~14% accurate ([Who&When](https://arxiv.org/pdf/2505.00212)); we ship attribution only as a **confidence-banded hypothesis** or a **measured A/B**, never a verdict.
- **Not** a full ADLC platform (the Cisco/Galileo lane a solo builder loses).

## The data model (4 artifacts, one `trace_id`)

KeenTouchstone is a **4-artifact data model keyed on a shared `trace_id`**, layered on OTel spans. The differentiation lives in the three artifacts the standards don't cover and in the **join keys** across the offline / online / replay / attribution boundary.

| Artifact | Role | Standard? | Schema |
|---|---|---|---|
| **`Span`** | the substrate (a step in an agent run) | OTel `gen_ai.*` + net-new fields | [`schemas/span.schema.json`](./schemas/span.schema.json) |
| **`Cassette`** | append-only record of a run → enables deterministic replay | net-new | [`schemas/cassette.schema.json`](./schemas/cassette.schema.json) |
| **`EvalVerdict`** | a grader's judgment on a run, with provenance + calibration link | net-new | [`schemas/eval-verdict.schema.json`](./schemas/eval-verdict.schema.json) |
| **`ReliabilityAggregate`** | N runs of one task → `pass^k` + CIs + decay + attribution | net-new | [`schemas/reliability-aggregate.schema.json`](./schemas/reliability-aggregate.schema.json) |

Full field-level spec: [`SPEC.md`](./SPEC.md).

## Architecture in one line

*Coherent at the data-model layer, split at the storage-engine layer* — **one tool, two stores**: unify the 4-artifact schema, `trace_id` join, cassette format, scorer/verdict/reliability libraries, and CLI; internally split a hot append-only ingest path (observability SLA) from a columnar analytics/replay store (eval SLA). That separation is what makes a *unified* harness buildable by one person.

## Roadmap (Phase 0 → 5)

| Phase | Deliverable |
|---|---|
| **0 (this repo)** | the 4-artifact open spec + JSON schemas + prior-art concession |
| 1 | trace-bootstrap onboarding → `pass^k` ± CI + decay curve over your own agent (no golden dataset) — the *Minimum Lovable Product* |
| 2 | judge calibration as a gate (`judge.calibration.jsonl` → κ + alt-test → `JUDGE_LICENSED`/`NEEDS_HUMAN`, CI-blocking) |
| 3 | cassette + deterministic replay (harness-logic re-execution) |
| 4 | online layer + the unified offline↔online loop |
| 5 | model-vs-harness attribution (counterfactual adjudication + measured A/B) |

Full roadmap with timings: thesis §11.

## Open questions

- **Name.** Working name **KeenTouchstone** / `keen-touchstone`. Bare `touchstone`, `touchstone-eval`, and `assay*` are taken by adjacent tools; the `keen-` prefix gives a clean handle. Re-confirm PyPI/GitHub availability at publish time.
- **Task identity for organic traffic** (the online-`pass^k` keystone): how to derive a stable `task_signature` from a production trace so repeated runs group correctly. See `SPEC.md` §"Task identity".
- **License:** MIT vs Apache-2.0 (patent grant).
- **`$id` namespace** for the JSON schemas (depends on the eventual repo/domain).

## Status & contributing

Phase 0 RFC. Issues/PRs against the spec and schemas welcome. Nothing here is load-bearing code yet — it's a **design contract**, published so it can be argued with before it's built.
