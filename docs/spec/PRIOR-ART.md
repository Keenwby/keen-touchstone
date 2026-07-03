# Prior art & the verified moat (KeenTouchstone)

> Honest competitive analysis. The differentiation rests on three **absence-of-evidence** claims; each was adversarially stress-tested on 2026-06-23 by a skeptic agent prompted to *refute* it against ~18–20 shipping tools' current docs/source. All three HOLD. Confidence: **HIGH** that no clearly-documented shipping feature exists; **MEDIUM** on absolute "none exists anywhere" (gated dashboards / unannounced betas can't be fully excluded).

## Build on, don't beat

| Project | License | What it owns (we reuse it) | What it lacks (our surface) |
|---|---|---|---|
| **Inspect AI** (UK AISI) | MIT | offline `pass_k`/`Epochs`/reducers, clustered+bootstrap SE, hermetic sandbox, 20+ providers, MCP tools, transcript viewer | online/live, replay, judge-calibration gate, model-vs-harness attribution, CI on the `pass^k` estimate, decay curve |
| **HAL** (Princeton) | **none (all-rights-reserved)** | reliability-at-scale, cost-control, automated log inspection (as a *study*) | **unusable as a build base (no license)**; offline-only |
| **DeepEval / Ragas** | Apache-2.0 | metric/scorer primitives; Ragas auto-computes κ | no reliability-over-traffic, no gate, no replay, no attribution |
| **OTel GenAI** | spec | the `gen_ai.*` span vocabulary | "captures what happened; does not assess whether it was good" |

**Strategy:** wrap Inspect as the offline run substrate + sandbox + provider layer; wrap DeepEval/Ragas scorers; ride OTel at the wire. Net-new = normalization layer, cassette tap, replay engine, judge-calibration *gate*, reliability-stats layer (online + decay + CI-on-estimate), attribution layer, CI gate.

---

## Wall 1 — `pass^k` / reliability variance over LIVE traffic → **HOLDS** (HIGH)

~18 tools checked against 2026 docs. Every "reliability"-adjacent capability resolves to either (a) per-trace LLM-judge scoring, or (b) simple pass-rate aggregation over independent traces. **No tool groups repeated *production* runs of the same task into a `pass^k` / repeated-trial-variance distribution online.** The repeated-trial machinery, where it exists, is exclusively offline/batch.

**Closest competitors (and where each stops):**
- **AWS Bedrock AgentCore Evaluations** — *literally documents* repeated-trial variance ("run ≥10 trials per question… variance reveals consistency") — but scoped to a **dev/batch workflow on curated questions**, separate from the online lane.
- **Maxim AI** — re-runs the same scenario across runs/personas and reports consistency — but **"Pre-Production / before deployment"**; in-prod path is plain per-trace scoring.
- **Braintrust** — `trialCount` buckets by identical input offline; online scoring is strictly per-trace.
- **Patronus** — "Pass@K" in vocabulary, offline RL-tutorial only.
- **Fiddler** — CIs on aggregate metric *drift*, not per-task `pass^k`.

**The keystone insight:** the hard part is **task identity** for organic traffic (SPEC §5). Bedrock/Maxim/Braintrust all sidestep it by using *curated* datasets where task identity is given.

## Wall 2 — judge-vs-human calibration as a CI-blocking gate → **HOLDS, narrowed** (HIGH)

~20 tools checked. **Correction to the original thesis:** "nobody computes κ" is **false**. The accurate claim: tools that compute κ never **gate** on it.

- **Ragas `validate_alignment()`** — reads as: calls `sklearn.metrics.cohen_kappa_score` vs a human-labeled split, returns `{correlation, agreement_rate, df}` — **zero threshold / assert / CI hook.** Computes the exact metric; deliberately informational.
- **Vertex AI AutoSxS** — auto-emits Cohen's κ + confusion matrix + accuracy/precision/recall vs human preference — **report-only, no gate.**
- **LangSmith Align Evals** — tracks a baseline *% agreement* (not κ), in an evaluator-playground loop **explicitly decoupled from CI.**
- **Braintrust / DeepEval / Patronus / Galileo** — all **manual** "compare scores with human spot-checks, recalibrate on drift," or unidirectional auto-*improvement* loops. Every CI gate that exists gates on the judge's **score of the app output** ("faithfulness > 0.8 or fail"), never on the judge's **agreement with humans**.

**The unowned move:** auto-compute κ/TPR-FPR vs a human-confirmed anchor set **AND** wire it into a promotion-blocking gate (`JUDGE_LICENSED`/`NEEDS_HUMAN`). The *gate* is the differentiator, not the metric.

## Wall 3 — model-vs-harness failure attribution → **HOLDS** (HIGH)

Shipping tools do symptom/trace-level RCA; none decomposes failure into **model-caused vs harness/scaffold-caused** as a first-class axis. An Arize-authored 2026 roundup of 8 tools concedes the gap: *"the missing link is rarely model quality or the orchestration framework — the missing link is visibility."*

- **LangSmith Engine** (closest *productized* RCA) — detect→diagnose→propose-fix→deploy loop; generates code/prompt changes but **no structured model-vs-scaffold classification.**
- **DeepEval** — you can A/B-swap model/prompt/tools across experiments and diff — but **manual variable isolation**, not a built-in attribution feature. *(DeepEval's own docs say: "to determine model vs scaffold failures you'd conduct separate A/B tests… rather than use built-in diagnostic attribution.")*
- **Galileo** — failure-*mode* clustering, not model-vs-harness.
- **HarnessFix/ETCLOVG** (arXiv 2606.06324) — closest *conceptually* (7-layer harness taxonomy + auto-repair) but **research-stage** and *presupposes* harness flaws (harness-repair, not adjudication). **AgenTracer/Aegis/Who&When** attribute to **step/agent**, not model-vs-harness.

**The unowned move:** **counterfactual adjudication** — auto model-swap and harness-ablation on the *same* failed task → a *measured* attribution (SPEC §6). The A/B tooling exists but is manual; the research hasn't been fused into a productized model-vs-harness feature.

---

## Honesty ledger

- Verdicts rest on vendor docs / GitHub source / blogs; some product dashboards are gated and some doc pages were JS-rendered or 404'd (corroborated via alternate surfaces). A privately-shipped, undocumented beta could falsify a wall — re-verify tool-by-tool with dated screenshots before any *public* "no tool does X" claim.
- One WebSearch hallucinated a non-existent Ragas "κ > 0.6" threshold; the source-read caught and discounted it — the "no gate" finding is not contaminated by that error.
- The Arize roundup (Wall 3) is a competitor's framing (mild bias) but corroborates independent per-tool fetches.
