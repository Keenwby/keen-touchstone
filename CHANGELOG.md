# Changelog

## [Unreleased] — 0.1.0 (Phase 1 MLP, built 2026-07-02)

**The MLP contract, delivered:** traces in → `pass^k` ± CI + reliability decay curve out, no golden dataset authored.

- **Stats core** (`stats/`): exact `pass^k`/`pass@k` estimators (τ-bench UMVUE, parity-tested against Inspect AI's reducers on real logs), suite decay curve with k ≤ min(nᵢ) (no composition drift), cluster bootstrap over tasks with shared resample indices (coherent band), Jeffreys Beta-Binomial per-task CIs (exact x↦xᵏ endpoint transform), Wilson, variance decomposition with a "which lever" readout.
- **Statistical honesty guards** found by building: headline k defaults to ⌈k_max/2⌉ (the per-task estimator degenerates to {0,1} at k=n); zero-width bootstrap intervals are widened with Jeffreys-posterior draws and loudly warned (a zero-width CI from resampling is a false claim of certainty); unresolvable outcomes are excluded and counted, never treated as failures.
- **Inspect adapter**: `.eval`/`.json` logs via the public `inspect_ai.log` API only; success semantics identical to Inspect's reducers; multi-log merge; strict one-(task, model) rule.
- **OTel adapter**: `gen_ai.*` span JSONL → runs by `trace_id`; declared-tag task signatures behind the SPEC §5 `TaskSignatureStrategy` interface; outcome priority `harness.outcome` → `gen_ai.evaluation.score.*` → `--outcome-regex`.
- **Demo, keyless end to end**: `touchstone demo` (mockllm + deterministic seeded flaky scorer) and `touchstone ingest --demo` (synthetic gen_ai.* traces with known true pᵢ, used as a truth-recovery test harness).
- **Report**: schema-validated `aggregate.json` + self-contained theme-aware `report.html` (SVG decay chart, CI band, pass@k-vs-pass^k contrast, hover tooltips, per-task and curve tables, honesty footer).

- Repo bootstrapped from the Phase 0 spec (2026-07-02). Spec snapshot in `docs/spec/`, canonical JSON Schemas packaged at `src/keen_touchstone/schemas/`.
- **Schema change vs Phase 0** (`reliability-aggregate.schema.json`): added nullable `headline_k` — the Phase 0 draft stored a headline `pass_hat_k` without recording which k it referred to, which implementation proved ambiguous. Suite-level rollups use the reserved `task_key` `"__suite__"` (documented; a dedicated SuiteAggregate shape is a candidate for spec v0.2).
