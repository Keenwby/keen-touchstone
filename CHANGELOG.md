# Changelog

## [Unreleased] — 0.1.0 (Phase 1 MLP, built 2026-07-02)

**The MLP contract, delivered:** traces in → `pass^k` ± CI + reliability decay curve out, no golden dataset authored.

- **Stats core** (`stats/`): exact `pass^k`/`pass@k` estimators (τ-bench UMVUE, parity-tested against Inspect AI's reducers on real logs), suite decay curve with k ≤ min(nᵢ) (no composition drift), cluster bootstrap over tasks with shared resample indices (coherent band), Jeffreys Beta-Binomial per-task CIs (exact x↦xᵏ endpoint transform), Wilson, variance decomposition with a "which lever" readout.
- **Statistical honesty guards** found by building: headline k defaults to ⌈k_max/2⌉ (the per-task estimator degenerates to {0,1} at k=n); zero-width bootstrap intervals are widened with Jeffreys-posterior draws and loudly warned (a zero-width CI from resampling is a false claim of certainty); unresolvable outcomes are excluded and counted, never treated as failures.
- **Inspect adapter**: `.eval`/`.json` logs via the public `inspect_ai.log` API only; success semantics identical to Inspect's reducers; multi-log merge; strict one-(task, model) rule.
- **OTel adapter**: `gen_ai.*` span JSONL → runs by `trace_id`; declared-tag task signatures behind the SPEC §5 `TaskSignatureStrategy` interface; outcome priority `harness.outcome` → `gen_ai.evaluation.score.*` → `--outcome-regex`.
- **Demo, keyless end to end**: `touchstone demo` (mockllm + deterministic seeded flaky scorer) and `touchstone ingest --demo` (synthetic gen_ai.* traces with known true pᵢ, used as a truth-recovery test harness).
- **Report**: schema-validated `aggregate.json` + self-contained theme-aware `report.html` (SVG decay chart, CI band, pass@k-vs-pass^k contrast, hover tooltips, per-task and curve tables, honesty footer).

### Adversarial review round 1 (2026-07-02, two independent skeptic agents; all findings reproduced, tagged `[review …]`)

- `[review S1, Major]` **Decay-band monotonicity**: widening only degenerate k's mixed frequentist and Bayesian bounds — the upper band could *rise* at deep k (impossible: P(all k+1) > P(all k)). Fixed: suite CI = element-wise **envelope of the task bootstrap and a Jeffreys-posterior band on the same resamples** at every k (`ci_method: bootstrap_posterior_envelope`, schema-documented); monotone by construction, never narrower than either component. Regression: band-monotonicity property test over 31 suites incl. the reviewer's.
- `[review P1, Major]` **Dict/list `Score.value` silently flattened to 0.0** — a 100%-reliable agent read as 0% reliable. Fixed: hard error naming the keys; `--score-key` selects the success metric (dict), lists rejected. Regression tests cover error, decoy key, and parity with the scalar path.
- `[review P2, Major]` **Same-name/same-model logs with different `task_args` silently merged** into one pass^k. Fixed: per-log `config_hash` guard mirrors the OTel adapter's strictness.
- `[review P3, Major]` **Overlapping log sources double-counted** (dir + file inside it, or a repeated path). Fixed: canonical-realpath de-duplication (handles `file://` URIs vs plain paths).
- `[review P4, Minor]` Domain errors now surface as clean CLI messages (`click.ClickException`), not tracebacks; `analyze` validates paths exist.
- `[review P5, Minor]` Malformed token-usage attributes (e.g. `"300.5"`) no longer abort an ingest — tolerant coercion, garbage counts 0.
- `[review S2, Minor]` Documented (honesty footer): the per-task *point* is the UMVUE (exactly 0 when c < k) while the per-task CI is Bayesian — at that boundary the point can sit marginally below its own interval. Two correct estimators, one edge; shown unreconciled rather than fudged.
- `[review, unscored]` `--k-max` beyond min trials now warns when clamped.

- Repo bootstrapped from the Phase 0 spec (2026-07-02). Spec snapshot in `docs/spec/`, canonical JSON Schemas packaged at `src/keen_touchstone/schemas/`.
- **Schema change vs Phase 0** (`reliability-aggregate.schema.json`): added nullable `headline_k` — the Phase 0 draft stored a headline `pass_hat_k` without recording which k it referred to, which implementation proved ambiguous. Suite-level rollups use the reserved `task_key` `"__suite__"` (documented; a dedicated SuiteAggregate shape is a candidate for spec v0.2).
