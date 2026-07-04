# Changelog

## [Unreleased] — 0.3.0 (Phase 3: cassette + deterministic replay, built 2026-07-03)

**The third moat, delivered:** no shipping tool can deterministically replay an agent run. `touchstone replay` re-executes your orchestration logic against the taped model/tool outputs — same result or same crash, zero network — or names the exact step where the harness diverged.

- **Cassette I/O** (`cassette/io.py`): Artifact B concretized — append-only JSONL, every line schema-validated on write AND read, flushed per event (a crash still leaves a replayable prefix); spliced/reordered/mixed-run tapes refused with line numbers. Reserved decisions: `__task_input__` (self-contained tape), `__clock__`, `__final__` (crashes are taped as `{type, message}` — reproducing the failure is the point).
- **RecordingIO** (`cassette/record.py`): the explicit seam (`llm_call`/`tool_call`/`now`/`decision`) — no SDK monkey-patching; context manager tapes crashes and re-raises; co-emits Spans (same `trace_id`/`step_id`, `harness.replay.cassette_ref` back-pointer) so recorded runs feed `ingest` → three artifacts joined on one trace_id.
- **ReplayIO + replay_run** (`cassette/replay.py`): all seven primitives — per-kind cursors; canonical-JSON input matching with first-difference snippets; model/tool identity validation; `CassetteExhausted` (never a silent live fallback); clock from tape; outcome reconciliation (`REPLAY FAITHFUL` / `DIVERGED …at step N`); unconsumed-events report (the other half of control-flow drift).
- **Demo** (`touchstone replay-demo`, keyless): a deterministic harness bug — the ledger returns `"1,130.00"` and the scaffold's `float()` crashes (the model did nothing wrong; the harness did). Record 18 runs → pass^k report from the recorded spans → replay the failing run: same `ValueError`, zero network. Failures designed by trigger, not probability (Phase 2 demo lesson).
- **Honest limits stated everywhere**: harness-logic re-execution, NOT model reproducibility; only seam-routed calls are taped; async and zero-code-change interception deferred.

## 0.2.0 (Phase 2: the judge-calibration gate, built 2026-07-03)

**The second moat, delivered:** the field computes judge-vs-human agreement; KeenTouchstone gates on it — in CI (`judge gate`, exit 1) and in the data path (unlicensed judges' verdicts are refused at ingest).

- **The exam** (`judge/kappa.py`): hand-implemented Cohen's κ (sklearn dev-parity to 1e-12) + item-bootstrap CI; TPR/FPR with Jeffreys CIs (FPR = the rubber-stamp direction); abstention excluded-and-counted; one-class anchor guard; κ-paradox warning on imbalanced sets.
- **The alt-test** (`judge/alt_test.py`, from the paper's formulas — the reference repo is unlicensed and untouched): leave-one-annotator-out alignment scores, one-sided d̄-vs-ε tests (t at n≥30, Wilcoxon below, degenerate branch), Benjamini–Yekutieli FDR, winning rate ω and ρ̄; requires ≥3 annotators, honestly inapplicable below.
- **License + gate** (`judge/license.py`): auditable JudgeCalibration artifact (new canonical schema, SPEC §3b concretized); <30 items withholds κ entirely; abstention cap; κ threshold on point (default) or CI-low (`--strict`); a failed alt-test outranks a passable κ; `judge gate` exits 1 on NEEDS_HUMAN.
- **Anti-circularity, structural**: `label_source: "human"` pinned in schema, pydantic, and the anchors parser — auto-labels may expand eval sets, never certify their own judge.
- **The loop** (`judge/verdicts.py` + `ingest --outcomes-from --license`): EvalVerdict JSONL (contract-validated per line) as the authoritative outcome source for unlabeled traces; model-graded verdicts require a matching JUDGE_LICENSED license; licenses are not transferable across judges/prompts (prompt hashed into identity).
- **Thin runner** (`judge/runner.py`): one fixed hashed judging prompt via Inspect's model interface (mockllm = keyless deterministic; anthropic/ollama unchanged); unparseable replies become abstentions. Deliberately thin — "not another judge wrapper" (Phase 0 anti-goal).
- **Keyless demo** (`touchstone judge demo`): 3 simulated annotators; good judge (κ≈0.97) licensed → labels unlabeled traces → pass^k report; sloppy judge designed by confusion matrix (FPR≈0.5) blocked by κ AND alt-test, and refused in the data path. Design note: an "accuracy + guess-bias" sloppy judge flattered itself at high base rates — error-rate design is the honest construction.

### Adversarial review round 2 (2026-07-03, two independent skeptics attacking the gate and the exam math; all findings reproduced, tagged `[review2 …]`)

- `[review2 A1, Critical]` **Verdict laundering**: relabeling a model-graded verdict `programmatic` (one word) bypassed the license entirely while it still carried `judge_model`. Fixed at three layers: pydantic validator, schema `allOf` conditional (non-model_graded ⇒ judge fields null), and file read. Honest scope note: a verdict stripped of ALL judge tells is indistinguishable from a genuine deterministic check — the gate defends against mislabeling and honest mistakes, not an operator lying to themselves.
- `[review2 A4, Major]` **License transfer**: any judge could cite a real license's `calibration_id` (printed in reports — a name, not a secret). Fixed: verdicts must match the license's `judge_id` AND `judge_model`.
- `[review2 A2+A3, Major]` **Gate trusted the stored `status`**: a hand-flipped NEEDS_HUMAN→JUDGE_LICENSED (schema-valid!) passed, even printing κ=0.44 while stamping LICENSED. Fixed: `check_license` **re-derives the decision from the license's own recorded numbers** and refuses self-contradictory licenses; thresholds bounded in schema+pydantic (κ∈[0,1], min_items≥1); lenient-threshold licenses pass but are named out loud. **Trust boundary documented**: license.json is an artifact of your own pipeline, not a cryptographic credential; HMAC signing is future work if licenses ever cross trust boundaries.
- `[review2 K1, Major]` **alt-test small-n dishonesty**: the degenerate branch fabricated p=0.0 (inverting the evidence ordering — all-ties beat strictly-better-on-half under BY, encoded in our own golden test), and 3-item panels could print "may replace humans". Fixed: exact conservative Bernoulli bounds ((1−ε)^n ties, ((1−ε)/2)^n sweep — derived in the docstring), ≥10 comparable items per annotator or the whole test is honestly "underpowered". Golden test rewritten at 12 items; the old 4-item panel now correctly proves nothing.
- `[review2 K2, Major]` **κ CI undercoverage**: percentile bootstrap reported "95%" that measured as low as ~77% with a tiny minority class (90% prevalence, n=30 — the tool's own stated operating regime). Fixed: κ CI withheld with a plain-language note when the minority class has <10 items; strict mode (`gate_on=ci_low`) consequently refuses rather than leaning on an over-narrow bound.
- `[review2 A6, Minor]` Dangling verdicts (matching no trace) now warn. `[review2, Minor]` Escaped two latent HTML sinks (calibration_id, created_at).

## 0.1.0 (Phase 1 MLP, built 2026-07-02)

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
