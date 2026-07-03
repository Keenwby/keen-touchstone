# KeenTouchstone data model — specification (Phase 0 / RFC)

> Status: draft. Field-level definitions for the 4 artifacts. JSON Schemas in [`schemas/`](./schemas/). Rationale: design thesis §10.
> Convention: **REQUIRED** / **RECOMMENDED** / **OPTIONAL** follow RFC-2119 sense. `harness.*` = net-new fields no standard defines.

---

## 0. Design invariants (the non-negotiables)

1. **Everything joins on `trace_id`.** Span, Cassette, EvalVerdict, ReliabilityAggregate all carry or derive a `trace_id`. The *join* across offline/online/replay/attribution is the entire differentiation — break it and KeenTouchstone is just another wrapper.
2. **OTel-GenAI-native at the wire, opinionated above it.** Ingest/emit `gen_ai.*` (W3C trace context). Pin a snapshot of the spec (it's experimental). Value lives in the judgment/replay layer the spec omits, never in a new span format.
3. **One metric, three execution contexts.** A check (groundedness, goal-alignment, schema, pass^k) is *defined once* and runs in: an offline suite, an inline guardrail, or an async online sampler. Never re-implement the same logic three times.
4. **Store distributions, not point estimates.** Reliability numbers (`pass^k`, etc.) always carry CIs + sample size. A bare point estimate is a bug.
5. **Every LLM-based instrument ships its own calibration.** No judge/attributor is trusted without a κ-vs-human artifact. Anti-circularity: κ is computed only against *human-confirmed* labels (auto-labels may expand a set but may not compute the judge's own κ).

---

## 1. Artifact A — `Span` (the substrate)

A single step in an agent run. Adopt OTel `gen_ai.*` as the canonical on-disk vocabulary; add `harness.*` fields the standard lacks.

### Standard (OTel) fields
| Field | Level | Notes |
|---|---|---|
| `trace_id`, `span_id`, `parent_span_id` | REQUIRED | W3C trace context — the run tree |
| `gen_ai.operation.name` | REQUIRED | ∈ `{chat, embeddings, execute_tool, invoke_agent, create_agent, retrieval}` |
| `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.model` | RECOMMENDED | provider/model (for attribution + model-swap) |
| `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` | RECOMMENDED | cost as a co-equal axis |
| `gen_ai.conversation.id` | RECOMMENDED | session join |
| `gen_ai.agent.id`, `gen_ai.agent.name` | RECOMMENDED | multi-agent attribution |
| `gen_ai.tool.name`, `gen_ai.tool.call.id` | RECOMMENDED | tool-step identity |
| `input.messages`, `output.messages` | OPTIONAL | **privacy-gated, opt-in**; oversized payloads stored by reference (content-addressed hash) |

### Net-new `harness.*` fields (no standard defines these)
| Field | Level | Why |
|---|---|---|
| `harness.step_id` | REQUIRED | **monotonic per-run integer** → a deterministic *linear* replay order (the span *tree* doesn't give you one) |
| `harness.task_signature` | RECOMMENDED | **the task-identity key** — see §5. The keystone for online `pass^k`: groups *organic* production runs into "the same task" |
| `harness.replay.cassette_ref` | OPTIONAL | pointer from a live span to its reproducing cassette event |
| `harness.nondeterminism_class` | OPTIONAL | ∈ `{llm_sample, tool, clock, rng, concurrency, data_drift, config_drift}` — makes the six sources of non-determinism first-class for replay |
| `harness.agent_config_hash` | RECOMMENDED | hash of {prompts, tools, model params, scaffold version} → the **harness identity** for reliability grouping + model-vs-harness A/B |

Schema: [`schemas/span.schema.json`](./schemas/span.schema.json).

---

## 2. Artifact B — `Cassette` (the replay artifact)

Append-only **JSONL**, one `TraceEvent` per line. The *only* artifact that turns a production trace into an offline test for free. JSONL (not YAML) for append-only streaming + O(1) line indexing.

```
TraceEvent {
  run_id,            // == Span.trace_id  (JOIN KEY)
  step_id,           // == Span.harness.step_id (linear order)
  timestamp,
  kind,              // ∈ {llm_call, tool_call, decision}
  input,             // exact recorded input (for divergence detection)
  output,            // recorded output (replayed verbatim)
  metadata           // model_id / tool_id (replay validates these match)
}
```

**Replay contract** (thesis §10.5):
- **Stub isolation:** `ReplayLLMClient` / `ReplayToolClient` fetch the next event by `kind`, **validate `model_id`/`tool_id` matches the recording**, return recorded output with zero network.
- **Exhaustion detection:** per-`kind` cursors; calling past the end raises `Exhausted events of kind …` — **never silently falls back to live systems** (that would invalidate the debug session).
- **Clock virtualization:** intercept `time.time()`/`datetime.now()`, serve recorded timestamps.
- **Divergence detection:** stubs assert replay inputs match recorded inputs → catches control-flow drift.

**Honest scope (load-bearing):** replay reproduces the **harness orchestration logic given frozen model/tool outputs** — **NOT** model reproducibility. Hosted-API non-determinism is unfixable from outside (vendors update weights silently; `model_id` only *detects* divergence). Stating this limit is itself the credibility.

Schema: [`schemas/cassette.schema.json`](./schemas/cassette.schema.json).

---

## 3. Artifact C — `EvalVerdict` (the missing link)

A grader's judgment on a run, keyed to `trace_id`, carrying provenance and a **calibration link** (anti-circularity). Shape borrows Inspect's `Score` (full edit-history provenance).

```
EvalVerdict {
  verdict_id, trace_id (FK), sample_id,
  scorer_id, scorer_version,
  scorer_kind,             // ∈ {programmatic, model_graded, trajectory}
  tier,                    // ∈ {T0_deterministic, T1_reference, T2_ungrounded} — the grader cascade
  value,                   // bool | float | categorical
  explanation,
  judge_model,             // null for programmatic
  judge_calibration_ref,   // → JudgeCalibration run (REQUIRED when scorer_kind=model_graded)
  trajectory_invariant,    // bool — did the verdict depend on the EXACT path? (false = path-invariant, gaming-resistant)
  score_history[]          // Inspect-style provenance (author, reason, edits)
}
```

- `scorer_kind=trajectory` + `trajectory_invariant` are the load-bearing net-new fields for "valid *path*, not just valid end-state."
- **Grader cascade is enforced:** authors declare `tier`, default to `T0_deterministic`, and must *justify* falling back to `T2_ungrounded`. Refuse "LLM-judge" where a deterministic check would do.
- **`judge_calibration_ref` is REQUIRED for `model_graded`** — a model-graded verdict with no calibration link is rejected. This is the structural enforcement of "prove the judge before you trust the judgment."

Schema: [`schemas/eval-verdict.schema.json`](./schemas/eval-verdict.schema.json).

### 3b. `JudgeCalibration` (referenced, not a top-level artifact in v0)
Computed against a **human-confirmed** anchor set (Anthropic's empirical floor: **20–50 tasks from real failures**; below ~30 labels, display "calibration unreliable (n too small)" — never a κ point estimate). Emits: Cohen's **κ** (not raw %-agreement), TPR/FPR, and a status ∈ `{JUDGE_LICENSED, NEEDS_HUMAN}`. **The gate:** CI promotion of an `EvalVerdict` whose `judge_calibration_ref` resolves to `NEEDS_HUMAN` (or κ below threshold) is **blocked**. *(This is the verified-unowned move — Ragas/Vertex compute κ but never gate.)*

---

## 4. Artifact D — `ReliabilityAggregate` (the stats layer)

Per `(task_id | task_signature, agent_config_hash, model)`, rolling up N rollouts of `EvalVerdict`.

```
ReliabilityAggregate {
  task_key,                // task_id (offline) OR task_signature (online) — see §5
  agent_config_hash, model, n_rollouts,
  pass_rate,
  pass_at_k, pass_hat_k,       // pass^k headline (P(all k succeed))
  pass_hat_k_ci_low, _ci_high, // CI on the pass^k ESTIMATE itself (Inspect lacks this)
  reliability_decay_curve[],   // k = 1..N  (Inspect lacks this)
  variance, skew,              // distribution shape, not a point
  cost_mean, cost_p95, token_mean,   // cost as co-equal axis
  context,                     // ∈ {offline, online, replay} — the SAME metric across all three
  attribution                  // optional: model-vs-harness adjudication result (§6)
}
```

- **UI honesty requirement:** empirical `pass^k` is highly variable at small N → always store CIs + skew, never a bare point estimate.
- **`context`** is what unifies offline benchmark, live traffic, and replay under one metric definition — the offline↔online loop.
- Stats the layer must ship (thesis §6.4): variance decomposition (tell the user *which lever* — more samples vs more tasks), clustered SE by default, paired-difference testing, a power-analysis gate (`SIGNIFICANT / NOISE / UNDERPOWERED-NEED-N-MORE`), and a **Bayesian sequential-stopping** mode (Beta-Binomial posterior) for expensive rollouts.

Schema: [`schemas/reliability-aggregate.schema.json`](./schemas/reliability-aggregate.schema.json).

---

## 5. Task identity (the online-`pass^k` keystone) — OPEN PROBLEM

`pass^k` needs **k runs of the same task**. Offline, `task_id` comes from the dataset. **Online, organic traffic has no labels** — so we must *derive* a `task_signature` that groups "the same task" without over- or under-merging. This is the single unsolved sub-problem that, per the adversarial audit (2026-06-23), no shipping tool has cracked — and therefore the keystone of moat #1.

Candidate strategies (to prototype + measure, not yet decided):
- **Intent/embedding clustering** of the initial user request (cluster centroid = task).
- **Tool-sequence signature** (the DAG of tool calls as a structural key).
- **Template extraction** (parameterize variable slots: "book flight {X}→{Y}" ⇒ one task).
- **Declared task tags** (let the app annotate `harness.task_signature` — highest fidelity, lowest coverage).

Requirements for any strategy: **stable** (same task → same signature across runs), **discriminative** (different tasks don't collide), **cheap** (runs online), and **inspectable** (a human can see why two runs grouped). The chosen approach ships behind a swappable `TaskSignatureStrategy` interface, with a quality readout (intra-cluster coherence) so users can see grouping quality.

---

## 6. Model-vs-harness attribution (counterfactual adjudication)

The verified-unowned move (#3). When a run fails, produce a **measured** "X% attributable to the model, Y% to the harness," not a guessed label.

- **Method:** on a failed trajectory, run **counterfactuals** — swap the model (same harness) and/or swap/ablate harness components (same model) on the *same task*, and measure the success-rate delta. This sidesteps the unreplicated "harness explains 15–50%" literature by *measuring the swing on the user's own data*.
- **Output:** a confidence-banded attribution on `ReliabilityAggregate.attribution`, plus the ETCLOVG harness-layer hypothesis (Execution/Tooling/Context/Lifecycle/Observability/Verification/Governance) as a *ranked hint*, never an authoritative verdict.
- **Honest ceiling:** step-level attribution SOTA is ~14% (Who&When); agent-level ~53%. Ship the *measured A/B* (high trust) as the headline; ship the *inferred layer hypothesis* (low trust) with a loud confidence band.

---

## 7. Extensibility (plugin contracts)

- **Scorers** = plugins (`@scorer`, inheriting Inspect's decorator model) — wrap any DeepEval/Ragas metric, write deterministic checks, register custom trajectory invariants.
- **Reducers** = plugins (`@score_reducer`) — `pass^k` decay, Bayesian sequential, custom reliability metrics.
- **Guardrails** = scorers-that-can-block — the "one metric, three contexts" property: a scorer declared once is reusable as offline check, inline guardrail, or async sampler.
- **Ingest adapters** = one per dialect (`gen_ai.*`, `openinference.*`, `claude_code.*`) → normalize to the canonical `Span`.
- **TaskSignatureStrategy** = the §5 interface.

---

## Changelog
- **2026-06-23** — Phase 0 draft. Incorporates the adversarial verification of the three moat walls (online `pass^k`, judge-calibration-gate, model-vs-harness attribution — all HOLD) and the κ correction (Ragas/Vertex compute κ; the gate is the unowned part).
