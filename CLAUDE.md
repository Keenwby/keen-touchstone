# keen-touchstone — agent instructions

**What this is:** KeenTouchstone, an OSS agentic eval & reliability harness (`pass^k` ± CI + decay curves over Inspect AI logs and OTel GenAI traces). Phase 1 MLP. The design contract is `docs/spec/` (Phase 0 snapshot) + the thesis at `~/workspace/claude-projects/research/research/deep-research/agentic-eval-reliability-harness-thesis.md`.

## Hard rules

- **NEVER `git push` or add a remote without an explicit ask.** Local commits only. The repo is pre-publication.
- **Wrap Inspect AI, don't fork it.** All inspect-ai touchpoints stay isolated in `src/keen_touchstone/adapters/inspect_logs.py` and the CLI `run` command. Pinned `~=0.3.244`; use the public log API (`inspect_ai.log`), never parse `.eval` files by hand.
- **Store distributions, not point estimates.** Any reliability number leaving the stats core carries n, CI, and `ci_method`. A bare point estimate is a bug (SPEC invariant 4).
- **Statistical honesty over polish:** decay curves use k ≤ min(nᵢ) unless the user explicitly opts into task-dropping (then label it loudly). CIs on aggregates resample *tasks*, never pooled attempts.

## Commands

```bash
uv sync                 # env
uv run pytest           # all tests (hypothesis property tests included — they must stay green)
uv run ruff check .     # lint
uv run touchstone demo  # keyless E2E: Inspect eval → stats → report
```

## Layout

- `src/keen_touchstone/stats/` — the crown jewel; estimator math documented in README §stats contract. Change nothing here without running the full property-test suite.
- `src/keen_touchstone/adapters/` — inspect_logs (offline), otel_traces (online). One module per dialect.
- `src/keen_touchstone/schemas/` — the canonical Phase 0 JSON Schemas (packaged; runtime-validated on emit).
- `docs/spec/` — frozen Phase 0 spec text (README/SPEC/PRIOR-ART). Treat as a snapshot; schema truth lives in `src/keen_touchstone/schemas/`.
