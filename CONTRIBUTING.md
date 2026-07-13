# Contributing

## Setup

```bash
uv sync
uv run pytest -q       # 255 tests, all green before and after your change
uv run ruff check .
```

## The bar

This repo's discipline comes from eight adversarial review rounds, and contributions are
held to the same standard:

- **Bug reports:** include a runnable reproducer (exact command, observed vs expected
  output). Findings without reproducers are hypotheses — welcome, but labeled as such.
- **Bug fixes:** ship with a regression test that fails before the fix and passes after.
  Check whether the same root cause has siblings one door over (grep for the pattern the
  fix replaced) — that is where our reviewers found most of their catches.
- **Statistics changes:** must keep the truth-recovery tests green (synthetic data with
  designed ground truth recovered within CI) and preserve the honesty invariants: no
  suppressed error bars, no silently clamped values, UNDERPOWERED said out loud.
- **New surfaces:** honor the exit-code contract (0 pass / 1 gate fired / 2 usage /
  3 domain error) and the "ingest, don't collect" stance (no daemons, no dashboards).

## Scope guardrails

The deliberately-deferred list (embedding clustering, sequential-testing corrections,
OTLP collectors, LLM-inference attribution, and more) lives in the CHANGELOG history —
check it before proposing; most "missing features" are documented decisions.
