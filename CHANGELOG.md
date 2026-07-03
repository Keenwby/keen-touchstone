# Changelog

## [Unreleased] — 0.1.0 (Phase 1 MLP)

- Repo bootstrapped from the Phase 0 spec (2026-07-02). Spec snapshot in `docs/spec/`, canonical JSON Schemas packaged at `src/keen_touchstone/schemas/`.
- **Schema change vs Phase 0** (`reliability-aggregate.schema.json`): added nullable `headline_k` — the Phase 0 draft stored a headline `pass_hat_k` without recording which k it referred to, which implementation proved ambiguous. Suite-level rollups use the reserved `task_key` `"__suite__"` (documented; a dedicated SuiteAggregate shape is a candidate for spec v0.2).
