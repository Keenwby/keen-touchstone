"""compare + slo-gate: hand goldens, scipy parity, honesty paths, exit codes."""


import numpy as np
import pytest

from keen_touchstone.aggregate import build_suite_result
from keen_touchstone.online import compare, load_aggregate_tasks, slo_gate
from keen_touchstone.online.compare import _sign_flip_p_value
from keen_touchstone.report import RunMeta, emit
from keen_touchstone.stats import TaskTrials

# ---------------------------------------------------------- permutation core


def test_exact_sign_flip_hand_golden() -> None:
    """6 identical negative deltas: only the all-plus and all-minus sign
    patterns reach |mean| = 0.2 → p = 2/64 exactly."""
    p, method = _sign_flip_p_value(np.array([-0.2] * 6), seed=0, n_resamples=0)
    assert method == "exact_sign_flip"
    assert p == pytest.approx(2 / 64)


def test_exact_matches_scipy_reference() -> None:
    from scipy import stats as scipy_stats

    rng = np.random.default_rng(3)
    for _ in range(5):
        d = rng.normal(-0.1, 0.2, size=9)
        ours, method = _sign_flip_p_value(d, seed=0, n_resamples=0)
        assert method == "exact_sign_flip"
        ref = scipy_stats.permutation_test(
            (d,), lambda x: np.mean(x), permutation_type="samples",
            n_resamples=np.inf, alternative="two-sided",
        )
        assert ours == pytest.approx(ref.pvalue, abs=1e-12)


def test_sampled_path_reproducible_and_close_to_exact() -> None:
    rng = np.random.default_rng(5)
    d = rng.normal(-0.15, 0.1, size=20)  # > enumeration limit
    p1, method = _sign_flip_p_value(d, seed=7, n_resamples=20000)
    p2, _ = _sign_flip_p_value(d, seed=7, n_resamples=20000)
    assert method == "sampled_sign_flip"
    assert p1 == p2


# ----------------------------------------------------------------- compare()


def _cfg(rates: dict[str, tuple[int, int]]) -> dict[str, tuple[int, int]]:
    return rates


def test_regression_detected_with_magnitude_and_ci() -> None:
    baseline = {f"t{i}": (10, 9) for i in range(8)}
    candidate = {f"t{i}": (10, 5) for i in range(8)}
    result = compare(baseline, candidate, at_k=2, seed=1)
    assert result.verdict == "SIGNIFICANT_REGRESSION"
    assert result.mean_delta < -0.4
    assert result.ci_high < 0
    assert not result.underpowered
    assert result.n_shared == 8


def test_identical_configs_are_noise() -> None:
    cfg = {f"t{i}": (10, 7 + i % 3) for i in range(8)}
    result = compare(cfg, dict(cfg), at_k=2, seed=1)
    assert result.verdict == "NOISE"
    assert result.mean_delta == pytest.approx(0.0)


def test_no_shared_tasks_is_a_clean_error() -> None:
    with pytest.raises(ValueError, match="no shared tasks"):
        compare({"a": (5, 5)}, {"b": (5, 5)})


def test_underpowered_flag_and_unpaired_accounting() -> None:
    baseline = {"a": (10, 9), "b": (10, 9), "c": (10, 9), "x": (10, 9)}
    candidate = {"a": (10, 2), "b": (10, 2), "c": (10, 2), "y": (10, 2)}
    result = compare(baseline, candidate, at_k=2, seed=1)
    assert result.underpowered  # 3 shared < 5
    assert result.n_only_baseline == 1 and result.n_only_candidate == 1
    assert any("absence of evidence" in n for n in result.notes)


def test_k_clamped_to_shared_floor() -> None:
    baseline = {"a": (4, 4), "b": (4, 3), "c": (4, 4), "d": (4, 2), "e": (4, 4)}
    candidate = {key: (4, 1) for key in baseline}
    result = compare(baseline, candidate, at_k=9, seed=1)
    assert result.k == 4
    assert any("clamped" in n for n in result.notes)


# ------------------------------------------------- aggregate.json round trip


def _write_aggregate(tmp_path, name: str, per_task: dict[str, tuple[int, int]]):
    tasks = [TaskTrials(key, n, c) for key, (n, c) in per_task.items()]
    result = build_suite_result(tasks, context="online", model="m", agent_config_hash=name)
    from rich.console import Console

    out = tmp_path / name
    emit(result, out, RunMeta(source=name, task_name="t", model="m"), console=Console(quiet=True))
    return out / "aggregate.json"


def test_reconstruction_from_aggregate_is_exact(tmp_path) -> None:
    per_task = {f"task/{i}": (12, 12 - i) for i in range(6)}
    path = _write_aggregate(tmp_path, "base", per_task)
    loaded, payload = load_aggregate_tasks(path)
    assert loaded == per_task
    assert payload["suite"]["task_key"] == "__suite__"


def test_cli_compare_end_to_end(tmp_path) -> None:
    from click.testing import CliRunner

    from keen_touchstone.cli import main

    base = _write_aggregate(tmp_path, "base", {f"t{i}": (10, 9) for i in range(8)})
    regressed = _write_aggregate(tmp_path, "cand", {f"t{i}": (10, 4) for i in range(8)})
    runner = CliRunner()
    bad = runner.invoke(main, ["compare", str(base), str(regressed), "--at-k", "2"])
    assert bad.exit_code == 1, bad.output
    assert "SIGNIFICANT_REGRESSION" in bad.output

    same = runner.invoke(main, ["compare", str(base), str(base), "--at-k", "2"])
    assert same.exit_code == 0
    assert "NOISE" in same.output


# ------------------------------------------------------------------ slo-gate


def test_slo_gate_matrix(tmp_path) -> None:
    # healthy: rates ~0.9 → pass^2 ≈ 0.8 with CI well above 0.3
    healthy = _write_aggregate(tmp_path, "healthy", {f"t{i}": (10, 9) for i in range(8)})
    # broken: all zero → pass^2 = 0, CI hugging zero
    broken = _write_aggregate(tmp_path, "broken", {f"t{i}": (10, 0) for i in range(8)})
    # borderline: point below 0.62 but CI straddles
    borderline = _write_aggregate(tmp_path, "border", {f"t{i}": (10, 7 + (i % 3)) for i in range(8)})

    ok = slo_gate(healthy, "0.3@2")
    assert ok.ok and ok.level == "pass"

    breach = slo_gate(broken, "0.3@2")
    assert not breach.ok and breach.level == "breach"
    assert "CONFIDENT BREACH" in breach.message

    warn = slo_gate(borderline, "0.62@2")
    assert warn.ok and warn.level == "warning"
    strict = slo_gate(borderline, "0.62@2", strict=True)
    assert not strict.ok and "STRICT MODE" in strict.message

    with pytest.raises(ValueError, match="decay curve has no k=40"):
        slo_gate(healthy, "0.5@40")


def test_cli_slo_gate_exit_codes(tmp_path) -> None:
    from click.testing import CliRunner

    from keen_touchstone.cli import main

    broken = _write_aggregate(tmp_path, "broken", {f"t{i}": (10, 0) for i in range(8)})
    runner = CliRunner()
    fail = runner.invoke(main, ["slo-gate", str(broken), "--slo", "0.3@2"])
    assert fail.exit_code == 1
    healthy = _write_aggregate(tmp_path, "healthy", {f"t{i}": (10, 9) for i in range(8)})
    good = runner.invoke(main, ["slo-gate", str(healthy), "--slo", "0.3@2"])
    assert good.exit_code == 0, good.output
