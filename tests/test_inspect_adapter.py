"""Inspect adapter: real eval → real .eval log → TaskTrials, plus E2E parity.

These tests run the bundled demo task against Inspect's mockllm provider —
local, keyless, a few seconds — so the adapter is exercised against the real
log format of the pinned inspect-ai version, not a hand-rolled imitation.
"""

import json

import jsonschema
import pytest

EPOCHS = 6


@pytest.fixture(scope="session")
def demo_logs(tmp_path_factory: pytest.TempPathFactory) -> list:
    import inspect_ai

    from keen_touchstone.demo.flaky_task import touchstone_demo

    log_dir = tmp_path_factory.mktemp("logs")
    return list(
        inspect_ai.eval(
            tasks=touchstone_demo(),
            model="mockllm/model",
            epochs=EPOCHS,
            log_dir=str(log_dir),
        )
    )


def test_trials_from_logs_shape(demo_logs: list) -> None:
    from keen_touchstone.adapters.inspect_logs import trials_from_logs
    from keen_touchstone.demo.flaky_task import FLAKY_TASK_P

    ingest = trials_from_logs(demo_logs)
    assert ingest.model == "mockllm/model"
    assert ingest.scorer_name == "simulated_flaky_scorer"
    assert sorted(t.task_key for t in ingest.tasks) == sorted(FLAKY_TASK_P)
    assert all(t.n == EPOCHS for t in ingest.tasks)
    assert all(0 <= t.c <= EPOCHS for t in ingest.tasks)
    assert all(t.tokens is not None and len(t.tokens) == EPOCHS for t in ingest.tasks)
    assert len(ingest.agent_config_hash) == 12


def test_deterministic_across_runs(demo_logs: list, tmp_path_factory: pytest.TempPathFactory) -> None:
    """The seeded scorer must give identical outcomes on a fresh eval run,
    regardless of sample concurrency — this is what makes the demo a demo."""
    import inspect_ai

    from keen_touchstone.adapters.inspect_logs import trials_from_logs
    from keen_touchstone.demo.flaky_task import touchstone_demo

    rerun = list(
        inspect_ai.eval(
            tasks=touchstone_demo(),
            model="mockllm/model",
            epochs=EPOCHS,
            log_dir=str(tmp_path_factory.mktemp("logs2")),
        )
    )
    first = {t.task_key: (t.n, t.c) for t in trials_from_logs(demo_logs).tasks}
    second = {t.task_key: (t.n, t.c) for t in trials_from_logs(rerun).tasks}
    assert first == second


def test_parity_with_inspect_end_to_end(demo_logs: list) -> None:
    """Feed the per-sample epoch scores from the real log through Inspect's own
    reducers and through our estimators — identical numbers, sample by sample."""
    from inspect_ai.scorer import pass_at, pass_k

    from keen_touchstone.adapters.inspect_logs import trials_from_logs
    from keen_touchstone.stats import pass_at_k, pass_hat_k

    log = demo_logs[0]
    by_sample: dict[str, list] = {}
    for sample in log.samples:
        by_sample.setdefault(str(sample.id), []).append(sample.scores["simulated_flaky_scorer"])

    trials = {t.task_key: t for t in trials_from_logs(demo_logs).tasks}
    k = 3
    for sample_id, scores in by_sample.items():
        t = trials[sample_id]
        assert pass_k(k)(scores).value == pytest.approx(pass_hat_k(t.n, t.c, k), abs=1e-12)
        assert pass_at(k)(scores).value == pytest.approx(pass_at_k(t.n, t.c, k), abs=1e-12)


def test_multi_model_logs_rejected(demo_logs: list) -> None:
    from keen_touchstone.adapters.inspect_logs import trials_from_logs

    log = demo_logs[0]
    clone = log.model_copy(deep=True)
    clone.eval.model = "another/model"
    with pytest.raises(ValueError, match="one \\(task, model\\) configuration"):
        trials_from_logs([log, clone])


def test_mixed_configurations_rejected(demo_logs: list) -> None:
    """Adversarial-review regression: same task name + model but different
    task_args must not silently pool into one pass^k."""
    from keen_touchstone.adapters.inspect_logs import trials_from_logs

    log = demo_logs[0]
    clone = log.model_copy(deep=True)
    clone.eval.task_args = {"difficulty": "hard"}
    with pytest.raises(ValueError, match="differ in configuration"):
        trials_from_logs([log, clone])


def test_dict_scores_error_without_key_and_work_with_key(demo_logs: list) -> None:
    """Adversarial-review regression: dict-valued Score.value silently
    flattened to 0.0 (a 100%-reliable agent reported as 0%). Now: hard error
    without --score-key; correct counts with it."""
    from keen_touchstone.adapters.inspect_logs import trials_from_logs

    log = demo_logs[0].model_copy(deep=True)
    for sample in log.samples:
        score = sample.scores["simulated_flaky_scorer"]
        score.value = {"acc": score.value, "style": "I"}

    with pytest.raises(ValueError, match="--score-key"):
        trials_from_logs([log])
    with pytest.raises(ValueError, match="not in score dict"):
        trials_from_logs([log], score_key="nope")

    keyed = trials_from_logs([log], score_key="acc")
    plain = trials_from_logs([demo_logs[0]])
    assert {t.task_key: (t.n, t.c) for t in keyed.tasks} == {
        t.task_key: (t.n, t.c) for t in plain.tasks
    }
    assert keyed.scorer_name == "simulated_flaky_scorer[acc]"
    # the decoy key would have flagged everything as failure
    style = trials_from_logs([log], score_key="style")
    assert all(t.c == 0 for t in style.tasks)


def test_list_scores_rejected(demo_logs: list) -> None:
    from keen_touchstone.adapters.inspect_logs import trials_from_logs

    log = demo_logs[0].model_copy(deep=True)
    for sample in log.samples:
        sample.scores["simulated_flaky_scorer"].value = [1.0, 0.0]
    with pytest.raises(ValueError, match="list values"):
        trials_from_logs([log])


def test_resolve_log_paths_dedupes_overlapping_sources(tmp_path_factory) -> None:
    """Adversarial-review regression: dir + a file inside it (or the same path
    twice) must not double every task's n and c."""
    import inspect_ai

    from keen_touchstone.adapters.inspect_logs import resolve_log_paths
    from keen_touchstone.demo.flaky_task import touchstone_demo

    log_dir = tmp_path_factory.mktemp("dedupe-logs")
    inspect_ai.eval(
        tasks=touchstone_demo(), model="mockllm/model", epochs=2, log_dir=str(log_dir)
    )
    only = resolve_log_paths([str(log_dir)])
    assert len(only) == 1
    file_path = next(log_dir.glob("*.eval"))
    combined = resolve_log_paths([str(log_dir), str(file_path), str(file_path)])
    assert len(combined) == 1


def test_cli_demo_end_to_end(tmp_path) -> None:
    """The first-session payoff, as the user would run it: aggregate.json is
    written and every element validates against the canonical schema."""
    from click.testing import CliRunner

    from keen_touchstone.artifacts import load_schema
    from keen_touchstone.cli import main

    out = tmp_path / "demo-out"
    result = CliRunner().invoke(
        main, ["demo", "--epochs", "4", "--out", str(out), "--resamples", "300"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads((out / "aggregate.json").read_text())
    schema = load_schema("reliability-aggregate")
    jsonschema.validate(payload["suite"], schema)
    for task in payload["tasks"]:
        jsonschema.validate(task, schema)
    assert payload["suite"]["task_key"] == "__suite__"
    assert payload["suite"]["headline_k"] == 2  # ceil(4/2)
    assert len(payload["suite"]["reliability_decay_curve"]) == 4
