"""online-demo: the full keyless story, asserted end to end."""

import json

from click.testing import CliRunner
from rich.console import Console

from keen_touchstone.cli import main
from keen_touchstone.demo.online_demo import run_online_demo


def test_online_demo_full_story(tmp_path) -> None:
    summary = run_online_demo(tmp_path, seed=2026, console=Console(quiet=True))

    # ① identity: unlabeled traffic recovers the 8 true tasks perfectly
    assert summary["readout"].n_clusters == 8
    assert summary["readout"].singleton_rate == 0.0
    assert summary["truth_purity"] == 1.0

    # ② watch: clean before the incident, confident breach after — exact windows
    statuses = [w.status for w in summary["watch"].windows]
    assert statuses == ["ok", "ok", "breach", "breach"]
    assert summary["watch"].breached

    # ③ compare: paired regression with a decisive exact p
    comparison = summary["comparison"]
    assert comparison.verdict == "SIGNIFICANT_REGRESSION"
    assert comparison.n_shared == 8
    assert comparison.method == "exact_sign_flip"
    assert comparison.p_value < 0.01
    assert comparison.ci_high < -0.5  # a huge, confidently-bounded drop

    # artifacts on disk, joined
    for rel in ("stream.unlabeled.jsonl", "baseline/aggregate.json", "candidate/aggregate.json"):
        assert (tmp_path / rel).exists(), rel
    baseline = json.loads((tmp_path / "baseline" / "aggregate.json").read_text())
    assert baseline["suite"]["task_key_source"] == "task_signature"


def test_cli_online_demo(tmp_path) -> None:
    result = CliRunner().invoke(main, ["online-demo", "--out", str(tmp_path / "od")])
    assert result.exit_code == 0, result.output
    assert "purity vs hidden truth 100%" in result.output
    assert "BREACH" in result.output
    assert "SIGNIFICANT_REGRESSION" in result.output
    assert "HYPOTHESIS" in result.output  # the caveat is not optional
