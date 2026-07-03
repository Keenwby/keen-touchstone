"""Report rendering: HTML structure, honesty content, and emit() outputs."""

import numpy as np

from keen_touchstone.aggregate import build_suite_result
from keen_touchstone.report import RunMeta, emit
from keen_touchstone.report.html import render_html
from keen_touchstone.stats import TaskTrials


def _result(n_tasks: int = 6, n: int = 10, seed: int = 9):
    rng = np.random.default_rng(seed)
    tasks = [
        TaskTrials(f"task/{i}", n, int(rng.binomial(n, rng.uniform(0.4, 0.95))),
                   tokens=tuple(int(x) for x in rng.integers(100, 500, size=n)))
        for i in range(n_tasks)
    ]
    return build_suite_result(
        tasks, context="offline", model="mockllm/model", agent_config_hash="cfg123",
    )


META = RunMeta(source="unit test", task_name="demo", model="mockllm/model", scorer="s")


def test_html_contains_chart_tables_and_honesty_footer() -> None:
    html = render_html(_result(), META)
    assert "<svg" in html and "polyline" in html and "polygon" in html  # band + lines
    assert "pass^k — all k attempts succeed" in html  # legend
    assert "pass@k — at least one succeeds" in html
    assert "curve data (table view)" in html  # the relief-rule table view
    assert "honesty footer" in html
    assert "C(c,k)/C(n,k)" in html
    assert "cluster bootstrap" in html
    assert "prefers-color-scheme: dark" in html  # theme-aware
    assert "cdn" not in html.lower() and "https://" not in html  # self-contained


def test_html_escapes_task_keys() -> None:
    tasks = [TaskTrials("<script>alert(1)</script>", 5, 3), TaskTrials("ok", 5, 4)]
    result = build_suite_result(tasks, context="online", model="m<b>", agent_config_hash="h")
    html = render_html(result, RunMeta(source="s", task_name="t", model="m<b>", scorer=None))
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_warnings_and_lever_surface_in_html() -> None:
    result = _result(n_tasks=3)  # triggers small-sample warning
    html = render_html(result, META)
    assert "Read this before acting" in html
    assert "between-task tail" in html


def test_emit_writes_json_and_html(tmp_path) -> None:
    from rich.console import Console

    result = _result()
    json_path = emit(result, tmp_path, META, console=Console(quiet=True))
    assert json_path.exists()
    assert (tmp_path / "report.html").exists()
    text = (tmp_path / "report.html").read_text()
    assert text.startswith("<!doctype html>")
