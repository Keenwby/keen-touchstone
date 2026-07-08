"""Keyless end-to-end demo of the online loop.

The staged story (all synthetic, seeded, zero keys — and it says so):

1. A stream of UNLABELED production traffic (no task tags at all): 8 task
   types, healthy at first, then every task degrades sharply mid-stream (the
   incident). Trigger-designed, not probabilistic.
2. Derived task identity: the template strategy groups the unlabeled runs —
   and because this is synthetic, we can grade it against the hidden truth.
3. `watch`: tumbling windows over the stream — clean windows read ok, the
   incident window fires a CONFIDENT BREACH (whole CI below the SLO).
4. `compare`: pre-incident vs post-incident halves, paired on the derived
   tasks — SIGNIFICANT_REGRESSION with magnitude, CI, and p-value.

One metric definition, offline and online — the thesis sentence, executable.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from rich.console import Console

from keen_touchstone.adapters.otel_traces import trials_from_traces
from keen_touchstone.adapters.signatures import (
    TemplateStrategy,
    grouping_readout,
    runs_from_spans,
)
from keen_touchstone.aggregate import build_suite_result
from keen_touchstone.demo.tracegen import gen_spans_labeled, write_spans_jsonl
from keen_touchstone.online import compare, watch_stream
from keen_touchstone.report import RunMeta, emit

ONLINE_DEMO_TASKS: dict[str, float] = {
    "support/refund-flow": 0.95,
    "support/address-change": 0.95,
    "support/subscription-cancel": 0.9,
    "ops/log-triage": 0.9,
    "ops/incident-summary": 0.95,
    "ops/capacity-report": 0.9,
    "research/competitor-scan": 0.95,
    "research/pricing-scan": 0.9,
}
DRIFT_AFTER = 10  # each task's 11th run onward is degraded
DRIFT_P = 0.15
RUNS_PER_TASK = 20


def run_online_demo(out: Path, seed: int = 2026, console: Console | None = None) -> dict[str, Any]:
    console = console or Console()
    console.print(
        "[dim]all synthetic, seeded, zero API keys — an 8-task traffic stream with NO task "
        "tags, healthy for 10 runs per task, then every task degrades (the incident)[/dim]\n"
    )
    spans, labels = gen_spans_labeled(
        seed=seed, tasks=ONLINE_DEMO_TASKS, runs_per_task=RUNS_PER_TASK,
        unsigned_runs=0, include_signatures=False,
        drift={"task": "*", "after": DRIFT_AFTER, "p": DRIFT_P},
    )
    stream = write_spans_jsonl(out / "stream.unlabeled.jsonl", spans)
    console.print(f"[dim]stream: {len(labels)} runs → {stream}[/dim]")

    # ---- 1. derived task identity, graded against the hidden truth --------
    console.print("\n[bold]① task identity from nothing:[/bold] template strategy on unlabeled traffic")
    readout = grouping_readout(spans, TemplateStrategy())
    truth_purity = _purity_vs_truth(spans, labels)
    console.print(
        f"  {readout.n_grouped} runs → {readout.n_clusters} clusters · singleton rate "
        f"{readout.singleton_rate:.0%} · [bold]purity vs hidden truth {truth_purity:.0%}[/bold] "
        "[dim](only computable because this stream is synthetic)[/dim]"
    )
    for cluster in readout.clusters[:3]:
        console.print(f"  [dim]· {cluster.size}× “{(cluster.template or '')[:64]}”[/dim]")
    console.print(f"  [yellow]caveat:[/yellow] {readout.caveat}")

    # ---- 2. watch: the incident fires a confident breach -------------------
    console.print("\n[bold]② watch:[/bold] tumbling windows, SLO pass^2 ≥ 0.5")
    watch_report = watch_stream(
        spans, window_size=40, slo="0.5@2", strategy=TemplateStrategy(), seed=seed,
    )
    from keen_touchstone.cli import _print_watch

    _print_watch(watch_report)

    # ---- 3. compare: pre vs post incident, paired --------------------------
    console.print("\n[bold]③ compare:[/bold] pre-incident vs post-incident, paired on derived tasks")
    runs = sorted(runs_from_spans(spans), key=lambda r: str(r.attr("start_time")))
    half = len(runs) // 2
    halves = {}
    for name, chunk in (("baseline", runs[:half]), ("candidate", runs[half:])):
        chunk_spans = [span for run in chunk for span in run.spans]
        ingested = trials_from_traces(chunk_spans, strategy=TemplateStrategy())
        result = build_suite_result(
            ingested.tasks, context="online", model=ingested.model,
            agent_config_hash=f"demo-{name}", task_key_source="task_signature", seed=seed,
        )
        emit(result, out / name,
             RunMeta(source=f"{name} half of the stream", task_name="(derived)",
                     model=ingested.model, scorer="declared outcomes"),
             console=Console(quiet=True))
        halves[name] = {t.task_key: (t.n_rollouts, round(t.pass_rate * t.n_rollouts))
                        for t in result.tasks}

    comparison = compare(halves["baseline"], halves["candidate"], at_k=2, seed=seed)
    style = "red" if comparison.verdict == "SIGNIFICANT_REGRESSION" else "yellow"
    console.print(
        f"  [{style}]{comparison.verdict}[/{style}] at pass^2: mean delta "
        f"{comparison.mean_delta:+.3f} [{comparison.ci_low:+.3f}, {comparison.ci_high:+.3f}], "
        f"p={comparison.p_value:.4f} ({comparison.n_shared} shared derived tasks)"
    )
    console.print(
        "\n[dim]the loop, closed: unlabeled traffic → derived tasks → windowed pass^k → "
        "breach alert → paired regression verdict — one metric definition throughout[/dim]"
    )
    return {"readout": readout, "truth_purity": truth_purity,
            "watch": watch_report, "comparison": comparison}


def _purity_vs_truth(spans, labels) -> float:
    strategy = TemplateStrategy()
    clusters: dict[str, list[str]] = {}
    for run in runs_from_spans(spans):
        sig = strategy.signature(run)
        if sig is not None:
            clusters.setdefault(sig, []).append(labels[run.trace_id]["task"])
    total = pure = 0
    for members in clusters.values():
        total += len(members)
        pure += Counter(members).most_common(1)[0][1]
    return pure / total if total else 0.0
