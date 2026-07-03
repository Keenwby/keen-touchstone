"""KeenTouchstone CLI: demo / analyze / run / ingest."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from keen_touchstone.aggregate import build_suite_result
from keen_touchstone.report import RunMeta, emit

console = Console()


@click.group()
@click.version_option(package_name="keen-touchstone")
def main() -> None:
    """KeenTouchstone: pass^k ± CI + reliability decay curves for agents.

    pass@1 answers "can it work?" — deployment asks "does it work every
    time?". Feed it Inspect eval logs (offline) or OTel GenAI traces
    (online); get the same honest reliability statistics either way.
    """


_shared = [
    click.option("--k-max", type=int, default=None, help="Cap the decay curve at this k (default: min trials per task)."),
    click.option("--headline-k", type=int, default=None, help="k for the headline pass^k (default: half the curve depth — near k=n the per-task estimator degenerates)."),
    click.option("--resamples", type=int, default=2000, show_default=True, help="Bootstrap resamples for the suite CI."),
    click.option("--seed", type=int, default=2026, show_default=True, help="Seed for the bootstrap (reproducible CIs)."),
]


def shared_options(fn):
    for opt in reversed(_shared):
        fn = opt(fn)
    return fn


def _analyze_logs(
    logs: list, scorer: str | None, out: Path, k_max: int | None, headline_k: int | None,
    resamples: int, seed: int, source: str,
) -> None:
    from keen_touchstone.adapters.inspect_logs import trials_from_logs

    ingest = trials_from_logs(logs, scorer=scorer)
    result = build_suite_result(
        ingest.tasks,
        context="offline",
        model=ingest.model,
        agent_config_hash=ingest.agent_config_hash,
        task_key_source="dataset_id",
        k_max=k_max,
        headline_k=headline_k,
        n_resamples=resamples,
        seed=seed,
    )
    result.warnings.extend(ingest.warnings)
    emit(
        result,
        out,
        RunMeta(source=source, task_name=ingest.task_name, model=ingest.model, scorer=ingest.scorer_name),
        console=console,
    )


@main.command()
@click.argument("logs", nargs=-1, required=True)
@click.option("--scorer", default=None, help="Scorer name to read (required if the logs carry several).")
@click.option("--out", type=click.Path(path_type=Path), default=Path("out/analysis"), show_default=True)
@shared_options
def analyze(logs: tuple[str, ...], scorer: str | None, out: Path, k_max: int | None, headline_k: int | None, resamples: int, seed: int) -> None:
    """Analyze existing Inspect .eval/.json logs (files or log directories)."""
    from keen_touchstone.adapters.inspect_logs import resolve_log_paths

    paths = resolve_log_paths(list(logs))
    _analyze_logs(paths, scorer, out, k_max, headline_k, resamples, seed, source=f"inspect logs: {len(paths)} file(s)")


@main.command()
@click.argument("task_file", type=click.Path(exists=True))
@click.option("--model", default="mockllm/model", show_default=True, help="Inspect model spec, e.g. anthropic/claude-sonnet-5.")
@click.option("--epochs", type=int, default=10, show_default=True, help="Trials per task (the N in pass^k).")
@click.option("--out", type=click.Path(path_type=Path), default=Path("out/run"), show_default=True)
@click.option("--scorer", default=None)
@shared_options
def run(task_file: str, model: str, epochs: int, out: Path, scorer: str | None, k_max: int | None, headline_k: int | None, resamples: int, seed: int) -> None:
    """Run an Inspect task with N epochs, then analyze the resulting log."""
    import inspect_ai

    console.print(f"[dim]running {task_file} on {model} with {epochs} epochs via Inspect AI…[/dim]")
    eval_logs = inspect_ai.eval(
        tasks=task_file, model=model, epochs=epochs, log_dir=str(out / "logs")
    )
    _analyze_logs(list(eval_logs), scorer, out, k_max, headline_k, resamples, seed, source=f"inspect run: {task_file} ({epochs} epochs)")


@main.command()
@click.argument("traces", type=click.Path(exists=True, path_type=Path), required=False)
@click.option("--demo", "use_demo", is_flag=True, help="Generate synthetic demo traces instead of reading a file.")
@click.option("--outcome-regex", default=None, help="Fallback outcome rule: regex matched against the root span's output.messages.")
@click.option("--threshold", type=float, default=1.0, show_default=True, help="Success threshold for gen_ai.evaluation.score.value.")
@click.option("--out", type=click.Path(path_type=Path), default=Path("out/ingest"), show_default=True)
@shared_options
def ingest(traces: Path | None, use_demo: bool, outcome_regex: str | None, threshold: float,
           out: Path, k_max: int | None, headline_k: int | None, resamples: int, seed: int) -> None:
    """Ingest OTel gen_ai.* spans (JSONL) — production traces in, pass^k out.

    No golden dataset: runs group by the declared harness.task_signature and
    outcomes come from harness.outcome / gen_ai.evaluation.score.* /
    --outcome-regex. Runs with no signature or no resolvable outcome are
    excluded and counted, never guessed.
    """
    from keen_touchstone.adapters.otel_traces import read_spans_jsonl, trials_from_traces

    if use_demo == (traces is not None):
        raise click.UsageError("pass exactly one of: a TRACES file, or --demo")
    if use_demo:
        from keen_touchstone.demo.tracegen import gen_spans, write_spans_jsonl

        traces = write_spans_jsonl(out / "traces.demo.jsonl", gen_spans(seed=seed))
        console.print(f"[dim]generated synthetic demo traces → {traces}[/dim]")

    ingested = trials_from_traces(
        read_spans_jsonl(traces), threshold=threshold, outcome_regex=outcome_regex
    )
    result = build_suite_result(
        ingested.tasks,
        context="online",
        model=ingested.model,
        agent_config_hash=ingested.agent_config_hash,
        task_key_source="declared_tag",
        k_max=k_max,
        headline_k=headline_k,
        n_resamples=resamples,
        seed=seed,
    )
    result.warnings.extend(ingested.warnings)
    emit(
        result,
        out,
        RunMeta(
            source=f"otel traces: {traces} ({ingested.n_runs} runs)",
            task_name="(derived from traces)",
            model=ingested.model,
            scorer="declared outcomes",
        ),
        console=console,
    )


@main.command()
@click.option("--epochs", type=int, default=12, show_default=True)
@click.option("--out", type=click.Path(path_type=Path), default=Path("out/demo"), show_default=True)
@click.option("--demo-seed", type=int, default=2026, show_default=True, help="Seed of the simulated flaky agent.")
@shared_options
def demo(epochs: int, out: Path, demo_seed: int, k_max: int | None, headline_k: int | None, resamples: int, seed: int) -> None:
    """Keyless end-to-end demo: simulated flaky agent → Inspect log → decay curve.

    Uses Inspect's mockllm provider and a deterministic seeded scorer that
    simulates per-task success probabilities — no API keys, no network.
    """
    import inspect_ai

    from keen_touchstone.demo.flaky_task import touchstone_demo

    console.print(f"[dim]running the bundled flaky-agent demo ({epochs} epochs, mockllm, no keys)…[/dim]")
    eval_logs = inspect_ai.eval(
        tasks=touchstone_demo(seed=demo_seed),
        model="mockllm/model",
        epochs=epochs,
        log_dir=str(out / "logs"),
    )
    _analyze_logs(list(eval_logs), None, out, k_max, headline_k, resamples, seed, source=f"bundled demo ({epochs} epochs, simulated flaky agent)")
