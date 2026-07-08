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


class _CleanErrors:
    """Domain errors (bad inputs, mixed configs) become clean CLI messages."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type in (ValueError, FileNotFoundError):
            raise click.ClickException(str(exc))
        return False


def _analyze_logs(
    logs: list, scorer: str | None, out: Path, k_max: int | None, headline_k: int | None,
    resamples: int, seed: int, source: str, score_key: str | None = None,
) -> None:
    from keen_touchstone.adapters.inspect_logs import trials_from_logs

    ingest = trials_from_logs(logs, scorer=scorer, score_key=score_key)
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
@click.argument("logs", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--scorer", default=None, help="Scorer name to read (required if the logs carry several).")
@click.option("--score-key", default=None, help="For dict-valued scores: the key that defines success.")
@click.option("--out", type=click.Path(path_type=Path), default=Path("out/analysis"), show_default=True)
@shared_options
def analyze(logs: tuple[str, ...], scorer: str | None, score_key: str | None, out: Path, k_max: int | None, headline_k: int | None, resamples: int, seed: int) -> None:
    """Analyze existing Inspect .eval/.json logs (files or log directories)."""
    from keen_touchstone.adapters.inspect_logs import resolve_log_paths

    with _CleanErrors():
        paths = resolve_log_paths(list(logs))
        _analyze_logs(paths, scorer, out, k_max, headline_k, resamples, seed,
                      source=f"inspect logs: {len(paths)} file(s)", score_key=score_key)


@main.command()
@click.argument("task_file", type=click.Path(exists=True))
@click.option("--model", default="mockllm/model", show_default=True, help="Inspect model spec, e.g. anthropic/claude-sonnet-5.")
@click.option("--epochs", type=int, default=10, show_default=True, help="Trials per task (the N in pass^k).")
@click.option("--out", type=click.Path(path_type=Path), default=Path("out/run"), show_default=True)
@click.option("--scorer", default=None)
@click.option("--score-key", default=None, help="For dict-valued scores: the key that defines success.")
@shared_options
def run(task_file: str, model: str, epochs: int, out: Path, scorer: str | None, score_key: str | None, k_max: int | None, headline_k: int | None, resamples: int, seed: int) -> None:
    """Run an Inspect task with N epochs, then analyze the resulting log."""
    import inspect_ai

    console.print(f"[dim]running {task_file} on {model} with {epochs} epochs via Inspect AI…[/dim]")
    eval_logs = inspect_ai.eval(
        tasks=task_file, model=model, epochs=epochs, log_dir=str(out / "logs")
    )
    with _CleanErrors():
        _analyze_logs(list(eval_logs), scorer, out, k_max, headline_k, resamples, seed,
                      source=f"inspect run: {task_file} ({epochs} epochs)", score_key=score_key)


@main.command()
@click.argument("traces", type=click.Path(exists=True, path_type=Path), required=False)
@click.option("--demo", "use_demo", is_flag=True, help="Generate synthetic demo traces instead of reading a file.")
@click.option("--outcome-regex", default=None, help="Fallback outcome rule: regex matched against the root span's output.messages.")
@click.option("--threshold", type=float, default=1.0, show_default=True, help="Success threshold for gen_ai.evaluation.score.value.")
@click.option("--signature-strategy", "sig_strategy", type=click.Choice(["declared", "template", "tool-sequence"]),
              default="declared", show_default=True,
              help="Task identity: declared harness.task_signature tags, or DERIVED grouping (template-masked input / tool sequence) — derived grouping is a hypothesis and ships with a quality readout.")
@click.option("--outcomes-from", "verdicts_path", type=click.Path(exists=True), default=None,
              help="EvalVerdict JSONL as the authoritative outcome source (joined on trace_id).")
@click.option("--license", "license_path", type=click.Path(exists=True), default=None,
              help="Judge license (license.json) — REQUIRED when verdicts contain model-graded scores.")
@click.option("--out", type=click.Path(path_type=Path), default=Path("out/ingest"), show_default=True)
@shared_options
def ingest(traces: Path | None, use_demo: bool, outcome_regex: str | None, threshold: float,
           sig_strategy: str, verdicts_path: str | None, license_path: str | None,
           out: Path, k_max: int | None, headline_k: int | None, resamples: int, seed: int) -> None:
    """Ingest OTel gen_ai.* spans (JSONL) — production traces in, pass^k out.

    No golden dataset: runs group by the declared harness.task_signature and
    outcomes come from harness.outcome / gen_ai.evaluation.score.* /
    --outcome-regex — or, for unlabeled traffic, from a LICENSED judge's
    verdicts (--outcomes-from + --license; an unlicensed judge is refused).
    Runs with no signature or no resolvable outcome are excluded and counted,
    never guessed.
    """
    from keen_touchstone.adapters.otel_traces import read_spans_jsonl, trials_from_traces

    if use_demo == (traces is not None):
        raise click.UsageError("pass exactly one of: a TRACES file, or --demo")
    if use_demo:
        from keen_touchstone.demo.tracegen import gen_spans, write_spans_jsonl

        traces = write_spans_jsonl(out / "traces.demo.jsonl", gen_spans(seed=seed))
        console.print(f"[dim]generated synthetic demo traces → {traces}[/dim]")

    with _CleanErrors():
        outcome_overrides = None
        scorer_label = "declared outcomes"
        if verdicts_path is not None:
            from keen_touchstone.judge.verdicts import (
                load_license,
                outcomes_from_verdicts,
                read_verdicts,
            )

            verdicts = read_verdicts(verdicts_path)
            license_ = load_license(license_path) if license_path else None
            outcome_overrides = outcomes_from_verdicts(verdicts, license_, threshold=threshold)
            scorer_label = (
                f"licensed judge {license_.judge_id} ({license_.calibration_id})"
                if license_ is not None
                else "programmatic verdicts"
            )
        spans = read_spans_jsonl(traces)
        strategy = None
        readout = None
        if sig_strategy != "declared":
            from keen_touchstone.adapters.signatures import STRATEGIES, grouping_readout

            strategy = STRATEGIES[sig_strategy]()
            readout = grouping_readout(spans, strategy)
            strategy = STRATEGIES[sig_strategy]()  # fresh cursors for the ingest pass
        ingested = trials_from_traces(
            spans, strategy=strategy, threshold=threshold, outcome_regex=outcome_regex,
            outcome_overrides=outcome_overrides,
        )
        result = _build_ingest_result(
            ingested, k_max, headline_k, resamples, seed,
            task_key_source="declared_tag" if sig_strategy == "declared" else "task_signature",
        )
        if readout is not None:
            result.warnings.append(readout.caveat)
            _print_grouping_readout(readout)
    emit(
        result,
        out,
        RunMeta(
            source=f"otel traces: {traces} ({ingested.n_runs} runs)",
            task_name="(derived from traces)",
            model=ingested.model,
            scorer=scorer_label,
        ),
        console=console,
        extra={"grouping": readout.to_dict()} if readout is not None else None,
    )


def _print_grouping_readout(readout) -> None:
    from rich.table import Table

    console.print(f"\n[bold]task-identity readout[/bold] [dim](strategy: {readout.strategy})[/dim]")
    console.print(
        f"  {readout.n_grouped}/{readout.n_runs} runs grouped into {readout.n_clusters} clusters · "
        f"{readout.n_unsigned} unsigned · singleton rate {readout.singleton_rate:.0%} · "
        f"largest cluster {readout.largest_cluster_share:.0%}"
        + (f" · purity vs declared tags {readout.purity_vs_declared:.0%}" if readout.purity_vs_declared is not None else "")
    )
    table = Table(show_header=True)
    table.add_column("cluster (why these grouped)")
    table.add_column("size", justify="right")
    table.add_column("exemplar")
    for cluster in readout.clusters[:8]:
        table.add_row(
            (cluster.template or cluster.signature)[:70],
            str(cluster.size),
            (cluster.exemplars[0] if cluster.exemplars else "—")[:60],
        )
    console.print(table)
    console.print(f"  [yellow]caveat:[/yellow] {readout.caveat}")


def _build_ingest_result(ingested, k_max, headline_k, resamples, seed, task_key_source="declared_tag"):
    result = build_suite_result(
        ingested.tasks,
        context="online",
        model=ingested.model,
        agent_config_hash=ingested.agent_config_hash,
        task_key_source=task_key_source,
        k_max=k_max,
        headline_k=headline_k,
        n_resamples=resamples,
        seed=seed,
    )
    result.warnings.extend(ingested.warnings)
    return result


@main.command()
@click.argument("traces", type=click.Path(exists=True))
@click.option("--window", type=click.IntRange(min=2), default=30, show_default=True, help="Runs per tumbling window.")
@click.option("--slo", required=True, help='Reliability red line, e.g. "0.6@4" = pass^4 must stay ≥ 60%.')
@click.option("--signature-strategy", "sig_strategy", type=click.Choice(["declared", "template", "tool-sequence"]), default="declared", show_default=True)
@click.option("--threshold", type=float, default=1.0, show_default=True)
@click.option("--outcome-regex", default=None)
@click.option("--out", "out_json", type=click.Path(path_type=Path), default=None, help="Also write the window series as JSON.")
@click.option("--follow", is_flag=True, help="Keep re-scanning the file (Ctrl-C to stop).")
@click.option("--every", type=float, default=30.0, show_default=True, help="Re-scan interval for --follow, seconds.")
@click.option("--seed", type=int, default=2026, show_default=True)
def watch(traces: str, window: int, slo: str, sig_strategy: str, threshold: float,
          outcome_regex: str | None, out_json: Path | None, follow: bool, every: float,
          seed: int) -> None:
    """Continuous reliability: tumbling windows over a trace stream, with a
    two-level SLO alert (confident breach = exit 1)."""
    import json as json_mod
    import time as time_mod
    from dataclasses import asdict

    from keen_touchstone.adapters.otel_traces import read_spans_jsonl
    from keen_touchstone.online import watch_stream

    def one_pass():
        from keen_touchstone.adapters.signatures import STRATEGIES

        strategy = STRATEGIES[sig_strategy]() if sig_strategy != "declared" else None
        with _CleanErrors():
            report = watch_stream(
                read_spans_jsonl(traces), window_size=window, slo=slo,
                strategy=strategy, threshold=threshold, outcome_regex=outcome_regex, seed=seed,
            )
        _print_watch(report)
        if out_json is not None:
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json_mod.dumps(asdict(report) | {
                "windows": [asdict(w) for w in report.windows]}, indent=2) + "\n")
            console.print(f"[dim]wrote[/dim] {out_json}")
        return report

    report = one_pass()
    while follow:
        time_mod.sleep(every)
        report = one_pass()
    if report.breached:
        raise SystemExit(1)


_WATCH_GLYPHS = {"ok": "[green]✓ ok[/green]", "warning": "[yellow]⚠ warning[/yellow]",
                 "breach": "[red]✗ BREACH[/red]", "insufficient_n": "[dim]… insufficient n[/dim]",
                 "pending": "[dim]· pending[/dim]", "no_data": "[dim]— no data[/dim]"}


def _print_watch(report) -> None:
    from rich.table import Table

    console.print()
    console.rule(f"[bold]watch — SLO pass^{report.slo_k} ≥ {report.slo_value}[/bold]")
    table = Table(show_header=True)
    for col, justify in (("win", "right"), ("time", "left"), ("runs", "right"),
                         ("tasks", "right"), ("pass^k", "right"), ("95% CI", "right"),
                         ("status", "left")):
        table.add_column(col, justify=justify)
    for w in report.windows:
        table.add_row(
            str(w.index),
            (w.start_time or "?")[11:16] + "–" + (w.end_time or "?")[11:16],
            str(w.n_runs), str(w.n_tasks),
            f"{w.pass_hat_k:.2f}" if w.pass_hat_k is not None else "—",
            f"[{w.ci_low:.2f}, {w.ci_high:.2f}]" if w.ci_low is not None else "—",
            _WATCH_GLYPHS.get(w.status, w.status),
        )
    console.print(table)
    latest = next((w for w in reversed(report.windows) if w.status != "pending"), None)
    if latest is not None and latest.detail:
        style = "red" if latest.status == "breach" else "yellow"
        console.print(f"  [{style}]{latest.detail}[/{style}]")
    for warning in report.warnings:
        console.print(f"  [dim]note: {warning}[/dim]")


def _load_entry(entry: str):
    import importlib

    if ":" not in entry:
        raise click.UsageError(f"--entry must be module:function, got {entry!r}")
    module_name, fn_name = entry.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as err:
        raise click.UsageError(f"cannot import module {module_name!r}: {err}") from err
    except Exception as err:  # import-time side effects — a usage problem, not a divergence
        raise click.UsageError(
            f"importing {module_name!r} raised {type(err).__name__}: {err}"
        ) from err
    fn = getattr(module, fn_name, None)
    if fn is None:
        raise click.UsageError(f"{module_name!r} has no attribute {fn_name!r}")
    if not callable(fn):
        raise click.UsageError(f"{entry!r} is not callable")
    import inspect as _inspect

    try:
        _inspect.signature(fn).bind("io", "task_input")
    except TypeError as err:
        raise click.UsageError(
            f"entry function must accept (io, task_input); {entry!r} does not ({err})"
        ) from err
    return fn


@main.command()
@click.argument("cassette", type=click.Path(exists=True))
@click.option("--entry", required=True, help="Your agent function as module:function — it will be EXECUTED locally against the tape (zero network for taped calls; this is not a sandbox).")
def replay(cassette: str, entry: str) -> None:
    """Deterministically re-execute a recorded run against its cassette.

    Reproduces your orchestration logic under the frozen model/tool outputs —
    same result or same crash — or names the exact step where the harness
    diverged from the recording. Never falls back to live systems.
    """
    from keen_touchstone.cassette import replay_run

    console.print(f"[dim]importing and executing entry {entry} against {cassette}…[/dim]")
    with _CleanErrors():
        entry_fn = _load_entry(entry)
        report = replay_run(cassette, entry_fn)
    style = "green" if report.faithful else "red"
    console.print(f"[{style}]{report.verdict}[/{style}]")
    if not report.faithful:
        raise SystemExit(1)


@main.command("replay-demo")
@click.option("--invoices", type=click.IntRange(min=1), default=6, show_default=True)
@click.option("--runs-per-invoice", type=click.IntRange(min=1), default=3, show_default=True)
@click.option("--out", type=click.Path(path_type=Path), default=Path("out/replay-demo"), show_default=True)
def replay_demo(invoices: int, runs_per_invoice: int, out: Path) -> None:
    """Keyless end-to-end: record a buggy agent live → pass^k report from the
    recorded spans → replay the failing run and reproduce the exact crash."""
    from keen_touchstone.adapters.otel_traces import read_spans_jsonl, trials_from_traces
    from keen_touchstone.cassette import replay_run
    from keen_touchstone.demo.replay_agent import (
        DEMO_CONFIG_HASH,
        demo_agent,
        record_demo_runs,
    )

    with _CleanErrors():
        console.print(
            f"[dim]recording {invoices}×{runs_per_invoice} live runs of the buggy demo agent "
            "(simulated tools, zero keys)…[/dim]"
        )
        results = record_demo_runs(out, invoices=invoices, runs_per_invoice=runs_per_invoice)
        n_failed = sum(1 for _, _, ok in results if not ok)
        console.print(
            f"  recorded {len(results)} runs → {len(results) - n_failed} succeeded, "
            f"[red]{n_failed} failed[/red] · cassettes in {out}/cassettes/"
        )

        ingested = trials_from_traces(read_spans_jsonl(out / "spans.jsonl"))
        result = build_suite_result(
            ingested.tasks, context="offline", model=ingested.model,
            agent_config_hash=DEMO_CONFIG_HASH, task_key_source="declared_tag",
        )
        result.warnings.extend(ingested.warnings)
        emit(
            result, out / "report",
            RunMeta(
                source=f"recorded runs ({len(results)} cassettes)",
                task_name="demo/reconcile-invoice", model=ingested.model,
                scorer="recorded outcomes",
            ),
            console=console,
        )

        failed = next(((rid, inv) for rid, inv, ok in results if not ok), None)
        if failed is None:
            console.print("[yellow]no failing run this time — nothing to replay[/yellow]")
            return
        run_id, invoice = failed
        cassette = out / "cassettes" / f"{run_id}.cassette.jsonl"
        console.print(f"\n[bold]the flagship story:[/bold] replaying the failed run of {invoice} "
                      f"[dim]({run_id[:12]}…)[/dim]")
        report = replay_run(cassette, demo_agent)
        style = "green" if report.faithful else "red"
        console.print(f"  [{style}]{report.verdict}[/{style}]")
        console.print(
            f"\n[dim]replay it yourself:[/dim]\n  touchstone replay {cassette} "
            "--entry keen_touchstone.demo.replay_agent:demo_agent"
        )


@main.group()
def judge() -> None:
    """Judge exam & license: prove the judge before trusting its judgments.

    Everyone computes judge-vs-human agreement; nobody blocks on it. Here an
    unlicensed judge's scores are not evidence — `judge gate` fails the build.
    """


@judge.command()
@click.argument("anchors", type=click.Path(exists=True))
@click.option("--judge-labels", "judge_labels_path", type=click.Path(exists=True), default=None,
              help="BYO judge answers (JSONL: item_id + judge_label true/false/\"unknown\").")
@click.option("--run-judge", "run_judge_flag", is_flag=True, help="Run the built-in thin judge over the anchors instead.")
@click.option("--model", default=None, help="Judge model for --run-judge (e.g. anthropic/claude-sonnet-5, ollama/qwen3:14b).")
@click.option("--judge-id", default=None, help="Judge identity on the license (default: model or labels filename).")
@click.option("--epsilon", type=float, default=0.2, show_default=True, help="alt-test cost-benefit margin (paper: 0.1–0.2).")
@click.option("--q", type=float, default=0.05, show_default=True, help="alt-test FDR level (Benjamini–Yekutieli).")
@click.option("--kappa-threshold", type=float, default=0.6, show_default=True, help="Minimum κ for a license (0.8 = strong).")
@click.option("--min-items", type=int, default=30, show_default=True)
@click.option("--max-abstention", type=float, default=0.2, show_default=True)
@click.option("--strict", is_flag=True, help="Gate on the κ CI lower bound instead of the point estimate.")
@click.option("--seed", type=int, default=2026, show_default=True)
@click.option("--out", type=click.Path(path_type=Path), default=Path("out/judge"), show_default=True)
def calibrate(anchors: str, judge_labels_path: str | None, run_judge_flag: bool, model: str | None,
              judge_id: str | None, epsilon: float, q: float, kappa_threshold: float,
              min_items: int, max_abstention: float, strict: bool, seed: int, out: Path) -> None:
    """Exam a judge against human-labeled anchors and issue (or refuse) a license."""
    from keen_touchstone.artifacts import CalibrationThresholds
    from keen_touchstone.judge.anchors import read_anchors, read_judge_labels
    from keen_touchstone.judge.calibrate import calibrate as run_calibration
    from keen_touchstone.report.judge_emit import emit_license

    if (judge_labels_path is None) == (not run_judge_flag):
        raise click.UsageError("pass exactly one of --judge-labels FILE or --run-judge --model M")
    if run_judge_flag and not model:
        raise click.UsageError("--run-judge requires --model")

    with _CleanErrors():
        anchor_set = read_anchors(anchors)
        prompt_hash = None
        extra_notes: list[str] = []
        if run_judge_flag:
            from keen_touchstone.judge.runner import run_judge

            console.print(f"[dim]running judge {model} over {len(anchor_set.items)} anchor items…[/dim]")
            run = run_judge(anchor_set, model=model)
            labels, prompt_hash = run.labels, run.prompt_hash
            extra_notes.extend(run.warnings)
            resolved_judge_id = judge_id or model
        else:
            labels = read_judge_labels(judge_labels_path)
            resolved_judge_id = judge_id or Path(judge_labels_path).stem

        thresholds = CalibrationThresholds(
            kappa_licensed=kappa_threshold, min_items=min_items,
            max_abstention=max_abstention, gate_on="ci_low" if strict else "point",
        )
        calibration, exam_result, alt = run_calibration(
            anchor_set, labels, judge_id=resolved_judge_id, judge_model=model,
            judge_prompt_hash=prompt_hash, thresholds=thresholds,
            epsilon=epsilon, q=q, seed=seed, extra_notes=extra_notes,
        )
        emit_license(calibration, exam_result, alt, out, console=console)
    if calibration.status != "JUDGE_LICENSED":
        raise SystemExit(1)


@judge.command("demo")
@click.option("--items", type=int, default=60, show_default=True, help="Anchor items in the synthetic exam.")
@click.option("--seed", type=int, default=2026, show_default=True)
@click.option("--out", type=click.Path(path_type=Path), default=Path("out/judge-demo"), show_default=True)
def judge_demo(items: int, seed: int, out: Path) -> None:
    """Keyless end-to-end: exam two judges → license one, block one → the
    licensed judge labels unlabeled traces → pass^k report; the unlicensed
    judge is refused in the data path."""
    from keen_touchstone.demo.judge_demo import run_judge_demo

    with _CleanErrors():
        run_judge_demo(out, n_items=items, seed=seed, console=console)


@judge.command()
@click.argument("license_file", type=click.Path(exists=True))
def gate(license_file: str) -> None:
    """The CI gate: exit 0 iff the license says JUDGE_LICENSED."""
    import json as json_mod

    import jsonschema as jsonschema_mod

    from keen_touchstone.artifacts import JudgeCalibration, load_schema
    from keen_touchstone.judge.license import check_license

    with _CleanErrors():
        data = json_mod.loads(Path(license_file).read_text())
        try:
            jsonschema_mod.validate(data, load_schema("judge-calibration"))
        except jsonschema_mod.ValidationError as err:
            raise ValueError(f"{license_file} is not a valid license: {err.message}") from err
        calibration = JudgeCalibration.model_validate(data)
    ok, message = check_license(calibration)
    console.print(("[green]" if ok else "[red]") + message)
    if not ok:
        raise SystemExit(1)


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
