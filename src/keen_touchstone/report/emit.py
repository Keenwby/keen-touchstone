"""Write the analysis outputs: schema-valid aggregate.json + terminal summary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from keen_touchstone import __version__
from keen_touchstone.aggregate import SuiteResult


@dataclass(frozen=True)
class RunMeta:
    """Provenance stamped into every output."""

    source: str  # human-readable: "inspect logs: <task>" / "otel traces: <file>"
    task_name: str
    model: str
    scorer: str | None = None


def emit(result: SuiteResult, out_dir: Path, meta: RunMeta, console: Console | None = None,
         extra: dict | None = None) -> Path:
    """Write aggregate.json (+ report.html once M5 lands) and print a summary.

    Returns the path of aggregate.json.
    """
    console = console or Console()
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "keen_touchstone_version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": meta.source,
        "task_name": meta.task_name,
        "model": meta.model,
        "scorer": meta.scorer,
        "warnings": result.warnings,
        "lever": result.lever,
        "suite": result.suite.to_schema_dict(),
        "tasks": [t.to_schema_dict() for t in result.tasks],
    }
    if extra:
        payload.update(extra)
    json_path = out_dir / "aggregate.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    from .html import render_html

    html_path = out_dir / "report.html"
    html_path.write_text(render_html(result, meta))

    _print_summary(result, meta, console)
    console.print(f"\n[dim]wrote[/dim] {json_path}")
    console.print(f"[dim]wrote[/dim] {html_path}  [dim](open in a browser)[/dim]")
    return json_path


def _print_summary(result: SuiteResult, meta: RunMeta, console: Console) -> None:
    suite = result.suite
    k = suite.headline_k
    console.print()
    console.rule("[bold]KeenTouchstone — reliability summary[/bold]")
    console.print(f"[dim]{meta.source} · model {meta.model} · context {suite.context}[/dim]\n")

    if k and k > 1 and suite.pass_hat_k is not None:
        pass1 = next(p for p in suite.reliability_decay_curve if p.k == 1)
        console.print(
            f"  pass@1 [bold]{pass1.pass_hat_k:.1%}[/bold]  →  "
            f"pass^{k} [bold red]{suite.pass_hat_k:.1%}[/bold red] "
            f"[dim]\\[{suite.pass_hat_k_ci_low:.1%}, {suite.pass_hat_k_ci_high:.1%}] "
            f"95% CI, {suite.ci_method}[/dim]"
        )
        console.print(
            f"  [dim]{len(result.tasks)} tasks × {suite.n_rollouts // max(len(result.tasks), 1)} trials "
            f"(n={suite.n_rollouts} rollouts)[/dim]\n"
        )

    table = Table(title=f"per-task reliability at k={k}", title_style="bold")
    table.add_column("task")
    table.add_column("n", justify="right")
    table.add_column("pass rate", justify="right")
    table.add_column(f"pass^{k}", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("mean tokens", justify="right")
    for t in result.tasks:
        table.add_row(
            t.task_key,
            str(t.n_rollouts),
            f"{t.pass_rate:.0%}",
            f"{t.pass_hat_k:.1%}" if t.pass_hat_k is not None else "—",
            f"[{t.pass_hat_k_ci_low:.1%}, {t.pass_hat_k_ci_high:.1%}]",
            f"{t.token_mean:,.0f}" if t.token_mean is not None else "—",
        )
    console.print(table)

    if result.lever == "more_rollouts_per_task":
        console.print(
            "  [yellow]lever:[/yellow] most remaining noise is within-task sampling — "
            "more trials per task tightens this faster than more tasks"
        )
    elif result.lever == "more_tasks":
        console.print(
            "  [yellow]lever:[/yellow] tasks genuinely differ in difficulty — "
            "more tasks improves the estimate faster than more reruns"
        )
    if suite.power_status:
        console.print(f"  [yellow]power:[/yellow] {suite.power_status} — the headline CI is wide")
    for w in result.warnings:
        console.print(f"  [red]warning:[/red] {w}")
