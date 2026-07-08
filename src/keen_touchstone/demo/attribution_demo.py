"""Keyless demo of measured model-vs-harness attribution.

One agent, two INDEPENDENTLY switchable fault sources (trigger-designed,
never probabilistic — the standing demo lesson):

- **model fault** (weak planner): on designed invoices the planner produces a
  wrong lookup id → the ledger finds nothing → the run ends "failed-lookup".
  The harness did everything right; the model planned wrong.
- **harness fault** (buggy parser): on designed invoices the ledger returns a
  comma-formatted amount and the scaffold's ``float()`` crashes. The model
  did everything right; the scaffold broke.

One invoice (15) trips BOTH — fixing either alone does not save it, which is
exactly what the 2×2 interaction term exists to expose.

Designed truth (24 tasks): baseline failure 13/24 ≈ 54.2%; model swap removes
6/24 = 25.0pp; harness fix removes 6/24 = 25.0pp; interaction +1/24 ≈ +4.2pp
(the needs-both invoice); irreducible under both swaps: 0. The tool must
recover these numbers — truth-recovery, fifth deployment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console

from keen_touchstone.aggregate import build_suite_result
from keen_touchstone.attribution import decompose, diagnose_cassette
from keen_touchstone.cassette import RecordingIO
from keen_touchstone.report import RunMeta, emit
from keen_touchstone.stats import TaskTrials

N_INVOICES = 24
RUNS_PER_TASK = 2
MODEL_FAULT_IDS = {1, 3, 5, 7, 9, 11, 15}  # weak planner mis-plans these
HARNESS_FAULT_IDS = {2, 4, 6, 8, 10, 12, 15}  # ledger returns comma amounts for these

CELLS = {
    "baseline": ("weak", "buggy"),
    "model_swap": ("strong", "buggy"),
    "harness_swap": ("weak", "fixed"),
    "both_swap": ("strong", "fixed"),
}


def scripted_llm(prompt: Any) -> str:
    return "plan: look up the ledger entry, then reconcile."


def sim_ledger(tool_input: dict[str, Any]) -> dict[str, Any]:
    invoice_id = str(tool_input["invoice_id"])
    num = int(invoice_id.split("-")[1])
    if num > N_INVOICES:  # a mis-planned lookup finds nothing
        return {"missing": True}
    amount = 100.0 + num * 3
    if num in HARNESS_FAULT_IDS:
        return {"invoice_id": invoice_id, "amount": f"{amount + 1000:,.2f}"}  # poison
    return {"invoice_id": invoice_id, "amount": amount}


def make_agent(model_quality: str, harness_version: str):
    """Same orchestration code; the two faults switch independently."""

    def agent(io, task_input: dict[str, Any]) -> dict[str, Any]:
        invoice_id = task_input["invoice_id"]
        num = int(invoice_id.split("-")[1])
        io.llm_call(f"Plan reconciliation for {invoice_id}",
                    f"demo/planner-{model_quality}", scripted_llm)
        # MODEL fault: the weak planner asks for the wrong invoice
        lookup_id = invoice_id
        if model_quality == "weak" and num in MODEL_FAULT_IDS:
            lookup_id = f"INV-{num + 500:03d}"
        entry = io.tool_call("ledger_lookup", {"invoice_id": lookup_id}, sim_ledger)
        if entry.get("missing"):
            return {"invoice_id": invoice_id, "status": "failed-lookup"}
        # HARNESS fault: the buggy parser can't survive comma amounts
        raw = entry["amount"]
        amount = float(raw) if harness_version == "buggy" else float(str(raw).replace(",", ""))
        return {"invoice_id": invoice_id, "status": "reconciled", "total": amount}

    return agent


def run_cell(out: Path, name: str, model_quality: str, harness_version: str) -> dict[str, tuple[int, int]]:
    """Run all tasks n times through the io seam; success = reconciled, no crash."""
    agent = make_agent(model_quality, harness_version)
    counts: dict[str, tuple[int, int]] = {}
    for i in range(1, N_INVOICES + 1):
        invoice_id = f"INV-{i:03d}"
        task_input = {"invoice_id": invoice_id}
        successes = 0
        for _ in range(RUNS_PER_TASK):
            rec = RecordingIO(out / name, task_input=task_input,
                              task_signature=f"demo/attr/{invoice_id}",
                              agent_config_hash=f"attr-{name}")
            try:
                with rec as io:
                    result = agent(io, task_input)
                    io.finish(result)
                if result.get("status") == "reconciled":
                    successes += 1
            except ValueError:
                pass  # the harness crash — taped by RecordingIO, counted as failure
        counts[invoice_id] = (RUNS_PER_TASK, successes)
    return counts


def run_attribution_demo(out: Path, seed: int = 2026, console: Console | None = None) -> dict[str, Any]:
    console = console or Console()
    console.print(
        "[dim]all synthetic, zero API keys — one agent, two switchable fault sources "
        "(weak planner × buggy parser), 24 tasks × 4 experiment cells[/dim]\n"
    )
    cells: dict[str, dict[str, tuple[int, int]]] = {}
    for name, (model_quality, harness_version) in CELLS.items():
        counts = run_cell(out, name, model_quality, harness_version)
        cells[name] = counts
        failure = 1 - sum(c for _, c in counts.values()) / sum(n for n, _ in counts.values())
        console.print(f"  cell [bold]{name:<13}[/bold] (planner={model_quality}, parser={harness_version}) "
                      f"→ failure rate {failure:.1%}")
        tasks = [TaskTrials(key, n, c) for key, (n, c) in counts.items()]
        result = build_suite_result(tasks, context="offline", model=f"demo/planner-{model_quality}",
                                    agent_config_hash=f"attr-{name}", seed=seed)
        emit(result, out / name / "report",
             RunMeta(source=f"attribution cell {name}", task_name="demo/attr",
                     model=f"demo/planner-{model_quality}"),
             console=Console(quiet=True))

    console.print("\n[bold]the sentence no other tool can produce:[/bold]")
    attribution = decompose(cells["baseline"], cells["model_swap"],
                            cells["harness_swap"], cells["both_swap"], seed=seed)
    console.print(f"  {attribution.sentence()}")

    console.print("\n[bold]and the low-trust hint, properly caged[/bold] — diagnosing a failed tape:")
    crashed = next((out / "baseline" / "cassettes").glob("*.cassette.jsonl"))
    for cassette in (out / "baseline" / "cassettes").glob("*.cassette.jsonl"):
        report = diagnose_cassette(cassette)
        if report.failed and report.error_type == "ValueError":
            crashed = cassette
            break
    diagnosis = diagnose_cassette(crashed)
    console.print(f"  [yellow]{diagnosis.disclaimer}[/yellow]")
    for rank, hyp in enumerate(diagnosis.hypotheses, 1):
        console.print(f"  {rank}. {hyp.layer} — {hyp.reason}")

    return {"attribution": attribution, "diagnosis": diagnosis, "cells": cells}
