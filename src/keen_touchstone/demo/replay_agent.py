"""The replay demo: a deterministic, keyless agent with a real orchestration bug.

The agent reconciles invoices. The simulated ledger returns most amounts as
plain numbers, but for certain invoices it returns the amount **as a
comma-formatted string** ("1,234.50") — and the harness parses with
``float(...)``, which crashes. Classic harness bug: the model did nothing
wrong; the scaffold's parsing did. Whether a given invoice is poisoned is a
pure function of its id (lesson from the Phase 2 demo: design failures by
trigger, never by probability).

Everything is scripted and local — zero API keys — but the seam is the real
one: swap ``scripted_llm`` for a real model call and nothing else changes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from keen_touchstone.cassette import RecordingIO

DEMO_CONFIG_HASH = "replaydemo-v1"
DEMO_SIGNATURE_PREFIX = "demo/reconcile"


def _poisoned(invoice_id: str) -> bool:
    return int(invoice_id.split("-")[1]) % 4 == 0


def scripted_llm(prompt: Any) -> str:
    """Deterministic stand-in for a model call (hash-keyed canned text)."""
    digest = hashlib.sha256(str(prompt).encode()).hexdigest()[:8]
    return f"[scripted reply {digest}] proceed step by step."


def sim_ledger(tool_input: dict[str, Any]) -> dict[str, Any]:
    invoice_id = tool_input["invoice_id"]
    base = 100.0 + (int(invoice_id.split("-")[1]) * 7.5)
    if _poisoned(invoice_id):
        return {"invoice_id": invoice_id, "amount": f"{base + 1000:,.2f}"}  # "1,1xx.xx"
    return {"invoice_id": invoice_id, "amount": base}


def sim_tax(tool_input: dict[str, Any]) -> dict[str, Any]:
    return {"tax": round(float(tool_input["amount"]) * 0.07, 2)}


def demo_agent(io, task_input: dict[str, Any]) -> dict[str, Any]:
    """The harness under test. Same code records and replays — only `io` changes."""
    invoice_id = task_input["invoice_id"]
    io.llm_call(f"Plan how to reconcile invoice {invoice_id}.", "demo/planner-v1", scripted_llm)
    entry = io.tool_call("ledger_lookup", {"invoice_id": invoice_id}, sim_ledger)
    amount = float(entry["amount"])  # BUG: crashes on comma-formatted amounts
    checked_at = io.now()
    tax = io.tool_call("tax_check", {"invoice_id": invoice_id, "amount": amount}, sim_tax)
    summary = io.llm_call(
        f"Summarize: {invoice_id} reconciled, total={amount + tax['tax']:.2f}.",
        "demo/writer-v1",
        scripted_llm,
    )
    return {
        "invoice_id": invoice_id,
        "total": round(amount + tax["tax"], 2),
        "checked_at": checked_at.isoformat(),
        "summary": summary,
    }


def demo_agent_v2(io, task_input: dict[str, Any]) -> dict[str, Any]:
    """A 'fixed' harness (different prompt wording) — used to demonstrate that
    replay diverges loudly, at the exact step, when the logic changes."""
    invoice_id = task_input["invoice_id"]
    io.llm_call(f"Please plan reconciliation for {invoice_id}!", "demo/planner-v1", scripted_llm)
    return {"invoice_id": invoice_id}


def record_demo_runs(
    out_dir: str | Path, invoices: int = 6, runs_per_invoice: int = 3
) -> list[tuple[str, str, bool]]:
    """Record invoices×runs live runs. Returns (run_id, invoice_id, ok)."""
    results: list[tuple[str, str, bool]] = []
    for i in range(1, invoices + 1):
        invoice_id = f"INV-{i:03d}"
        for _ in range(runs_per_invoice):
            task_input = {"invoice_id": invoice_id}
            rec = RecordingIO(
                out_dir,
                task_input=task_input,
                task_signature=f"{DEMO_SIGNATURE_PREFIX}/{invoice_id}",
                agent_config_hash=DEMO_CONFIG_HASH,
            )
            try:
                with rec as io:
                    result = demo_agent(io, task_input)
                    io.finish(result)
                results.append((rec.run_id, invoice_id, True))
            except ValueError:
                results.append((rec.run_id, invoice_id, False))
    return results
