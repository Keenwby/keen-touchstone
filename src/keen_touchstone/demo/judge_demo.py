"""Keyless end-to-end demo of the judge gate and the two-phase loop.

What it stages (all synthetic, all seeded, zero API keys — and it says so):

1. An anchor set: 60 items, latent truth, THREE simulated human annotators
   (95% accurate) — enough to activate the alt-test.
2. Two judges take the exam:
   - ``good-judge``: 93% accurate → licensed;
   - ``sloppy-judge``: 68% accurate and biased toward PASS (the rubber-stamp
     failure mode: high FPR) → NEEDS_HUMAN, gate blocks.
3. The loop: unlabeled synthetic production traces (no outcome attributes at
   all) + the licensed judge's verdicts → pass^k reliability report. The same
   verdicts presented with the sloppy judge's license are REFUSED — the gate
   working in the data path, not just in a dashboard.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from rich.console import Console

from keen_touchstone.adapters.otel_traces import read_spans_jsonl, trials_from_traces
from keen_touchstone.aggregate import build_suite_result
from keen_touchstone.artifacts import EvalVerdict, JudgeCalibration
from keen_touchstone.judge.anchors import read_anchors
from keen_touchstone.judge.calibrate import calibrate
from keen_touchstone.judge.verdicts import outcomes_from_verdicts, read_verdicts
from keen_touchstone.report import RunMeta, emit
from keen_touchstone.report.judge_emit import emit_license

DEMO_TASK_TEXTS = [
    "refund order #{i} and confirm by email",
    "rebook flight for booking {i} to the next day",
    "reconcile invoice {i} against the ledger",
    "summarize incident {i} for the postmortem",
    "find the top competitor mentions for product {i}",
]


def write_demo_anchors(path: Path, n_items: int = 60, seed: int = 2026) -> tuple[Path, dict[str, bool]]:
    """Simulated exam paper: latent truth + three noisy human annotators.

    The labels are stamped label_source="human" because they PLAY the humans
    in this demo — the console says so out loud."""
    rng = random.Random(f"anchors:{seed}")
    truth: dict[str, bool] = {}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for i in range(n_items):
            item_id = f"anchor-{i:03d}"
            really_passed = rng.random() < 0.55
            truth[item_id] = really_passed
            labels = {
                name: (really_passed if rng.random() < 0.95 else not really_passed)
                for name in ("annotator-a", "annotator-b", "annotator-c")
            }
            template = DEMO_TASK_TEXTS[i % len(DEMO_TASK_TEXTS)]
            fh.write(json.dumps({
                "item_id": item_id,
                "input": template.format(i=i),
                "output": f"(agent transcript for case {i})",
                "human_labels": labels,
                "label_source": "human",
            }) + "\n")
    return path, truth


def synth_judge_labels(
    truth: dict[str, bool], fnr: float, fpr: float, abstain: float, seed: str
) -> dict[str, bool | None]:
    """A simulated judge designed by its confusion matrix: misses real passes
    with p=fnr, rubber-stamps real failures with p=fpr, abstains with
    p=abstain. (Designing by error rates, not 'accuracy', is what makes the
    sloppy judge honestly sloppy — a random-guess branch flatters itself at
    high base rates.)"""
    rng = random.Random(seed)
    labels: dict[str, bool | None] = {}
    for item_id, actual in truth.items():
        if rng.random() < abstain:
            labels[item_id] = None
        elif actual:
            labels[item_id] = not (rng.random() < fnr)
        else:
            labels[item_id] = rng.random() < fpr
    return labels


def run_judge_demo(out: Path, n_items: int = 60, seed: int = 2026, console: Console | None = None) -> dict[str, JudgeCalibration]:
    console = console or Console()
    console.print(
        "[dim]all synthetic, seeded, zero API keys — three simulated annotators play the "
        "humans; swap in your own anchors.jsonl for the real thing[/dim]\n"
    )

    anchors_path, anchor_truth = write_demo_anchors(out / "anchors.jsonl", n_items, seed)
    anchors = read_anchors(anchors_path)

    judges = {
        "good-judge": synth_judge_labels(anchor_truth, fnr=0.05, fpr=0.04, abstain=0.03, seed=f"good:{seed}"),
        "sloppy-judge": synth_judge_labels(anchor_truth, fnr=0.10, fpr=0.55, abstain=0.05, seed=f"sloppy:{seed}"),
    }
    licenses: dict[str, JudgeCalibration] = {}
    for judge_id, labels in judges.items():
        calibration, exam_result, alt = calibrate(
            anchors, labels, judge_id=judge_id, judge_model="(simulated)", seed=seed
        )
        emit_license(calibration, exam_result, alt, out / judge_id, console=console)
        licenses[judge_id] = calibration

    # ---- the loop: unlabeled traces + licensed verdicts → pass^k ----------
    console.print("\n[bold]the loop:[/bold] unlabeled production traces + licensed judge → pass^k")
    from keen_touchstone.demo.tracegen import gen_spans_with_truth, write_spans_jsonl

    spans, trace_truth = gen_spans_with_truth(
        seed=seed, include_outcomes=False, unsigned_runs=0
    )
    traces_path = write_spans_jsonl(out / "traces.unlabeled.jsonl", spans)

    good = licenses["good-judge"]
    judge_rng = random.Random(f"loop:{seed}")
    verdicts_path = out / "verdicts.good-judge.jsonl"
    with open(verdicts_path, "w") as fh:
        for i, (trace_id, actual) in enumerate(sorted(trace_truth.items())):
            verdict = EvalVerdict(
                verdict_id=f"v-{i:04d}",
                trace_id=trace_id,
                scorer_id="good-judge",
                scorer_version="demo-1",
                scorer_kind="model_graded",
                tier="T2_ungrounded",
                value=(actual if judge_rng.random() < 0.93 else not actual),
                judge_model="(simulated)",
                judge_calibration_ref=good.calibration_id,
            )
            fh.write(json.dumps(verdict.to_schema_dict()) + "\n")

    verdicts = read_verdicts(verdicts_path)
    outcomes = outcomes_from_verdicts(verdicts, good)
    ingested = trials_from_traces(read_spans_jsonl(traces_path), outcome_overrides=outcomes)
    result = build_suite_result(
        ingested.tasks,
        context="online",
        model=ingested.model,
        agent_config_hash=ingested.agent_config_hash,
        task_key_source="declared_tag",
        seed=seed,
    )
    result.warnings.extend(ingested.warnings)
    result.warnings.append(
        f"outcomes judged by licensed judge {good.judge_id} "
        f"(κ={good.kappa:.2f}, license {good.calibration_id}) — not by ground truth"
    )
    emit(
        result,
        out / "loop",
        RunMeta(
            source=f"unlabeled traces + licensed judge verdicts ({len(verdicts)} runs)",
            task_name="(derived from traces)",
            model=ingested.model,
            scorer=f"licensed judge {good.judge_id}",
        ),
        console=console,
    )

    # ---- the blocked path: same mechanics, unlicensed judge ---------------
    console.print("\n[bold]the blocked path:[/bold] the same verdicts under the sloppy judge's license")
    sloppy = licenses["sloppy-judge"]
    try:
        outcomes_from_verdicts(verdicts, sloppy)
    except ValueError as err:
        console.print(f"  [red]refused:[/red] {err}")
    return licenses
