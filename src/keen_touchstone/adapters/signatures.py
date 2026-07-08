"""Derived task identity (SPEC §5) — the keystone of online pass^k.

Production traffic carries no exam numbers: to say "this task succeeded 7 of
9 times" you must first decide which runs ARE the same task. v0.4 ships two
DETERMINISTIC, INSPECTABLE strategies (embedding clustering is deliberately
deferred: it fails SPEC §5's "stable" and "inspectable" requirements today):

- **TemplateStrategy** — normalize the run's input text by masking the parts
  that vary per run (numbers, ids, emails, URLs, dates); the masked template
  IS the signature. A human can read exactly why two runs grouped.
- **ToolSequenceStrategy** — the ordered sequence of tool names the run
  invoked (consecutive repeats collapsed): a structural signature for runs
  whose inputs vary too much to template.

Non-negotiable honesty rule: **derived grouping is a hypothesis, not ground
truth.** Every readout carries per-cluster exemplars, a singleton rate, and —
when human-declared tags are also present — the weighted purity against them.
The caveat string ships in the output; it is not optional.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .otel_traces import Span, TraceRun

HYPOTHESIS_CAVEAT = (
    "derived task grouping is a HYPOTHESIS, not ground truth — inspect the exemplars "
    "before trusting any per-task reliability number built on it"
)

# masking order matters: specific patterns before the generic digit-mask
_MASKS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"https?://\S+"), "<url>"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "<email>"),
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I), "<id>"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b"), "<date>"),
    (re.compile(r"\b[A-Za-z]{1,6}-?\d[\w-]*\b"), "<id>"),  # INV-007, svc-42, C812, P99
    (re.compile(r"\b[0-9a-f]{7,}\b", re.I), "<id>"),  # bare hex runs: git SHAs etc. (r4-F5)
    (re.compile(r"#?\d[\d,.]*"), "<num>"),
]
# Known mask limits (r4-F5, documented not hidden): plural/singular variants
# ("1 subscription" vs "3 subscriptions") and CJK-adjacent Latin ids (no \b
# boundary between CJK and \w) still split clusters. The shred-guard note in
# the readout is the runtime safety net for such gaps.
_WS = re.compile(r"\s+")


def normalize_template(text: str) -> str:
    out = text.lower()
    for pattern, token in _MASKS:
        out = pattern.sub(token, out)
    return _WS.sub(" ", out).strip()


def _root_input_text(run: TraceRun) -> str | None:
    value = run.attr("input.messages")
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    if isinstance(value, list):  # message list: concatenate string-ish content
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str):
                    parts.append(content)
        return " ".join(parts) or None
    return str(value)


class TemplateStrategy:
    """Signature = hash of the masked input template. Stateful: keeps the
    human-readable template and up-to-3 raw exemplars per cluster for the
    readout (the inspectability requirement, made concrete)."""

    name = "template"

    def __init__(self) -> None:
        self.templates: dict[str, str] = {}
        self.exemplars: dict[str, list[str]] = {}

    def signature(self, run: TraceRun) -> str | None:
        text = _root_input_text(run)
        if not text:
            return None
        template = normalize_template(text)
        if not template:
            return None
        sig = "tpl:" + hashlib.sha256(template.encode()).hexdigest()[:10]
        self.templates.setdefault(sig, template)
        bucket = self.exemplars.setdefault(sig, [])
        if len(bucket) < 3:
            bucket.append(text[:160])
        return sig


class ToolSequenceStrategy:
    """Signature = the run's ordered tool names, consecutive repeats collapsed."""

    name = "tool_sequence"

    def __init__(self) -> None:
        self.templates: dict[str, str] = {}
        self.exemplars: dict[str, list[str]] = {}

    def signature(self, run: TraceRun) -> str | None:
        tool_spans = sorted(
            (s for s in run.spans if s.get("gen_ai.operation.name") == "execute_tool"),
            key=lambda s: s.get("harness.step_id", 0),
        )
        names: list[str] = []
        for span in tool_spans:
            name = span.get("gen_ai.tool.name")
            if name and (not names or names[-1] != name):
                names.append(str(name))
        if not names:
            return None
        sequence = ">".join(names)
        sig = "tools:" + hashlib.sha256(sequence.encode()).hexdigest()[:10]
        self.templates.setdefault(sig, sequence)
        bucket = self.exemplars.setdefault(sig, [])
        if len(bucket) < 3:
            text = _root_input_text(run)
            bucket.append((text or f"(run {run.trace_id[:12]})")[:160])
        return sig


STRATEGIES = {"template": TemplateStrategy, "tool-sequence": ToolSequenceStrategy}


@dataclass(frozen=True)
class ClusterInfo:
    signature: str
    size: int
    template: str | None  # the readable why-these-grouped string
    exemplars: list[str]
    majority_declared: str | None  # dominant human tag inside this cluster, if any


@dataclass(frozen=True)
class GroupingReadout:
    """SPEC §5's mandated quality readout for any derived grouping."""

    strategy: str
    n_runs: int
    n_grouped: int
    n_unsigned: int  # runs the strategy could not sign (excluded honestly)
    n_clusters: int
    singleton_rate: float
    largest_cluster_share: float
    purity_vs_declared: float | None  # weighted majority share, when tags exist
    declared_coverage: float | None  # share of grouped runs that carried a tag
    clusters: list[ClusterInfo] = field(default_factory=list)
    caveat: str = HYPOTHESIS_CAVEAT
    shred_warning: str | None = None
    """r4-F5: purity is trivially perfect under total under-merge (every run
    its own cluster). When the singleton share is high, purity carries no
    information and this says so."""

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def runs_from_spans(spans: list[Span]) -> list[TraceRun]:
    """Group spans into runs, PRESERVING first-seen (file) order.

    Review r4-F1: this used to sort by trace_id — lexicographic order over
    random ids, i.e. a temporal SCRAMBLE. An untimestamped stream fed to
    `watch` had its drift smeared across every window and the breach never
    fired. Dict insertion order is the file order; `watch` sorts runs that
    carry parseable timestamps ahead of that."""
    by_trace: dict[str, list[Span]] = {}
    for span in spans:
        by_trace.setdefault(str(span["trace_id"]), []).append(span)
    return [TraceRun(trace_id=tid, spans=s) for tid, s in by_trace.items()]


def grouping_readout(
    spans: list[Span],
    strategy: TemplateStrategy | ToolSequenceStrategy,
    max_clusters_listed: int = 12,
) -> GroupingReadout:
    """Run the strategy over all runs and produce the honesty readout.

    Uses a FRESH pass (independent of ingest) so the readout reflects exactly
    what the strategy did, including the runs it refused to sign.
    """
    runs = runs_from_spans(spans)
    assignments: dict[str, list[TraceRun]] = {}
    unsigned = 0
    for run in runs:
        sig = strategy.signature(run)
        if sig is None:
            unsigned += 1
            continue
        assignments.setdefault(sig, []).append(run)

    n_grouped = sum(len(v) for v in assignments.values())
    sizes = [len(v) for v in assignments.values()]

    declared_seen = 0
    purity_weighted = 0.0
    clusters: list[ClusterInfo] = []
    for sig, members in sorted(assignments.items(), key=lambda kv: -len(kv[1])):
        declared = [
            str(m.attr("harness.task_signature"))
            for m in members
            if m.attr("harness.task_signature")
        ]
        majority: str | None = None
        if declared:
            majority, majority_count = Counter(declared).most_common(1)[0]
            declared_seen += len(declared)
            purity_weighted += majority_count
        clusters.append(
            ClusterInfo(
                signature=sig,
                size=len(members),
                template=strategy.templates.get(sig),
                exemplars=strategy.exemplars.get(sig, []),
                majority_declared=majority,
            )
        )

    singleton_rate = (sizes.count(1) / len(sizes)) if sizes else 0.0
    runs_in_singletons = sum(1 for size in sizes if size == 1)
    shred_warning = None
    if sizes and runs_in_singletons / max(n_grouped, 1) > 0.5:
        shred_warning = (
            f"{runs_in_singletons}/{n_grouped} runs sit in singleton clusters — the grouping "
            "is likely SHREDDING one task into many (mask gaps?); purity numbers are "
            "meaningless in this regime (a fully shredded grouping scores 100% purity)"
        )
    return GroupingReadout(
        strategy=strategy.name,
        n_runs=len(runs),
        n_grouped=n_grouped,
        n_unsigned=unsigned,
        n_clusters=len(assignments),
        singleton_rate=singleton_rate,
        largest_cluster_share=(max(sizes) / n_grouped) if n_grouped else 0.0,
        purity_vs_declared=(purity_weighted / declared_seen) if declared_seen else None,
        declared_coverage=(declared_seen / n_grouped) if n_grouped else None,
        clusters=clusters[:max_clusters_listed],
        shred_warning=shred_warning,
    )
