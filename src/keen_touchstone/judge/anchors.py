"""The anchor set: the judge's exam paper, human-labeled by construction.

``anchors.jsonl`` — one item per line:

    {"item_id": "run-042", "input": "<what the agent was asked>",
     "output": "<what the agent produced>",
     "human_labels": {"keen": true, "reviewer2": false},
     "label_source": "human",
     "trace_id": "abc123"}            # optional; joins back to Phase 1 traces

Hard rules enforced here (not merely documented):

- ``label_source`` MUST be the literal string "human" on every item. This is
  the anti-circularity wall: auto-generated labels may expand an eval set,
  but they may never certify the judge that generated them.
- ``human_labels`` maps annotator name -> boolean. One annotator is fine
  (κ path); three or more unlocks the alt-test.
- Consensus per item = majority vote; exact ties are EXCLUDED from the κ exam
  (counted and reported) — a tied panel is not ground truth.

Anthropic's practical guidance for building one: 20–50 items drawn from real
failures; this tool withholds κ below 30 (thresholds.min_items).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AnchorItem:
    item_id: str
    input: str
    output: str
    human_labels: Mapping[str, bool]
    trace_id: str | None = None


@dataclass(frozen=True)
class AnchorSet:
    items: list[AnchorItem]
    annotators: list[str]  # sorted union of annotator names
    path: str | None = None
    warnings: list[str] = field(default_factory=list)

    def consensus_labels(self) -> tuple[list[AnchorItem], list[bool], list[str]]:
        """Majority-vote labels; returns (items_kept, labels, tied_item_ids)."""
        kept: list[AnchorItem] = []
        labels: list[bool] = []
        ties: list[str] = []
        for item in self.items:
            votes = list(item.human_labels.values())
            n_pass = sum(votes)
            if n_pass * 2 == len(votes):
                ties.append(item.item_id)
                continue
            kept.append(item)
            labels.append(n_pass * 2 > len(votes))
        return kept, labels, ties

    def annotator_matrix(self) -> dict[str, list[bool | None]]:
        """Per-annotator labels over ALL items (None where not labeled) — the
        alt-test's input shape."""
        return {
            name: [
                item.human_labels.get(name) for item in self.items
            ]
            for name in self.annotators
        }


def read_anchors(path: str | Path) -> AnchorSet:
    path = Path(path)
    items: list[AnchorItem] = []
    seen_ids: set[str] = set()
    annotators: set[str] = set()
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"{path}:{lineno}: not valid JSON ({err.msg})") from err

            source = raw.get("label_source")
            if source != "human":
                raise ValueError(
                    f"{path}:{lineno}: label_source is {source!r}, not 'human' — "
                    "anti-circularity rule: only human-confirmed labels may certify a judge "
                    "(auto-generated labels may expand an eval set, never license the judge)"
                )
            item_id = raw.get("item_id")
            if not item_id or not isinstance(item_id, str):
                raise ValueError(f"{path}:{lineno}: missing/invalid item_id")
            if item_id in seen_ids:
                raise ValueError(f"{path}:{lineno}: duplicate item_id {item_id!r}")
            labels = raw.get("human_labels")
            if not isinstance(labels, dict) or not labels:
                raise ValueError(f"{path}:{lineno}: human_labels must be a non-empty object")
            clean: dict[str, bool] = {}
            for name, value in labels.items():
                if not isinstance(value, bool):
                    raise ValueError(
                        f"{path}:{lineno}: human_labels[{name!r}] must be true/false "
                        f"(got {value!r}) — grade pass/fail, not scores"
                    )
                clean[str(name)] = value
            seen_ids.add(item_id)
            annotators.update(clean)
            items.append(
                AnchorItem(
                    item_id=item_id,
                    input=str(raw.get("input", "")),
                    output=str(raw.get("output", "")),
                    human_labels=clean,
                    trace_id=raw.get("trace_id"),
                )
            )
    if not items:
        raise ValueError(f"{path}: no anchor items found")

    warnings: list[str] = []
    if len(items) < 30:
        warnings.append(
            f"only {len(items)} anchor items — κ will be withheld below 30; aim for 20–50 "
            "items drawn from real failures, then keep labeling"
        )
    return AnchorSet(
        items=items, annotators=sorted(annotators), path=str(path), warnings=warnings
    )


def read_judge_labels(path: str | Path) -> dict[str, bool | None]:
    """BYO judge answers: {"item_id": ..., "judge_label": true|false|"unknown"}."""
    path = Path(path)
    labels: dict[str, bool | None] = {}
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"{path}:{lineno}: not valid JSON ({err.msg})") from err
            item_id = raw.get("item_id")
            if not item_id:
                raise ValueError(f"{path}:{lineno}: missing item_id")
            if str(item_id) in labels:
                # [xfam X4] last-row-wins silently rewrote earlier labels —
                # anchors and verdicts already reject duplicates; so does this
                raise ValueError(
                    f"{path}:{lineno}: duplicate judge_label for item_id {item_id!r} — "
                    "one label per anchor item"
                )
            value = raw.get("judge_label")
            if isinstance(value, bool) or value is None:
                labels[str(item_id)] = value
            elif isinstance(value, str) and value.lower() in ("unknown", "unsure", "abstain"):
                labels[str(item_id)] = None
            else:
                raise ValueError(
                    f"{path}:{lineno}: judge_label must be true/false/\"unknown\" (got {value!r})"
                )
    if not labels:
        raise ValueError(f"{path}: no judge labels found")
    return labels
