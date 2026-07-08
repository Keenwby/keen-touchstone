"""Anchor set parsing (anti-circularity), judge runner (mockllm), calibrate CLI."""

import json

import pytest

from keen_touchstone.judge.anchors import read_anchors, read_judge_labels


def _write_anchors(path, items):
    with open(path, "w") as fh:
        for item in items:
            fh.write(json.dumps(item) + "\n")
    return path


def _item(i, labels, source="human", **extra):
    return {
        "item_id": f"item-{i:03d}", "input": f"task {i}", "output": f"result {i}",
        "human_labels": labels, "label_source": source, **extra,
    }


def test_read_anchors_happy_path(tmp_path) -> None:
    path = _write_anchors(
        tmp_path / "a.jsonl",
        [_item(i, {"keen": i % 3 != 0, "amy": i % 2 == 0}) for i in range(40)],
    )
    anchors = read_anchors(path)
    assert len(anchors.items) == 40
    assert anchors.annotators == ["amy", "keen"]
    assert anchors.warnings == []  # >= 30 items
    matrix = anchors.annotator_matrix()
    assert len(matrix["keen"]) == 40


def test_anti_circularity_rejects_non_human_labels(tmp_path) -> None:
    path = _write_anchors(
        tmp_path / "bad.jsonl",
        [_item(0, {"keen": True}), _item(1, {"gpt": True}, source="llm_generated")],
    )
    with pytest.raises(ValueError, match="anti-circularity"):
        read_anchors(path)


def test_anchors_validation_errors(tmp_path) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        read_anchors(_write_anchors(
            tmp_path / "dup.jsonl", [_item(1, {"k": True}), _item(1, {"k": False})]
        ))
    with pytest.raises(ValueError, match="true/false"):
        read_anchors(_write_anchors(
            tmp_path / "score.jsonl", [_item(1, {"k": 0.7})]
        ))
    with pytest.raises(ValueError, match="non-empty"):
        read_anchors(_write_anchors(tmp_path / "empty-labels.jsonl", [_item(1, {})]))


def test_consensus_majority_and_ties() -> None:
    from keen_touchstone.judge.anchors import AnchorItem, AnchorSet

    items = [
        AnchorItem("a", "", "", {"h1": True, "h2": True, "h3": False}),
        AnchorItem("b", "", "", {"h1": True, "h2": False}),  # tie
        AnchorItem("c", "", "", {"h1": False}),
    ]
    anchors = AnchorSet(items=items, annotators=["h1", "h2", "h3"])
    kept, labels, ties = anchors.consensus_labels()
    assert [i.item_id for i in kept] == ["a", "c"]
    assert labels == [True, False]
    assert ties == ["b"]


def test_small_anchor_set_warns(tmp_path) -> None:
    path = _write_anchors(tmp_path / "s.jsonl", [_item(i, {"k": True}) for i in range(10)])
    anchors = read_anchors(path)
    assert any("withheld below 30" in w for w in anchors.warnings)


def test_read_judge_labels_variants(tmp_path) -> None:
    path = tmp_path / "j.jsonl"
    path.write_text(
        '{"item_id": "a", "judge_label": true}\n'
        '{"item_id": "b", "judge_label": false}\n'
        '{"item_id": "c", "judge_label": "unknown"}\n'
        '{"item_id": "d", "judge_label": null}\n'
    )
    labels = read_judge_labels(path)
    assert labels == {"a": True, "b": False, "c": None, "d": None}
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"item_id": "a", "judge_label": 0.9}\n')
    with pytest.raises(ValueError, match="judge_label"):
        read_judge_labels(bad)


def test_runner_with_mockllm_parses_all_verdict_shapes(tmp_path) -> None:
    from inspect_ai.model import ModelOutput

    from keen_touchstone.judge.runner import JUDGE_PROMPT_HASH, run_judge

    path = _write_anchors(tmp_path / "a.jsonl", [_item(i, {"k": True}) for i in range(5)])
    anchors = read_anchors(path)
    replies = ["PASS", "fail.", "**PASS**", "UNKNOWN", "the weather is nice"]
    outs = [ModelOutput.from_content("mockllm/model", r) for r in replies]
    run = run_judge(anchors, model="mockllm/model", model_args={"custom_outputs": outs})
    assert run.prompt_hash == JUDGE_PROMPT_HASH
    ids = [f"item-{i:03d}" for i in range(5)]
    assert [run.labels[i] for i in ids] == [True, False, True, None, None]
    assert run.n_unparseable == 1
    assert any("unparseable" in w for w in run.warnings)


def test_cli_calibrate_and_gate_end_to_end(tmp_path) -> None:
    from click.testing import CliRunner

    from keen_touchstone.cli import main

    # 40 items, single annotator; judge agrees on ~90%
    items = [_item(i, {"keen": i % 5 != 0}) for i in range(40)]
    anchors_path = _write_anchors(tmp_path / "anchors.jsonl", items)
    judge_path = tmp_path / "judge.jsonl"
    with open(judge_path, "w") as fh:
        for i, item in enumerate(items):
            label = item["human_labels"]["keen"]
            if i in (3, 7, 11):  # a few disagreements
                label = not label
            fh.write(json.dumps({"item_id": item["item_id"], "judge_label": label}) + "\n")

    out = tmp_path / "judge-out"
    runner = CliRunner()
    result = runner.invoke(main, [
        "judge", "calibrate", str(anchors_path),
        "--judge-labels", str(judge_path), "--out", str(out),
    ])
    assert result.exit_code == 0, result.output
    license_path = out / "license.json"
    assert license_path.exists()
    data = json.loads(license_path.read_text())
    assert data["status"] == "JUDGE_LICENSED"
    assert (out / "judge-report.html").exists()

    gate_ok = runner.invoke(main, ["judge", "gate", str(license_path)])
    assert gate_ok.exit_code == 0, gate_ok.output

    # tamper the license -> schema-invalid -> clean failure
    data["status"] = "SUPER_LICENSED"
    license_path.write_text(json.dumps(data))
    gate_bad = runner.invoke(main, ["judge", "gate", str(license_path)])
    assert gate_bad.exit_code == 3  # r4-F3: invalid file = domain error, not a gate verdict
    assert "not a valid license" in gate_bad.output


def test_cli_calibrate_blocks_bad_judge_with_exit_1(tmp_path) -> None:
    from click.testing import CliRunner

    from keen_touchstone.cli import main

    items = [_item(i, {"keen": i % 2 == 0}) for i in range(40)]
    anchors_path = _write_anchors(tmp_path / "anchors.jsonl", items)
    judge_path = tmp_path / "judge.jsonl"
    with open(judge_path, "w") as fh:
        for item in items:  # judge always says pass -> kappa 0
            fh.write(json.dumps({"item_id": item["item_id"], "judge_label": True}) + "\n")

    result = CliRunner().invoke(main, [
        "judge", "calibrate", str(anchors_path),
        "--judge-labels", str(judge_path), "--out", str(tmp_path / "o"),
    ])
    assert result.exit_code == 1
    data = json.loads((tmp_path / "o" / "license.json").read_text())
    assert data["status"] == "NEEDS_HUMAN"
