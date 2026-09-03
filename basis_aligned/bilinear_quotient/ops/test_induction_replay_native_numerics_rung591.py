"""CPU-only owner tests for the prospective R591 diagnostic."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).with_name("induction_replay_native_numerics_rung591.py")
DRYRUN = SCRIPT.parents[1] / "induction_replay_native_numerics_rung591_dryrun.json"


def load_runner():
    name = "r591_owner_test_target"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return load_runner()


@pytest.fixture(scope="module")
def execution(runner):
    _, authority = runner.load_authority()
    return authority


def test_exact_sources_and_prospective_amendment_are_hash_pinned(runner):
    assert runner.verify_sources() == {
        str(path): digest for path, digest in runner.SOURCE_HASHES.items()
    }
    assert runner.SOURCE_HASHES[runner.R585] == (
        "fd772c3b9d6df4271ecbfc90c00c893db5a65ea06601f0c8f6e7a9e34c9a531b"
    )
    assert runner.SOURCE_HASHES[runner.FACADE] == (
        "b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c"
    )
    assert runner.SOURCE_HASHES[runner.INDUCTION] == (
        "b2d43be8e260bbe4bfece494999d237d93258f676b19e2993eca09655e253e3a"
    )
    assert runner.SOURCE_HASHES[runner.METHOD_HANDOFF_V5] == (
        "810d15aa7f86a9896ca56e48c7ea33c60b10f6b0d266acefa5f3441333c8fe80"
    )
    assert runner.SOURCE_HASHES[runner.PREREGISTRATION] == (
        "e72cb386d65c68f55b767c8141c3c4d774b3c8ad9387ac7f8ad43bebef118593"
    )


def test_source_gate_fails_closed_on_tampering(runner, tmp_path, monkeypatch):
    source = tmp_path / "authority.txt"
    source.write_bytes(b"frozen")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(runner, "SOURCE_HASHES", {source: digest})
    assert runner.verify_sources() == {str(source): digest}
    source.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="frozen source mismatch"):
        runner.verify_sources()


def test_split_length_histograms_and_impossible_original_panel(runner, execution):
    from collections import Counter

    observed = {
        split: dict(sorted(Counter(
            int(row["length"])
            for row in execution["endpoints"] if row["split"] == split
        ).items()))
        for split in ("FIT", "SELECT")
    }
    assert observed == runner.EXPECTED_LENGTH_HISTOGRAMS
    assert set(observed["FIT"]) == {19, 20, 27, 28}
    assert set(observed["SELECT"]) == {21, 22, 29, 30}
    with pytest.raises(RuntimeError, match="insufficient FIT endpoints at length 21"):
        runner.select_panel_for_lengths(execution["endpoints"], runner.ALL_LENGTHS, 32)


def test_exact_256_panel_membership_and_hash(runner, execution):
    panel = runner.select_panel(execution["endpoints"])
    assert len(panel) == len({row["endpoint_id"] for row in panel}) == 256
    assert {row["split"] for row in panel} == {"FIT"}
    assert {
        length: sum(int(row["length"]) == length for row in panel)
        for length in runner.FIT_LENGTHS
    } == {19: 64, 20: 64, 27: 64, 28: 64}
    membership = [
        {"length": int(row["length"]), "endpoint_id": str(row["endpoint_id"])}
        for row in panel
    ]
    assert runner.content_sha256(membership) == runner.EXPECTED_PANEL_SHA256 == (
        "6b56a6740dbea7d0765d6a8668361ff43b06562152f091f6969ca8591522ebe4"
    )


def test_semantic_roles_exhaust_canonical_equality_support(runner, execution):
    audit = runner.audit_equality_support(execution["endpoints"])
    assert audit == {
        "endpoint_count": 2_592,
        "canonical_support_count_histogram": {"0": 432, "1": 2_160},
        "extra_position_count": 0,
        "missing_position_count": 0,
        "ordered_census_sha256": (
            "e2de29dcf3cb37187060ab72775533086612bbb349777d48bd9f8feb8911e9fa"
        ),
    }
    planted = [dict(row) for row in execution["endpoints"]]
    planted[0] = dict(planted[0])
    planted[0]["payload_positions"] = list(planted[0]["payload_positions"][:-1])
    planted[0]["source_positions"] = list(planted[0]["source_positions"][:-1])
    with pytest.raises(RuntimeError, match="support census changed"):
        runner.audit_equality_support(planted)


def test_all_schedule_sizes_padding_and_membership(runner, execution):
    full = runner.full_fit_schedules(execution["endpoints"])
    assert set(full) == {"M", "L"}
    assert len(full["M"]) == len(full["L"]) == 54
    assert all(len(batch["records"]) == 32 for batches in full.values() for batch in batches)
    assert all(
        batch["padding_length"] == max(row["length"] for row in batch["records"])
        for batches in full.values() for batch in batches
    )

    panel = runner.select_panel(execution["endpoints"])
    schedules = runner.panel_schedules(panel)
    assert set(schedules) == set(runner.PANEL_SCHEDULES)
    assert all(len(batches) == 8 for batches in schedules.values())
    expected = {row["endpoint_id"] for row in panel}
    for name, batches in schedules.items():
        assert all(len(batch["records"]) == 32 for batch in batches)
        assert {row["endpoint_id"] for batch in batches for row in batch["records"]} == expected
        if name != "L_native":
            assert {batch["padding_length"] for batch in batches} == {30}
    for index, batch in enumerate(schedules["M_30"]):
        assert [
            sum(int(row["length"]) == length for row in batch["records"])
            for length in runner.FIT_LENGTHS
        ] == [8, 8, 8, 8]
        for length in runner.FIT_LENGTHS:
            lexical = sorted(
                (row for row in panel if int(row["length"]) == length),
                key=lambda row: str(row["endpoint_id"]),
            )
            observed = [row for row in batch["records"] if int(row["length"]) == length]
            assert observed == lexical[8 * index:8 * index + 8]


def test_exact_forward_and_factor_operation_census(runner, execution):
    manifest = runner.build_call_manifest(execution["endpoints"])
    assert len(manifest) == 234
    assert runner.content_sha256(manifest) == runner.EXPECTED_CALL_MANIFEST_SHA256 == (
        "1e838190752e72eed6f35119c3e99bfb7620e787ae73c7a052046160d600ad3f"
    )
    census = runner.operation_census(manifest)
    assert census == {
        "model_forwards": 234,
        "model_backwards": 0,
        "model_weights_updated": False,
        "endpoint_forwards": 7_488,
        "dispatcher_forward_counts": {"N": 132, "F": 24, "R": 78},
        "factor_endpoint_site_operations": 13_056,
        "factor_endpoint_site_role_operations": 26_112,
        "factor_operations_by_site": {
            "L5H5": 3_264, "L7H3": 3_264, "L8H3": 3_264, "L8H4": 3_264,
        },
        "evaluated_splits": ["FIT"],
        "forbidden_splits_opened": [],
    }


def test_v4_forward_shapes_are_dynamic_and_facade_valid(runner, execution):
    torch = pytest.importorskip("torch")
    manifest = runner.build_call_manifest(execution["endpoints"])
    contract = runner.forward_shape_contract(manifest)
    assert contract["call_count"] == 234
    assert contract["all_batch_sizes"] == [32]
    assert contract["all_padding_lengths"] == [19, 20, 27, 28, 30]
    assert contract["facade_require_production"] is False
    facade = runner.load_module(runner.FACADE, "r591_shape_test_facade")
    for shape in sorted({(row["batch_size"], row["padding_length"]) for row in manifest}):
        tokens = torch.full(shape, runner.PAD_TOKEN, dtype=torch.long)
        facade.validate_tokens(tokens, production_shape=False)
        with pytest.raises(RuntimeError, match="tokens must have shape"):
            facade.validate_tokens(tokens, production_shape=True)
    full = runner.full_fit_schedules(execution["endpoints"])
    panel = runner.panel_schedules(runner.select_panel(execution["endpoints"]))
    all_batches = [
        batch for batches in (*full.values(), *panel.values()) for batch in batches
    ]
    for batch in all_batches:
        tokens = runner._padded_tokens(batch, torch=torch, device="cpu")
        facade.validate_tokens(tokens, production_shape=False)
        assert tuple(tokens.shape) == (32, batch["padding_length"])
        for row_index, row in enumerate(batch["records"]):
            width = len(row["token_ids"])
            assert tokens[row_index, :width].tolist() == row["token_ids"]
            assert set(tokens[row_index, width:].tolist()) <= {runner.PAD_TOKEN}


def test_exact_dispatcher_inventory_and_factor_writes(runner):
    torch = pytest.importorskip("torch")
    assert runner.DISPATCHERS == ("N", "F", "R")
    native = torch.zeros(1, 4, 1152)
    batch = [{"final_position": 3}]
    terms = [{
        name: {
            "term": torch.full((1152,), float(index + 2)),
            "canonical": torch.full((1152,), float(index)),
        }
        for index, name in enumerate(runner.TERM_NAMES)
    }]
    observed = runner.resolve_factor_write("F", native, terms, batch, torch=torch)
    assert observed is native
    replay = runner.resolve_factor_write("R", native, terms, batch, torch=torch)
    assert replay is not native
    assert torch.equal(native, torch.zeros_like(native))
    assert torch.equal(replay[0, :3], torch.zeros_like(replay[0, :3]))
    assert torch.equal(replay[0, 3], torch.full((1152,), 8.0))
    with pytest.raises(ValueError, match="only for F/R"):
        runner.resolve_factor_write("N", native, terms, batch, torch=torch)


def test_runtime_has_one_facade_path_no_direct_model_or_publication_calls(runner):
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    collector = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef)
        and node.name == "collect_condition"
    )
    facade_calls = [
        node for node in ast.walk(collector) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "forward_with_dispatch"
    ]
    assert len(facade_calls) == 1
    keyword = next(item for item in facade_calls[0].keywords if item.arg == "require_production")
    assert isinstance(keyword.value, ast.Constant) and keyword.value.value is False
    assert not [
        node for node in ast.walk(collector) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id in {"model", "forward"}
    ]
    forbidden = {
        "score_split", "select_candidate", "decide_terminal", "publish_result",
        "write_text", "write_bytes", "write", "save", "dump",
    }
    called = {
        node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
    } | {
        node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }
    assert called.isdisjoint(forbidden)
    source = SCRIPT.read_text(encoding="utf-8")
    assert "induction_selector_payload_frozen_factor_rung585_results.json" not in source
    assert "induction_selector_payload_frozen_factor_rung585_receipt.json" not in source
    assert "induction_selector_payload_frozen_factor_rung585_evidence" not in source


def test_checkpoint_and_model_gates_are_literal(runner):
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    runtime = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef)
        and node.name == "run_diagnostic"
    )
    loads = [
        node for node in ast.walk(runtime) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == "load_bilin18"
    ]
    assert len(loads) == 1
    keywords = {item.arg: item.value for item in loads[0].keywords}
    assert isinstance(keywords["device"], ast.Constant) and keywords["device"].value == "cuda"
    assert isinstance(keywords["verify_weights_sha256"], ast.Constant)
    assert keywords["verify_weights_sha256"].value is True
    assert any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "validate_production_model"
        for node in ast.walk(runtime)
    )
    assert runner.CHECKPOINT_SHA256 == (
        "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
    )


def test_comparison_and_interpretation_use_frozen_absolute_threshold(runner):
    torch = pytest.importorskip("torch")

    def cell(value):
        return {
            "logits": {"x": torch.tensor([value, 0.0])},
            "locations": {"x": {
                "endpoint_id": "x", "length": 19, "batch_index": 0, "row_index": 0,
            }},
        }

    summary = runner.difference_summary(cell(2e-5), cell(0.0), torch=torch)
    assert summary["max_abs"] == pytest.approx(2e-5)
    assert summary["endpoints_over_1e_5"] == 1
    zero = runner.difference_summary(cell(0.0), cell(0.0), torch=torch)
    comparisons = {
        "full_fit": {"total": summary, "hook": summary},
        "panel": {
            "observer": {name: zero for name in runner.PANEL_SCHEDULES},
            "hook": {name: summary for name in runner.PANEL_SCHEDULES},
            "padding": {name: zero for name in runner.DISPATCHERS},
            "membership": {name: zero for name in runner.DISPATCHERS},
        },
    }
    decision = runner.interpret(comparisons)
    assert decision["classification"] == "hook_dominated"
    assert decision["threshold_unchanged"] == 1e-5
    assert decision["licenses_r585_science"] is False


def test_strict_finite_json_and_stdout_only_dryrun(runner):
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="nonfinite"):
            runner.require_finite_json({"bad": bad})
        with pytest.raises(ValueError):
            json.dumps({"bad": bad}, allow_nan=False)
    dryrun = runner.build_dryrun()
    assert dryrun == json.loads(DRYRUN.read_text(encoding="utf-8"))
    assert dryrun["output_boundary"] == {
        "stdout_json_only": True,
        "writes_result": False,
        "writes_receipt": False,
        "writes_evidence": False,
        "calls_scoring": False,
        "calls_selection": False,
        "publishes_scientific_terminal": False,
    }
    assert dryrun["model_forwards"] == dryrun["model_backwards"] == 0
    assert dryrun["model_weights_updated"] is False


def test_managed_dryrun_is_model_free_and_deterministic(runner):
    environment = dict(os.environ, BQLIB_DRYRUN="1", CUDA_VISIBLE_DEVICES="")
    first = subprocess.run(
        [sys.executable, str(SCRIPT)], check=True, capture_output=True, text=True,
        env=environment,
    )
    second = subprocess.run(
        [sys.executable, str(SCRIPT)], check=True, capture_output=True, text=True,
        env=environment,
    )
    assert first.stderr == second.stderr == ""
    assert first.stdout == second.stdout
    assert json.loads(first.stdout) == runner.build_dryrun()


def test_invalid_managed_mode_fails_without_model_access():
    environment = dict(os.environ, BQLIB_DRYRUN="true", CUDA_VISIBLE_DEVICES="")
    process = subprocess.run(
        [sys.executable, str(SCRIPT)], capture_output=True, text=True, env=environment,
    )
    assert process.returncode != 0
    assert "must be absent or exactly '1'" in process.stderr
