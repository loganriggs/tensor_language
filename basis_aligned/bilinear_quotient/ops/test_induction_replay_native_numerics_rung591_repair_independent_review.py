"""Independent model/outcome-free review of exact R591 commit a5e1dd022."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


COMMIT = "a5e1dd022729c28dad99c1782f557b3162cdf45e"
OPS = Path(__file__).resolve().parent
ROOT = OPS.parent
REPO = ROOT.parents[1]
POLY = ROOT.parent / "polynomial_causal"

PRODUCER = OPS / "induction_replay_native_numerics_rung591.py"
OWNER = OPS / "test_induction_replay_native_numerics_rung591.py"
DRYRUN = ROOT / "induction_replay_native_numerics_rung591_dryrun.json"
PREREG = POLY / "INDUCTION_REPLAY_NATIVE_NUMERICS_RUNG591_PREREGISTRATION.md"
BUILDER = POLY / "INDUCTION_REPLAY_NATIVE_NUMERICS_RUNG591_BUILDER_HANDOFF.md"
ADAPTER = OPS / "execute_induction_replay_native_numerics_rung591.py"
ADAPTER_TEST = OPS / "test_execute_induction_replay_native_numerics_rung591.py"
ADAPTER_HANDOFF = POLY / "INDUCTION_REPLAY_NATIVE_NUMERICS_RUNG591_MANAGED_ADAPTER_HANDOFF.md"

HASHES = {
    PRODUCER: "fb8239ded4f3e99510f37ea72337c2d69e4640f7a2556748c9062aa82b2751bc",
    OWNER: "8a24a9903d10ada8a4048c7adcb33cb4ef3e8aeef11d6f9718f8e50e57b6212c",
    DRYRUN: "8a6331fb1a4d3800abff5ab6b7e291105872b06b41a43b003436312b6e50dc5d",
    PREREG: "2dd8f918f767a6e5d91af357cfaa14770b79334ebac837d1bf52e8046ce190a5",
    BUILDER: "202f1268e583a82f6cca385f4223b6edf4e8f8bbaee2c1cc975b09e51cd95f12",
    ADAPTER: "b0a0654c4b6fd28a9dfbfb947969049c203ef346cc580f87f5406701ac876d20",
    ADAPTER_TEST: "338dd545838e75ae8de4a8bd6405f4bac601fe2ad8a81f594bab8104151de0ed",
    ADAPTER_HANDOFF: "fab59548fd9529371f06156bbf2f9fa69c2c33a8a41abe2acb47a4780ff0ea96",
}


def _blob(path: Path) -> bytes:
    relative = path.relative_to(REPO)
    return subprocess.check_output(
        ["git", "show", f"{COMMIT}:{relative.as_posix()}"], cwd=REPO
    )


def _load_exact(path: Path, name: str):
    source = _blob(path)
    assert hashlib.sha256(source).hexdigest() == HASHES[path]
    assert path.read_bytes() == source
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


producer = _load_exact(PRODUCER, "r591_repair_independent_producer")
adapter = _load_exact(ADAPTER, "r591_repair_independent_adapter")


@pytest.fixture(scope="module")
def execution():
    _, value = producer.load_authority()
    return value


def _cell(value: float) -> dict[str, float]:
    return {"max_abs": value}


def _comparisons(*, padding=None, membership=None):
    zero = _cell(0.0)
    return {
        "full_fit": {"total": zero, "hook": zero},
        "panel": {
            "observer": {name: zero for name in producer.PANEL_SCHEDULES},
            "hook": {name: zero for name in producer.PANEL_SCHEDULES},
            "padding": {
                name: _cell(2e-5 if name == padding else 0.0)
                for name in producer.DISPATCHERS
            },
            "membership": {
                name: _cell(2e-5 if name == membership else 0.0)
                for name in producer.DISPATCHERS
            },
        },
    }


def test_exact_packet_hashes_are_git_blob_bound():
    for path, expected in HASHES.items():
        assert hashlib.sha256(_blob(path)).hexdigest() == expected


@pytest.mark.parametrize("family,dispatcher", [("padding", "R"), ("membership", "F")])
def test_auxiliary_dispatchers_cannot_assign_native_cause(family, dispatcher):
    kwargs = {family: dispatcher}
    result = producer.interpret(_comparisons(**kwargs))
    assert result["classification"] == "all_registered_components_within_threshold"
    assert result["active_components"] == []


@pytest.mark.parametrize(
    "family,classification", [("padding", "padding_dominated"), ("membership", "membership_gemm_dominated")]
)
def test_native_dispatcher_alone_assigns_registered_numerical_cause(family, classification):
    result = producer.interpret(_comparisons(**{family: "N"}))
    assert result["classification"] == classification
    assert len(result["active_components"]) == 1


def test_emitted_panel_ids_are_exact_unique_fit_support(execution):
    dry = producer.build_dryrun()
    panel = producer.select_panel(execution["endpoints"])
    ids = [str(row["endpoint_id"]) for row in panel]
    receipt = dry["panel"]
    assert receipt["split"] == "FIT"
    assert receipt["ordered_endpoint_ids"] == ids
    assert len(ids) == len(set(ids)) == 256
    assert receipt["ordered_endpoint_ids_sha256"] == producer.content_sha256(ids)
    assert receipt["length_counts"] == {"19": 64, "20": 64, "27": 64, "28": 64}


def test_dryrun_call_graph_does_not_read_r586_or_r587_outcomes(monkeypatch):
    forbidden = {
        "induction_selector_payload_native_capability_rung586_results.json",
        "induction_selector_payload_native_capability_rung586_receipt.json",
        "induction_selector_payload_native_capability_audit_rung587.json",
    }
    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text

    def checked_bytes(path):
        assert path.name not in forbidden
        return original_read_bytes(path)

    def checked_text(path, *args, **kwargs):
        assert path.name not in forbidden
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", checked_bytes)
    monkeypatch.setattr(Path, "read_text", checked_text)
    assert producer.build_dryrun()["model_forwards"] == 0


def test_v6_and_complete_local_executable_closure_are_bound():
    assert producer.SOURCE_HASHES[producer.METHOD_HANDOFF_V6] == (
        "d1fdedd90ffff29e6790042b9c9a6ad84278849c3f66707cb586317832fdad1c"
    )
    assert set(producer.SOURCE_HASHES) <= set(adapter.FROZEN_HASHES)
    executable = {
        producer.R585, producer.MANIFEST, producer.FACADE,
        producer.INDUCTION, producer.TT_MODEL,
    }
    producer.verify_sources()
    assert executable <= set(producer._VERIFIED_SOURCE_BYTES)


def test_immutable_launcher_uses_captured_producer_after_path_swap(tmp_path):
    planted = tmp_path / "producer.py"
    planted.write_text("print('verified bytes')\n")
    expected = adapter.sha256(planted)
    _, argv = adapter.diagnostic_command(planted, expected)
    planted.write_text("print('unverified swap')\n")
    completed = subprocess.run(argv, check=True, capture_output=True, text=True)
    assert completed.stdout.strip() == "verified bytes"


def test_exact_234_forward_fit_only_census(execution):
    manifest = producer.build_call_manifest(execution["endpoints"])
    census = producer.operation_census(manifest)
    assert len(manifest) == census["model_forwards"] == 234
    assert census["dispatcher_forward_counts"] == {"N": 132, "F": 24, "R": 78}
    assert census["endpoint_forwards"] == 7_488
    assert census["factor_endpoint_site_role_operations"] == 26_112
    assert census["evaluated_splits"] == ["FIT"]
    assert census["forbidden_splits_opened"] == []
    assert census["model_backwards"] == 0
    assert census["model_weights_updated"] is False


def test_stdout_only_non_scientific_boundary_and_namespaces_absent():
    source = _blob(PRODUCER).decode()
    tree = ast.parse(source)
    called = {
        node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    } | {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert called.isdisjoint({
        "score_split", "select_candidate", "decide_terminal", "publish_result",
        "write_text", "write_bytes", "save", "dump",
    })
    assert producer.STATUS == "diagnostic_only_no_scientific_terminal"
    assert all(not path.exists() for path in adapter.SCIENTIFIC_NAMESPACES)

