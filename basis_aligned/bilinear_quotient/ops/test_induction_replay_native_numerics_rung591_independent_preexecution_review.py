"""Model/outcome-free planted attacks on exact R591 commit 1396747c0."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


OPS = Path(__file__).resolve().parent
ROOT = OPS.parent
POLY = ROOT.parent / "polynomial_causal"
PRODUCER = OPS / "induction_replay_native_numerics_rung591.py"
OWNER = OPS / "test_induction_replay_native_numerics_rung591.py"
DRYRUN = ROOT / "induction_replay_native_numerics_rung591_dryrun.json"
ADAPTER = OPS / "execute_induction_replay_native_numerics_rung591.py"
ADAPTER_TEST = OPS / "test_execute_induction_replay_native_numerics_rung591.py"
PREREG = POLY / "INDUCTION_REPLAY_NATIVE_NUMERICS_RUNG591_PREREGISTRATION.md"

EXACT_HASHES = {
    PRODUCER: "b2b266529f0f842211fea46856064133df5e3f4a8a7758c9095e7d29a94b6c49",
    OWNER: "e756ba3d17d3ebee2f81e97e573dd216090555de1fd3f1cfc926268f902d9ce7",
    DRYRUN: "161193de5d90da69aafcd681e375993fa91d32e99100f0ed02fb586d5a629d8b",
    ADAPTER: "5fe0a0d3bb4c149881a1d6d76f5adf7e661df35af39cc37e1cd9893b93cc33cd",
    ADAPTER_TEST: "b20ea468089c90629191f71c6e5f97d4caec180fce64bf0d1ce17f3f9565d7b6",
    PREREG: "e72cb386d65c68f55b767c8141c3c4d774b3c8ad9387ac7f8ad43bebef118593",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def producer():
    assert all(_sha256(path) == digest for path, digest in EXACT_HASHES.items())
    # Import defines functions/constants only.  It does not call load_authority,
    # the transitive function which would parse upstream outcome artifacts.
    return _load(PRODUCER, "r591_independent_review_producer")


def _cell(value: float) -> dict[str, float]:
    return {"max_abs": value}


def _comparisons(producer, *, padding_dispatcher=None, membership_dispatcher=None):
    zero = _cell(0.0)
    padding = {name: _cell(0.0) for name in producer.DISPATCHERS}
    membership = {name: _cell(0.0) for name in producer.DISPATCHERS}
    if padding_dispatcher is not None:
        padding[padding_dispatcher] = _cell(2e-5)
    if membership_dispatcher is not None:
        membership[membership_dispatcher] = _cell(2e-5)
    return {
        "full_fit": {"total": zero, "hook": zero},
        "panel": {
            "observer": {name: zero for name in producer.PANEL_SCHEDULES},
            "hook": {name: zero for name in producer.PANEL_SCHEDULES},
            "padding": padding,
            "membership": membership,
        },
    }


@pytest.mark.xfail(
    strict=True,
    reason="R591 classifies non-native F/R padding and membership as native numerical causes",
)
@pytest.mark.parametrize(
    "padding_dispatcher,membership_dispatcher",
    [("R", None), (None, "F")],
)
def test_padding_and_membership_classes_use_only_frozen_native_comparisons(
    producer, padding_dispatcher, membership_dispatcher,
):
    observed = producer.interpret(_comparisons(
        producer,
        padding_dispatcher=padding_dispatcher,
        membership_dispatcher=membership_dispatcher,
    ))
    assert observed["classification"] == "all_registered_components_within_threshold"
    assert observed["active_components"] == []


@pytest.mark.xfail(
    strict=True,
    reason="v5 requires emitted selected IDs in addition to their ordered hash",
)
def test_v5_panel_receipt_emits_exact_ordered_selected_ids():
    payload = json.loads(DRYRUN.read_text(encoding="utf-8"))
    panel = payload["panel"]
    identifiers = panel["ordered_endpoint_ids"]
    assert len(identifiers) == len(set(identifiers)) == 256
    assert hashlib.sha256(
        json.dumps(
            identifiers, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest() == panel["ordered_endpoint_ids_sha256"]


@pytest.mark.xfail(
    strict=True,
    reason="adapter hashes a mutable path and later execs that path without immutable handoff",
)
def test_managed_adapter_cannot_execute_bytes_swapped_after_preflight(tmp_path, monkeypatch):
    adapter = _load(ADAPTER, "r591_independent_review_adapter")
    planted = tmp_path / "producer.py"
    planted.write_bytes(b"original\n")
    expected = _sha256(planted)
    monkeypatch.setattr(adapter, "PRODUCER", planted)

    def preflight(*, namespace_paths):
        adapter.verify_frozen_bytes({planted: expected})
        return {"checked": True}

    observed = []

    class ExecObserved(Exception):
        pass

    def swap_then_exec(executable, argv):
        planted.write_bytes(b"swapped after hash check\n")
        observed.append(_sha256(Path(argv[1])))
        raise ExecObserved

    monkeypatch.setattr(adapter, "preflight", preflight)
    with pytest.raises(ExecObserved):
        adapter.dispatch({}, exec_function=swap_then_exec, namespace_paths=())
    assert observed == [expected]

