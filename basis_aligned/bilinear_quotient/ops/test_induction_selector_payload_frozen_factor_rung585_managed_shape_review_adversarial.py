"""Independent CPU-only review of immutable R585 managed-shape repair."""

# BQLANE: cpu

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest


SCRIPT = Path(__file__).resolve()
OPS = SCRIPT.parent
ROOT = OPS.parent
REPO = ROOT.parent.parent
PRODUCER = OPS / "induction_selector_payload_frozen_factor_rung585.py"
OWNER_TEST = OPS / "test_induction_selector_payload_frozen_factor_rung585.py"
DRYRUN = ROOT / "induction_selector_payload_frozen_factor_rung585_dryrun.json"
ADAPTER = OPS / "execute_induction_selector_payload_frozen_factor_rung585.py"
ADAPTER_TEST = OPS / "test_execute_induction_selector_payload_frozen_factor_rung585.py"
FACADE = ROOT.parent / "polynomial_causal" / "bilin18_observed_model_facade.py"

CANDIDATE_COMMIT = "c4288dbe8ee6213dfc4dcb538024dc119fbb642e"
CANDIDATE_HASHES = {
    PRODUCER: "fd772c3b9d6df4271ecbfc90c00c893db5a65ea06601f0c8f6e7a9e34c9a531b",
    OWNER_TEST: "fcaba664269de12a41a5adb8ff089fc9963eeec91577ef94993ff032c02fc885",
    DRYRUN: "580a570426ce48c9e43f5fce82c976dece6c71e8a11c1b057054c17cf958dcf8",
    ADAPTER: "a65b12c2e88ae57c4d563219ed76f14ddb413b77c4cafcb757a0af415278883a",
    ADAPTER_TEST: "725f0af145ae0883449ac93b7bdb7f29b1c2cc313d7ffc9e892e10efc74743aa",
}
REGISTERED_PREDICATES = {
    "pred_a_all_science_forwards_are_enumerated":
        "the whole producer contains exactly the three registered facade forward sites",
    "pred_b_shape_relaxation_preserves_validation":
        "variable shapes retain token, model, checkpoint, logit-shape, and finite checks",
    "pred_c_price_and_science_are_unchanged":
        "only facade shape assertions change; schedules, interventions, and 459/231 price remain",
}


def _git_blob(path: Path) -> bytes:
    relative = str(path.relative_to(REPO))
    return subprocess.check_output(
        ["git", "show", f"{CANDIDATE_COMMIT}:{relative}"], cwd=REPO
    )


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    descriptor, name = tempfile.mkstemp(
        prefix=".r585-shape-review-", suffix=".py", dir=OPS
    )
    os.close(descriptor)
    path = Path(name)
    path.write_bytes(_git_blob(PRODUCER))
    module = _load(path, "r585_managed_shape_review_producer")
    try:
        yield module
    finally:
        path.unlink(missing_ok=True)


def test_exact_candidate_commit_and_five_blobs():
    assert subprocess.check_output(
        ["git", "rev-parse", CANDIDATE_COMMIT], cwd=REPO, text=True
    ).strip() == CANDIDATE_COMMIT
    for path, expected in CANDIDATE_HASHES.items():
        assert hashlib.sha256(_git_blob(path)).hexdigest() == expected


class _ForwardCallCensus(ast.NodeVisitor):
    def __init__(self):
        self.functions = []
        self.calls = []
        self.aliases = []
        self.direct_model_forwards = []

    def visit_FunctionDef(self, node):
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node):
        if isinstance(node.value, ast.Attribute) and (
            node.value.attr == "forward_with_dispatch"
        ):
            self.aliases.append(node)
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and (
            node.func.attr == "forward_with_dispatch"
        ):
            keyword = next(
                (item for item in node.keywords if item.arg == "require_production"), None
            )
            self.calls.append((tuple(self.functions), keyword))
        if isinstance(node.func, ast.Name) and node.func.id == "model":
            self.direct_model_forwards.append(node)
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) \
                and node.func.value.id == "model" and node.func.attr in ("forward", "__call__"):
            self.direct_model_forwards.append(node)
        self.generic_visit(node)


def test_global_ast_census_cannot_hide_a_fourth_science_forward():
    tree = ast.parse(_git_blob(PRODUCER).decode())
    census = _ForwardCallCensus()
    census.visit(tree)
    assert census.aliases == []
    assert census.direct_model_forwards == []
    assert {stack[-1] for stack, _ in census.calls} == {
        "collect_capture_replay", "collect_native_comparator", "collect_intervention_arm"
    }
    assert len(census.calls) == 3
    for _, keyword in census.calls:
        assert keyword is not None
        assert isinstance(keyword.value, ast.Constant)
        assert keyword.value.value is False


def test_run_science_reaches_each_and_only_each_registered_collector(runner):
    tree = ast.parse(inspect.getsource(runner.run_science))
    invoked = [
        call.func.id for call in ast.walk(tree)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        and call.func.id in {
            "collect_capture_replay", "collect_native_comparator", "collect_intervention_arm"
        }
    ]
    assert sorted(invoked) == [
        "collect_capture_replay", "collect_intervention_arm", "collect_native_comparator"
    ]
    source = inspect.getsource(runner.run_science)
    assert "verify_weights_sha256=True" in source
    assert "checkpoint.weights_sha256 != CHECKPOINT_SHA256" in source


def test_false_flag_only_relaxes_fixed_shape_and_reuses_validated_model():
    facade_source = _git_blob(FACADE).decode()
    assert "if require_production:\n        validate_production_model(model)" in facade_source
    assert "validate_tokens(tokens, production_shape=require_production)" in facade_source
    assert "expected_vocab = LOGIT_VOCAB if require_production else model.config.vocab_size" \
        in facade_source
    assert "validate_production_model(model)\n    return model, receipt" in facade_source
    assert "if tuple(logits.shape) != (*tokens.shape, expected_vocab)" in facade_source
    assert 'raise RuntimeError("bilin18 logits are nonfinite")' in facade_source


def test_registered_batch_schedules_and_prices_are_unchanged(runner):
    execution = runner.build_execution_authority()
    expected = {
        "FIT": (54, 117, 459),
        "SELECT": (27, 59, 231),
    }
    for split, (endpoint_calls, direction_calls, price) in expected.items():
        schedules = runner.endpoint_schedules(execution, split)
        observed_endpoints = len(schedules["capture"])
        observed_comparators = len(schedules["comparator"])
        observed_directions = len(runner.direction_batches(execution, split))
        assert (observed_endpoints, observed_comparators, observed_directions) == (
            endpoint_calls, endpoint_calls, direction_calls
        )
        assert observed_endpoints + observed_comparators + 3 * observed_directions == price
    assert sum(value[2] for value in expected.values()) == 690


def test_adapter_pins_the_repaired_producer_owner_and_dryrun():
    adapter_source = _git_blob(ADAPTER).decode()
    for path in (PRODUCER, OWNER_TEST, DRYRUN):
        assert CANDIDATE_HASHES[path] in adapter_source
