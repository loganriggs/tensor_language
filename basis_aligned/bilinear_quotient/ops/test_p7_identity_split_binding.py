import hashlib
import json

import pytest

import p7_identity_split_binding as target
import run_temporal_iswas_p7_identity_a11_head_endpoint_atlas_ood_v1 as a11_executor
import run_temporal_iswas_p7_identity_m11_exact_product_factorial_ood_v1 as m11_executor


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def result(stable, *, instrument=True, selective=True):
    stable = list(stable)
    return {
        "schema": "temporal_iswas_p7_identity_background_module_leave_one_out_ood_result_v1",
        "P7": list(target.P7),
        "stable_necessary_modules": stable,
        "predictions": {
            target.PRED_A: instrument,
            target.PRED_B: len(stable) >= 2,
            target.PRED_C: "A11" in stable,
            target.PRED_D: any(name.startswith("M") for name in stable),
            target.PRED_E: selective,
        },
    }


def fixture_paths(tmp_path):
    necessity_runner = tmp_path / "necessity.py"
    necessity_hash = write(necessity_runner, "necessity\n")
    branches = {}
    for name, member, required in (
        ("a11", "A11", (target.PRED_A, target.PRED_C, target.PRED_E)),
        ("m11", "M11", (target.PRED_A, target.PRED_D, target.PRED_E)),
    ):
        runner = tmp_path / f"{name}.py"
        runner_hash = write(runner, f"{name}\n")
        branches[name] = {
            "member": member, "required_predictions": required,
            "runner": runner, "expected_runner_sha256": runner_hash,
            "binding": tmp_path / "bindings" / f"{name}.json",
        }
    return necessity_runner, necessity_hash, branches


def test_missing_result_is_a_model_free_wait(tmp_path):
    necessity_runner, necessity_hash, branches = fixture_paths(tmp_path)
    report = target.create_bindings(
        necessity_result=tmp_path / "missing.json",
        necessity_runner=necessity_runner,
        expected_necessity_runner_sha256=necessity_hash,
        branches=branches,
    )
    assert report == {"status": "awaiting_result", "created": [],
                      "existing": [], "eligible": []}
    assert not (tmp_path / "bindings").exists()


def test_both_eligible_branches_bind_once_and_are_idempotent(tmp_path):
    necessity_runner, necessity_hash, branches = fixture_paths(tmp_path)
    result_path = tmp_path / "result.json"
    write(result_path, json.dumps(result(("A11", "M11"))))
    first = target.create_bindings(
        necessity_result=result_path, necessity_runner=necessity_runner,
        expected_necessity_runner_sha256=necessity_hash, branches=branches,
    )
    second = target.create_bindings(
        necessity_result=result_path, necessity_runner=necessity_runner,
        expected_necessity_runner_sha256=necessity_hash, branches=branches,
    )
    assert first["created"] == ["a11", "m11"] and first["eligible"] == ["a11", "m11"]
    assert second["created"] == [] and second["existing"] == ["a11", "m11"]
    for name in ("a11", "m11"):
        payload = json.loads(branches[name]["binding"].read_text())
        assert payload["necessity_result_sha256"] == first["necessity_result_sha256"]
        assert payload["branch_runner_sha256"] == branches[name]["expected_runner_sha256"]


def test_only_prospectively_eligible_branch_is_emitted(tmp_path):
    necessity_runner, necessity_hash, branches = fixture_paths(tmp_path)
    result_path = tmp_path / "result.json"
    result_hash = write(result_path, json.dumps(result(("A11",))))
    report = target.create_bindings(
        necessity_result=result_path, necessity_runner=necessity_runner,
        expected_necessity_runner_sha256=necessity_hash, branches=branches,
        dry_run=True,
    )
    assert report["eligible"] == ["a11"] and report["created"] == []
    assert report["payloads"]["a11"]["necessity_result_sha256"] == result_hash
    assert not branches["a11"]["binding"].exists()
    assert "m11" not in report["payloads"]


def test_inconsistent_derived_prediction_fails_closed(tmp_path):
    necessity_runner, necessity_hash, branches = fixture_paths(tmp_path)
    bad = result(("A11", "M11"))
    bad["predictions"][target.PRED_C] = False
    result_path = tmp_path / "result.json"
    write(result_path, json.dumps(bad))
    with pytest.raises(target.P7SplitBindingError, match="disagree"):
        target.create_bindings(
            necessity_result=result_path, necessity_runner=necessity_runner,
            expected_necessity_runner_sha256=necessity_hash, branches=branches,
        )


def test_changed_authority_or_branch_runner_creates_nothing(tmp_path):
    necessity_runner, necessity_hash, branches = fixture_paths(tmp_path)
    result_path = tmp_path / "result.json"
    write(result_path, json.dumps(result(("A11", "M11"))))
    write(branches["a11"]["runner"], "changed\n")
    with pytest.raises(target.P7SplitBindingError, match="branch runner hash changed"):
        target.create_bindings(
            necessity_result=result_path, necessity_runner=necessity_runner,
            expected_necessity_runner_sha256=necessity_hash, branches=branches,
        )
    assert not (tmp_path / "bindings").exists()


def test_emitted_bindings_are_accepted_by_frozen_branch_executors(tmp_path, monkeypatch):
    necessity = result(("A11", "M11"))
    result_path = tmp_path / "result.json"
    result_hash = write(result_path, json.dumps(necessity))
    payloads = target.binding_plan(
        necessity,
        result_sha256=result_hash,
        necessity_runner_sha256=target.EXPECTED_NECESSITY_RUNNER_SHA256,
    )

    monkeypatch.setattr(a11_executor, "NECESSITY_RESULT", result_path)
    monkeypatch.setattr(m11_executor, "NECESSITY_RESULT", result_path)
    assert a11_executor.eligibility(payloads["a11"], necessity)
    assert m11_executor.eligibility(payloads["m11"], necessity)
