"""Prospective causal-validity and crash-recovery attacks for frozen R585.

This package pins the blocked a4e7c46c6 producer. Tests marked as strict xfail
are repair contracts: the frozen producer is expected to fail them, while a
future repair must make them pass before the marks are removed. No test opens
an R585 outcome or loads the model.
"""

# BQLANE: cpu

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve()
OPS = SCRIPT.parent
ROOT = OPS.parent
REPO = ROOT.parent.parent
ROWS = ROOT / "induction_selector_payload_three_source_rows_rung578.json"
PRODUCER = OPS / "induction_selector_payload_frozen_factor_rung585.py"
HANDOFF = OPS / "circuit_causal_validity_next_wave_handoff_rung585.json"
CHECKLIST = ROOT.parent / "polynomial_causal" / "CIRCUIT_CAUSAL_VALIDITY_NEXT_WAVE_CRITIC_CHECKLIST.md"

PRODUCER_SHA256 = "4911200ae12dd9c27a609879fded8aab1b5704ef1116f25079b5df7a40162ff3"
ROWS_SHA256 = "8893ff83ea6080ad704f38376715d19be8971867178a4edc3bfd61fe025b39b6"
FROZEN_COMMIT = "a4e7c46c6339c75fc7f89c1e35339e15e3b74fd9"
PRODUCER_RELATIVE = str(PRODUCER.relative_to(REPO))
TARGET_FAMILIES = (
    "two_valid_sources_selector_swap",
    "payload_swap_match_preserved",
    "selector_payload_joint_answer_preserved",
    "match_break_payload_preserved",
)
REQUIRED_CLAIMS = (
    "counterfactual_validity",
    "interaction_isolation",
    "held_out_prediction",
    "ood_prediction",
    "sufficiency",
    "selective_removal",
    "composition_and_reuse",
    "stable_identification",
)
BLOCKED_REPAIR = pytest.mark.xfail(
    strict=True,
    reason="atomic publication contract is unmet by frozen producer a4e7c46c6",
)
REGISTERED_PREDICATES = {
    "pred_a_counterfactuals_are_semantically_valid":
        "recipient and donor rows change only their registered causal variables",
    "pred_b_counterfactuals_have_multiple_realizations":
        "every target cell has lexical, layout, length, and direction diversity",
    "pred_c_publication_is_atomic_and_recoverable":
        "crashes cannot leave partial final evidence result or receipt namespaces",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_json(path: Path):
    def reject(value: str):
        raise ValueError(f"nonfinite JSON constant {value}")

    return json.loads(path.read_text(), parse_constant=reject)


def _git_blob(relative: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{FROZEN_COMMIT}:{relative}"], cwd=REPO
    )


def _load_producer(tmp_path_factory):
    frozen_root = tmp_path_factory.mktemp("r585-frozen-a4e7c46c6")
    target = frozen_root / PRODUCER_RELATIVE
    target.parent.mkdir(parents=True)
    target.write_bytes(_git_blob(PRODUCER_RELATIVE))
    owner_relative = (
        "basis_aligned/bilinear_quotient/ops/"
        "test_induction_selector_payload_frozen_factor_rung585.py"
    )
    owner = frozen_root / owner_relative
    owner.write_bytes(_git_blob(owner_relative))
    assert _sha256(target) == PRODUCER_SHA256
    name = "r585_next_wave_frozen_producer"
    spec = importlib.util.spec_from_file_location(name, target)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def producer(tmp_path_factory):
    return _load_producer(tmp_path_factory)


@pytest.fixture(scope="module")
def rows():
    document = _strict_json(ROWS)
    return [
        row for row in document["rows"]
        if row["split"] in ("FIT", "SELECT")
        and row["family_id"] in TARGET_FAMILIES
    ]


@pytest.fixture(scope="module")
def handoff():
    return _strict_json(HANDOFF)


def _validate_endpoint(row, prefix: str) -> None:
    ids = row[prefix + "_ids"]
    structure = row[prefix + "_structure"]
    sources = structure["source_positions"]
    payloads = structure["payload_positions"]
    assert len(sources) == len(payloads) == 2
    assert all(payload == source + 1 for source, payload in zip(sources, payloads))
    assert [ids[index] for index in sources] == structure["source_ids"]
    assert [ids[index] for index in payloads] == structure["payload_ids"]
    assert ids[structure["query_position"]] == structure["query_id"]
    assert structure["query_position"] == len(ids) - 1
    assert len(set(structure["source_ids"])) == 2


def _common_structure(base, donor, keys) -> None:
    for key in keys:
        assert base[key] == donor[key], f"held-fixed semantic field changed: {key}"


def _validate_target_counterfactual(row) -> None:
    assert row["evaluation_directions"] == ["base_to_donor", "donor_to_base"]
    assert row["base_ids"] != row["donor_ids"], "recipient and donor endpoints coincide"
    _validate_endpoint(row, "base")
    _validate_endpoint(row, "donor")
    base, donor = row["base_structure"], row["donor_structure"]
    family = row["family_id"]
    fixed_layout = ("source_positions", "payload_positions", "pair_order")

    if family == "two_valid_sources_selector_swap":
        assert row["base_selector"] != row["donor_selector"]
        assert row["base_payload_assignment"] == row["donor_payload_assignment"]
        assert row["answer_changes"] is True
        _common_structure(
            base, donor,
            (*fixed_layout, "source_ids", "payload_ids", "neutral_source_id",
             "neutral_payload_id"),
        )
        assert base["query_id"] != donor["query_id"]
    elif family == "payload_swap_match_preserved":
        assert row["base_selector"] == row["donor_selector"]
        assert row["base_payload_assignment"] != row["donor_payload_assignment"]
        assert row["answer_changes"] is True
        _common_structure(
            base, donor,
            (*fixed_layout, "source_ids", "query_id", "neutral_source_id",
             "neutral_payload_id"),
        )
        assert base["payload_ids"] != donor["payload_ids"]
        assert sorted(base["payload_ids"]) == sorted(donor["payload_ids"])
    elif family == "selector_payload_joint_answer_preserved":
        assert row["base_selector"] != row["donor_selector"]
        assert row["base_payload_assignment"] != row["donor_payload_assignment"]
        assert row["answer_changes"] is False
        assert row["base_answer_id"] == row["donor_answer_id"]
        _common_structure(base, donor, (*fixed_layout, "neutral_source_id", "neutral_payload_id"))
        assert sorted(base["source_ids"]) == sorted(donor["source_ids"])
        assert sorted(base["payload_ids"]) == sorted(donor["payload_ids"])
    elif family == "match_break_payload_preserved":
        assert row["base_selector"] == row["donor_selector"]
        assert row["base_payload_assignment"] == row["donor_payload_assignment"]
        assert row["answer_changes"] is False
        assert row["base_answer_id"] == row["donor_answer_id"]
        _common_structure(
            base, donor,
            (*fixed_layout, "payload_ids", "query_id", "neutral_source_id",
             "neutral_payload_id"),
        )
        support = {
            sum(source == endpoint["query_id"] for source in endpoint["source_ids"])
            for endpoint in (base, donor)
        }
        assert support == {0, 1}
    else:
        raise AssertionError(f"unexpected target family: {family}")


def test_exact_frozen_authority_and_outcome_boundary(handoff):
    assert hashlib.sha256(_git_blob(PRODUCER_RELATIVE)).hexdigest() == PRODUCER_SHA256
    assert hashlib.sha256(_git_blob(str(ROWS.relative_to(REPO)))).hexdigest() == ROWS_SHA256
    assert handoff["frozen_target"] == {
        "commit": "a4e7c46c6339c75fc7f89c1e35339e15e3b74fd9",
        "producer_sha256": PRODUCER_SHA256,
        "review_verdict": "blocked_pre_execution",
    }
    for relative, digest in handoff["authority_sha256"].items():
        if relative.endswith("IMPLEMENTATION_PREEXECUTION_REVIEW.md"):
            observed = _sha256(REPO / relative)
        else:
            observed = hashlib.sha256(_git_blob(relative)).hexdigest()
        assert observed == digest
    assert all(not (REPO / relative).exists()
               for relative in handoff["outcome_boundary"]["must_be_absent_before_work"])


def test_semantic_counterfactual_family_contract(rows):
    assert len(rows) == 1_080
    for row in rows:
        _validate_target_counterfactual(row)


def test_multiple_valid_counterfactual_realizations(rows):
    fit_groups = {row["group_id"] for row in rows if row["split"] == "FIT"}
    select_groups = {row["group_id"] for row in rows if row["split"] == "SELECT"}
    assert len(fit_groups) == 72 and len(select_groups) == 36
    assert fit_groups.isdisjoint(select_groups)
    for split, expected in (("FIT", 72), ("SELECT", 36)):
        variants = sorted({
            (row["family_id"], row["family_variant"])
            for row in rows if row["split"] == split
        })
        assert len(variants) == 10
        for family, variant in variants:
            cell = [row for row in rows if row["split"] == split
                    and row["family_id"] == family
                    and row["family_variant"] == variant]
            assert len(cell) == len({row["group_id"] for row in cell}) == expected
            assert len({tuple(row["base_structure"]["pair_order"]) for row in cell}) == 3
            assert len({len(row["base_ids"]) for row in cell}) == 2
            assert len({(tuple(row["base_ids"]), tuple(row["donor_ids"]))
                        for row in cell}) == expected


@pytest.mark.parametrize(
    "family,mutation",
    [
        ("two_valid_sources_selector_swap", "change_held_payload_assignment"),
        ("payload_swap_match_preserved", "change_held_selector"),
        ("match_break_payload_preserved", "change_held_payload_token"),
        ("selector_payload_joint_answer_preserved", "identical_endpoints"),
    ],
)
def test_mutated_counterfactual_is_rejected(rows, family, mutation):
    row = copy.deepcopy(next(item for item in rows if item["family_id"] == family))
    if mutation == "change_held_payload_assignment":
        row["donor_payload_assignment"] = 1 - row["base_payload_assignment"]
    elif mutation == "change_held_selector":
        row["donor_selector"] = 1 - row["base_selector"]
    elif mutation == "change_held_payload_token":
        position = row["donor_structure"]["payload_positions"][0]
        row["donor_ids"][position] += 1
        row["donor_structure"]["payload_ids"][0] += 1
    elif mutation == "identical_endpoints":
        row["donor_ids"] = copy.deepcopy(row["base_ids"])
        row["donor_structure"] = copy.deepcopy(row["base_structure"])
    with pytest.raises(AssertionError):
        _validate_target_counterfactual(row)


def test_recipient_donor_factor_source_mapping_is_exact(handoff):
    assert handoff["counterfactual_contract"]["arms"] == {
        "replay": ["recipient_score", "recipient_payload"],
        "score": ["donor_score", "recipient_payload"],
        "payload": ["recipient_score", "donor_payload"],
        "joint": ["donor_score", "donor_payload"],
    }
    recipient = {"score": (2, 3), "payload": (5, 7)}
    donor = {"score": (11, 13), "payload": (17, 19)}
    observed = {
        "replay": sum(e * u for e, u in zip(recipient["score"], recipient["payload"])),
        "score": sum(e * u for e, u in zip(donor["score"], recipient["payload"])),
        "payload": sum(e * u for e, u in zip(recipient["score"], donor["payload"])),
        "joint": sum(e * u for e, u in zip(donor["score"], donor["payload"])),
    }
    assert observed == {"replay": 31, "score": 146, "payload": 91, "joint": 434}


def test_same_group_factorial_interaction_requires_all_four_states(handoff, rows):
    contract = handoff["interaction_contract"]
    assert contract == {
        "formula": (
            "(joint_correct_margin-score_correct_margin-"
            "payload_correct_margin+replay_correct_margin)/4"
        ),
        "join_keys": ["split", "directed_id", "group_id"],
        "primitive_outcome": "correct_margin",
        "required_states": ["replay", "score", "payload", "joint"],
    }
    joint_rows = [
        row for row in rows
        if row["family_id"] == "selector_payload_joint_answer_preserved"
    ]
    assert len(joint_rows) == 216
    assert all(row["answer_changes"] is False for row in joint_rows)
    assert all(row["base_answer_id"] == row["donor_answer_id"] for row in joint_rows)
    replay, score, payload, joint = 2.0, 1.0, 1.0, 1.0
    interaction = (joint - score - payload + replay) / 4.0
    assert math.isfinite(interaction) and interaction == .25


def test_claim_ladder_keeps_all_eight_claims_distinct(handoff):
    claims = handoff["claim_ladder"]
    assert [row["claim"] for row in claims] == list(REQUIRED_CLAIMS)
    assert all(row["required_evidence"] for row in claims)
    evidence = {row["claim"]: set(row["required_evidence"]) for row in claims}
    assert "separately_preregistered_distribution_shift" in evidence["ood_prediction"]
    assert "separately_preregistered_live_term_removal" in evidence["selective_removal"]
    assert "component_reuse_outperforms_independent_memorization_baseline" \
        in evidence["composition_and_reuse"]
    assert "equivalent_gauges_are_quotiented" in evidence["stable_identification"]


def test_machine_checkable_two_agent_handoff_is_complete(handoff):
    assert handoff["schema"] == "circuit_causal_validity_next_wave_handoff_v1"
    assert set(handoff["agent_prompts"]) == {"builder", "critic"}
    assert handoff["agent_prompts"]["builder"] != handoff["agent_prompts"]["critic"]
    assert len(handoff["required_test_ids"]) == len(set(handoff["required_test_ids"])) == 12
    required = handoff["next_agent_handoff_required_fields"]
    assert len(required) == 19
    assert required["verdict"] == "approve_exact_bytes_or_block"
    assert required["claim_status"] == "object_claim_to_held_failed_or_not_tested"
    assert set(handoff["publication_contract"]["crash_points"]) == {
        "during_evidence_array_write",
        "after_evidence_before_result",
        "after_result_before_receipt",
        "during_final_publish",
    }
    package = handoff["package_artifacts"]
    assert _sha256(CHECKLIST) == package["critic_checklist_sha256"]
    assert _sha256(SCRIPT) == package["prospective_test_sha256"]


@BLOCKED_REPAIR
def test_crash_during_evidence_write_leaves_no_final_namespace(
    producer, tmp_path, monkeypatch
):
    final_evidence = tmp_path / "r585_final_evidence"
    monkeypatch.setattr(producer, "EVIDENCE_DIR", final_evidence)

    def planted_crash(*args, **kwargs):
        raise OSError("planted crash during evidence array write")

    monkeypatch.setattr(producer.np, "save", planted_crash)
    with pytest.raises(OSError, match="planted crash"):
        producer.write_evidence(
            {"endpoints": []}, {}, [], [], {}, {}
        )
    assert not final_evidence.exists()


@BLOCKED_REPAIR
def test_crash_after_evidence_before_result_leaves_no_final_namespace(
    producer, monkeypatch
):
    final_evidence = producer.ROOT / "r585_final_evidence_after_capture"
    monkeypatch.setattr(producer, "EVIDENCE_DIR", final_evidence)
    descriptors = producer.write_evidence(
        {"endpoints": []}, {}, [], [], {}, {}
    )
    assert descriptors
    try:
        raise OSError("planted crash after evidence before result")
    except OSError as crash_exception:
        assert "planted crash" in str(crash_exception)
    assert not final_evidence.exists()


@BLOCKED_REPAIR
def test_crash_between_result_and_receipt_is_atomic_and_retryable(
    producer, tmp_path, monkeypatch
):
    final_result = producer.ROOT / "r585_results.json"
    final_receipt = producer.ROOT / "r585_receipt.json"
    monkeypatch.setattr(producer, "OUT", final_result)
    monkeypatch.setattr(producer, "RECEIPT", final_receipt)
    result = producer.make_result_fixture("factor_capacity_null")
    original_make_receipt = producer.make_receipt_fixture

    def planted_crash(document):
        raise OSError("planted crash after result before receipt")

    monkeypatch.setattr(producer, "make_receipt_fixture", planted_crash)
    with pytest.raises(OSError, match="planted crash"):
        producer._finish_result(result)
    assert not final_result.exists() and not final_receipt.exists()

    monkeypatch.setattr(producer, "make_receipt_fixture", original_make_receipt)
    producer._finish_result(result)
    assert final_result.is_file() and final_receipt.is_file()
    receipt = _strict_json(final_receipt)
    assert receipt["result_sha256"] == hashlib.sha256(final_result.read_bytes()).hexdigest()
