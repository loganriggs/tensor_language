from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CONTRACT = ROOT / "ops/task14_mlp15_17_product_subspace_contract_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load():
    return json.loads(CONTRACT.read_text())


def test_all_frozen_sources_match_the_contract():
    contract = _load()
    sources = [
        contract["data"]["task_record"],
        contract["data"]["partition"],
        contract["data"]["donors"],
        contract["data"]["discovery_endpoints"],
        contract["weight_translation"]["compiler"],
    ]
    for source in sources:
        assert _sha256(ROOT / source["path"]) == source["sha256"]


def test_inner_fit_and_select_are_endpoint_disjoint_discovery_groups():
    contract = _load()
    fit = set(contract["data"]["inner_fit_group_numbers"])
    select = set(contract["data"]["inner_select_group_numbers"])
    assert len(fit) == len(select) == 8
    assert fit.isdisjoint(select)
    shard = json.loads((ROOT / contract["data"]["discovery_endpoints"]["path"]).read_text())
    endpoint_groups = {int(item["group_number"]) for item in shard["endpoints"]}
    assert fit | select == endpoint_groups
    fit_endpoints = {item["endpoint_id"] for item in shard["endpoints"] if item["group_number"] in fit}
    select_endpoints = {item["endpoint_id"] for item in shard["endpoints"] if item["group_number"] in select}
    assert fit_endpoints.isdisjoint(select_endpoints)


def test_contract_is_task_conditioned_not_reconstruction_or_rank_sweep():
    contract = _load()
    assert contract["execution_authorized"] is False
    assert contract["status"] == "blocked_and_superseded_by_phase0_full_rank_panel"
    assert "No projector fit" in contract["blocking_phase0"]["rule"]
    assert contract["modules"] == [15, 17]
    assert contract["model_dimensions"] == {
        "normalized_mlp_input": 1152, "bilinear_product": 4608, "mlp_output": 1152,
    }
    assert contract["candidate_ranks"] == [1, 2, 4, 8]
    assert "activation reconstruction" in contract["forbidden_fit_objectives"]
    assert "tensor rank" in contract["forbidden_fit_objectives"]
    assert "rank 8" in contract["registered_outcomes"][3]


def test_contract_requires_two_sided_causality_controls_and_composition():
    contract = _load()
    interventions = contract["interventions"]
    bars = contract["select_bars"]
    assert "f(z_H-P delta_z)-f(z_H)" in interventions["remove_selected"]
    assert "target E_full" in interventions["remove_selected"]
    assert "f(z_B+P delta_z)-f(z_B)" in interventions["keep_only_selected"]
    assert "target -E_full" in interventions["keep_only_selected"]
    assert bars["removal_cell_vector_relative_l2_max"] == 0.2
    assert bars["sufficiency_cell_vector_relative_l2_max"] == 0.2
    assert bars["control_full_vocab_rms_max"] == 0.05
    assert bars["joint_cell_vector_relative_l2_max"] == 0.25
    assert bars["unrelated_interchange_fraction_of_full_module_max"] == 0.1


def test_blocking_corrections_are_machine_readable():
    contract = _load()
    blocking = contract["blocking_phase0"]
    assert _sha256(ROOT / blocking["contract_path"]) == blocking["contract_sha256"]
    assert _sha256(ROOT / blocking["preregistration_path"]) == blocking["preregistration_sha256"]
    assert contract["data"]["role_rule"]["family_letters_never_define_role"] is True
    assert "recompute the live MLP17" in contract["interventions"]["joint_rule"]
    assert contract["unrelated_behavior_control"]["status"] \
        == "deferred_until_module_liveness_is_proven"
    assert "ker(W_D)" in contract["weight_translation"]["identifiability"]
