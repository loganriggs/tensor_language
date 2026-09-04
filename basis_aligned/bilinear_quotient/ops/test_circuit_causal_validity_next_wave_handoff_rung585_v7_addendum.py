import hashlib
import json
from pathlib import Path


OPS = Path(__file__).resolve().parent
CONTRACT = OPS / "circuit_causal_validity_next_wave_handoff_rung585_v7_addendum.json"
PREVIOUS = OPS / "circuit_causal_validity_next_wave_handoff_rung585_v6_addendum.json"
ROOT = OPS.parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_v7_extends_exact_v6_contract_and_references_exist():
    contract = load_contract()
    assert contract["schema"] == "circuit_causal_validity_next_wave_handoff_v7_addendum"
    assert sha256(PREVIOUS) == contract["v6_contract_sha256"]
    assert (ROOT / contract["reference_derivation_path"]).is_file()
    assert (ROOT / contract["reference_review_path"]).is_file()


def test_v7_has_one_exact_new_semantic_lesson():
    contract = load_contract()
    assert contract["accepted_lessons"] == [
        {
            "lesson": 27,
            "name": "name_factor_intervention_by_operational_level",
            "requirement": (
                "Keep a centered partial output-factor swap, a mass-compensated partial "
                "coefficient swap, a complete attention-pattern swap, a realizable "
                "query/key-state swap, and literal native remove-and-insert as distinct "
                "interventions. A result may claim only the operational level actually executed."
            ),
        }
    ]


def test_v7_requires_all_distinctions_and_negative_fixtures():
    contract = load_contract()
    builder = contract["builder_prompt_addendum"].lower()
    critic = contract["critic_prompt_addendum"].lower()
    for phrase in (
        "projected output factor",
        "mass compensation",
        "full attention pattern",
        "realizable query/key state",
        "literal-removal",
    ):
        assert phrase in builder
    assert "three-source attention pattern" in critic
    assert "contraction discrepancy" in critic
    assert len(contract["forbidden_claim_aliases"]) == 4
    assert len(contract["required_test_ids"]) == 5


def test_v7_is_strict_finite_standard_json():
    raw = CONTRACT.read_text(encoding="utf-8")
    parsed = json.loads(
        raw,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant: {value}")
        ),
    )
    assert json.dumps(parsed, allow_nan=False)
