import hashlib
import json
from pathlib import Path


OPS = Path(__file__).resolve().parent
CONTRACT = OPS / "circuit_causal_validity_next_wave_handoff_rung585_v6_addendum.json"
PREVIOUS = OPS / "circuit_causal_validity_next_wave_handoff_rung585_v5_addendum.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_v6_extends_exact_v5_contract():
    contract = load_contract()
    assert contract["schema"] == "circuit_causal_validity_next_wave_handoff_v6_addendum"
    assert contract["v5_contract_path"].endswith(
        "circuit_causal_validity_next_wave_handoff_rung585_v5_addendum.json"
    )
    assert sha256(PREVIOUS) == contract["v5_contract_sha256"]


def test_v6_lesson_and_fixture_census_is_exact():
    contract = load_contract()
    lessons = contract["accepted_lessons"]
    assert [row["lesson"] for row in lessons] == [24, 25, 26]
    assert [row["name"] for row in lessons] == [
        "registered_comparison_drives_classification",
        "outcome_blind_dryrun_transitive_closure",
        "verify_before_import_and_execute_immutable_bytes",
    ]
    assert len(contract["forbidden_fallbacks"]) == 4
    assert len(contract["required_test_ids"]) == 5


def test_v6_prompts_name_all_four_planted_attacks():
    contract = load_contract()
    builder = contract["builder_prompt_addendum"].lower()
    critic = contract["critic_prompt_addendum"].lower()
    assert "different-dispatcher" in builder
    assert "transitive dry-run call graph" in builder
    assert "transitive executable import graph" in builder
    assert "immutable verified source snapshot" in builder
    assert "auxiliary-only deviation" in critic
    assert "import-time side effects" in critic
    assert "changed after preflight" in critic


def test_v6_is_strict_finite_standard_json():
    raw = CONTRACT.read_text(encoding="utf-8")
    parsed = json.loads(
        raw,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant: {value}")
        ),
    )
    assert json.dumps(parsed, allow_nan=False)
