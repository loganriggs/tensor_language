from pathlib import Path

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v23 as candidate


def test_v23_exact_inventory_novelty_and_digest():
    rows = candidate.build_rows()
    assert len(rows) == len({row["row_id"] for row in rows}) == 64
    assert {row["family"] for row in rows} == {"A1", "A2", "P", "C"}
    assert all(sum(row["family"] == family for row in rows) == 16
               for family in ("A1", "A2", "P", "C"))
    assert candidate.validate_rows(rows) == "46e9e493a4b2b42ce0c121b30978f08208537300363fa91902184c8f8d6565c7"


def test_v23_every_patch_pair_is_position_compatible():
    rows = candidate.build_rows()
    assert all(row["base_semantic_position"] == row["donor_semantic_position"]
               for row in rows)
    assert all(len(row["base_ids"]) == len(row["donor_ids"]) for row in rows)


def test_v23_c_is_equal_length_answer_preserving_and_not_temporal():
    controls = [row for row in candidate.build_rows() if row["family"] == "C"]
    assert all(row["base_answer"] == row["donor_answer"] for row in controls)
    assert all("harbor" in row["base_text"] + row["donor_text"]
               and "canyon" in row["base_text"] + row["donor_text"] for row in controls)
    prior = Path(__file__).resolve().parents[1] / "circuits/prior_art/tense_auxiliary_is_was_fresh_lexicon_v23_capability_v1.json"
    assert prior.exists()
