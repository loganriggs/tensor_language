import collections

import final_ood_authority_attn8_h3_h7_cached_successor_v1 as authority


def test_full_family_pairing_is_frozen_and_complete():
    pairs = authority.build_pairs()
    assert len(pairs) == 256
    assert authority.canonical(pairs) == authority.EXPECTED_PAIRING_SHA256
    counts = collections.Counter((x["family_id"], x["split"]) for x in pairs)
    assert set(counts.values()) == {32}
    assert set(counts) == {(family, split) for family in authority.FAMILIES
                           for split in authority.SPLITS}


def test_pairing_is_adjacent_and_semantically_exact():
    for pair in authority.build_pairs():
        recipient, donor = pair["recipient"], pair["donor"]
        assert len(recipient["source_positions"]) == 3
        assert len(donor["source_positions"]) == 3
        assert recipient["query_position"] == len(recipient["ids"]) - 1
        assert donor["query_position"] == len(donor["ids"]) - 1
        assert abs(recipient["final_value"] - donor["final_value"]) == 1
        assert recipient["answer_id"] != donor["answer_id"]


def test_compile_plan_never_loads_model():
    plan = authority.compile_plan()
    assert plan["model_loaded"] is False
    assert plan["outcomes_opened"] == []
    assert plan["positive_family"] == "list_step_two_conflict"
