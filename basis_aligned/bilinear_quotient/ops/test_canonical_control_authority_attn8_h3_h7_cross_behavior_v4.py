#!/usr/bin/env python3

from collections import Counter

import canonical_control_authority_attn8_h3_h7_cross_behavior_v4 as authority


def test_full_family_digests_and_pairing_are_frozen():
    pairs = authority.build_pairs()
    assert len(pairs) == 384
    assert len({row["pair_id"] for row in pairs}) == 384
    assert authority.canonical(pairs) == authority.EXPECTED_PAIRING_SHA256
    counts = Counter((row["family_id"], row["split"]) for row in pairs)
    assert all(counts[(family, "FIT")] == 64 for family in authority.FAMILIES)
    assert all(counts[(family, "SELECT")] == 32 for family in authority.FAMILIES)


def test_every_pair_is_adjacent_and_has_frozen_three_source_maps():
    for row in authority.build_pairs():
        assert abs(row["recipient"]["final_value"] - row["donor"]["final_value"]) == 1
        assert len(row["recipient"]["source_positions"]) == 3
        assert len(row["donor"]["source_positions"]) == 3
        assert row["recipient"]["query_position"] == len(row["recipient"]["ids"])-1
        assert row["donor"]["query_position"] == len(row["donor"]["ids"])-1
