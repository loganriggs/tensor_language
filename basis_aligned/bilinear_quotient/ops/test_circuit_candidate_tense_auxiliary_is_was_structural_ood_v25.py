from collections import Counter

import circuit_candidate_tense_auxiliary_is_was_structural_ood_v25 as candidate


def test_v25_has_eight_target_structures_eight_matched_controls_and_length_range():
    rows = candidate.build_rows()
    constructions = Counter(row["construction_id"] for row in rows)
    assert len(rows) == len({row["row_id"] for row in rows}) == 64
    assert len(constructions) == 16 and set(constructions.values()) == {4}
    assert len({row["construction_id"] for row in rows if row["family"] in {"A1", "A2"}}) == 8
    assert len({row["construction_id"] for row in rows if row["family"] in {"P", "C"}}) == 8
    lengths = [len(row[key]) for row in rows for key in ("base_ids", "donor_ids")]
    assert min(lengths) == 8 and max(lengths) == 18


def test_every_structure_has_both_directions_and_pair_alignment():
    rows = candidate.build_rows()
    for construction_id in {row["construction_id"] for row in rows}:
        selected = [row for row in rows if row["construction_id"] == construction_id]
        assert set(Counter(row["direction_id"] for row in selected).values()) == {2}
    assert all(row["base_semantic_position"] == row["donor_semantic_position"] for row in rows)
    assert all(all(row["construction_checks"].values()) for row in rows)


def test_v25_reporters_are_history_disjoint_compound_roles():
    rows = candidate.build_rows()
    old = {str(row.get("reporter")) for module in candidate._history_modules()
           for row in (module._build() if hasattr(module, "_build") else module.build_rows())}
    reporters = {row["reporter"] for row in rows}
    assert len(reporters) == 16 and not reporters & old
    assert all(" " in reporter for reporter in reporters)
