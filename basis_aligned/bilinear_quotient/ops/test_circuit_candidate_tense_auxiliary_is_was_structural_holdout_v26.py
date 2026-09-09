from collections import Counter

import circuit_candidate_tense_auxiliary_is_was_structural_holdout_v26 as candidate


def test_v26_is_fresh_balanced_and_structurally_diverse():
    rows = candidate.build_rows()
    counts = Counter(row["construction_id"] for row in rows)
    assert len(rows) == len({row["row_id"] for row in rows}) == 64
    assert len(counts) == 8 and set(counts.values()) == {8}
    assert len({row["construction_id"] for row in rows if row["family"] in {"A1", "A2"}}) == 4
    assert len({row["construction_id"] for row in rows if row["family"] in {"P", "C"}}) == 4
    assert all("holdout_v26" in row["construction_id"] for row in rows)


def test_v26_has_four_rows_per_direction_and_pair_alignment():
    rows = candidate.build_rows()
    for construction_id in {row["construction_id"] for row in rows}:
        selected = [row for row in rows if row["construction_id"] == construction_id]
        assert set(Counter(row["direction_id"] for row in selected).values()) == {4}
    assert all(row["base_semantic_position"] == row["donor_semantic_position"] for row in rows)
    assert all(all(row["construction_checks"].values()) for row in rows)


def test_v26_reporters_rows_and_text_are_history_disjoint():
    rows = candidate.build_rows()
    old_modules = candidate.v25._history_modules() + (candidate.v25,)
    old_rows = [row for module in old_modules
                for row in (module._build() if hasattr(module, "_build") else module.build_rows())]
    assert not {row["reporter"] for row in rows} & {str(row.get("reporter")) for row in old_rows}
    assert not {row["row_id"] for row in rows} & {row["row_id"] for row in old_rows}
    assert not {row[key] for row in rows for key in ("base_text", "donor_text")} & {
        row[key] for row in old_rows for key in ("base_text", "donor_text")}

