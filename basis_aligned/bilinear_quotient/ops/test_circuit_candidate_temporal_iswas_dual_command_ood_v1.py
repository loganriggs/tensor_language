from collections import Counter

import circuit_candidate_temporal_iswas_dual_command_ood_v1 as target


def test_frozen_authority_and_balance():
    rows = target.build_rows()
    assert target.validate_rows(rows) == target.EXPECTED_AUTHORITY_SHA256
    assert len(rows) == 32 and len({row["row_id"] for row in rows}) == 32
    assert Counter((row["phase"], row["template_id"]) for row in rows) == Counter(
        {(phase, template): 8 for phase in ("FIT", "HOLDOUT") for template in target.TEMPLATES})


def test_cells_are_position_and_length_aligned():
    for row in target.build_rows():
        endpoints = list(row["endpoints"].values())
        assert len({len(endpoint["ids"]) for endpoint in endpoints}) == 1
        assert len({endpoint["temporal_position"] for endpoint in endpoints}) == 1
        assert len({endpoint["iswas_position"] for endpoint in endpoints}) == 1


def test_temporal_prefixes_are_exact_source_prefixes():
    for row in target.build_rows():
        for endpoint in row["endpoints"].values():
            position = endpoint["temporal_position"]
            assert endpoint["ids"][:position] == endpoint["temporal_prefix_ids"]
            assert endpoint["ids"][position] == endpoint["temporal_answer_id"]
            assert endpoint["ids"][endpoint["iswas_position"]] == endpoint["iswas_answer_id"]


def test_new_templates_and_source_authorities_are_used():
    rows = target.build_rows()
    assert set(row["template_id"] for row in rows) == set(target.TEMPLATES)
    assert all("registry" in row["endpoints"]["00"]["text"].lower()
               for row in rows if row["template_id"] == "registry_right_now")
    assert all("field report" in row["endpoints"]["00"]["text"].lower()
               for row in rows if row["template_id"] == "field_these_days")
