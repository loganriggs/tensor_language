import causal_response_factorization_v1_candidate_price_audit as audit


def test_frozen_grid_has_exactly_51_unique_fits():
    assert len(audit.RANK_PAIRS) == 17
    assert len(set(audit.RANK_PAIRS)) == 17
    assert len(audit.SEEDS) == 3
    assert len(audit.RANK_PAIRS) * len(audit.SEEDS) == 51
    assert sum(audit.OWNER_GROUP_SIZES) == audit.SOURCES


def test_structured_price_uses_separate_persistent_and_document_coordinates():
    assert audit.structured_price(32, 0) == (3200, 32)
    assert audit.structured_price(0, 8) == (2840, 48)
    assert audit.structured_price(16, 4) == (3020, 40)


def test_literal_dense_match_is_rank_zero_everywhere():
    rows = audit.audit_rows()
    assert max(row.persistent_values for row in rows) == 3200
    assert audit.PHASES * audit.SOURCES * audit.TARGETS == 4802
    assert all(row.strict_dense_matched_rank == 0 for row in rows)
    assert {row.amortized_total_dense_rank for row in rows} == {0, 1, 2}


def test_receipt_makes_amortized_view_noncontrolling_and_reads_no_response():
    value = audit.build_receipt()
    assert value["structured_fits"] == 51
    assert value["conclusions"][
        "strict_dense_matched_rank_zero_for_every_candidate"
    ] is True
    assert value["conclusions"]["amortized_price_is_a_distinct_noncontrolling_view"] is True
    assert value["conclusions"]["response_values_read"] is False
