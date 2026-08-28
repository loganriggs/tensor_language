from __future__ import annotations

import json

import freeze_finite_horizon_tangent_plan as freeze


def test_frozen_plan_matches_authoritative_cache_and_is_document_disjoint(tmp_path) -> None:
    original = freeze.OUT
    freeze.OUT = tmp_path / "plan.json"
    try:
        freeze.main()
        result = json.loads(freeze.OUT.read_text())
    finally:
        freeze.OUT = original
    assert result["status"] == "frozen_cpu_plan_no_gpu_authority"
    assert result["row_cache"]["tensor_raw_sha256"] == freeze.EXPECTED_RAW_SHA256
    assert result["unique_source_documents"] == 33
    assert sum(split["rows"] for split in result["splits"].values()) == 96
    assert sum(split["source_documents"] for split in result["splits"].values()) == 33
    primary = set(result["splits"]["primary"]["row_indices"])
    replication = set(result["splits"]["replication"]["row_indices"])
    assert primary.isdisjoint(replication)
    assert primary | replication == set(range(96))
    assert result["operator"]["primary_shape_at_cut3"][1] == 96
    assert result["operator"]["replication_shape_at_cut3"][1] == 96
    assert len(result["scored_positions"]) == 96
    assert all(64 <= position < 256 for position in result["scored_positions"])
    assert result["scored_positions"][0] == freeze.scored_position("n96_skip80:0")
