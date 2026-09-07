import tiktoken

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v16 as candidate


def test_v16_authority_is_deterministic_and_balanced():
    rows = candidate.build_rows()
    assert len(rows) == 64
    assert candidate.validate_rows(rows) == candidate.authority_sha256()
    assert {panel: sum(row["transform_id"] == panel for row in rows)
            for panel in ("A1", "A2", "P", "C")} == {panel: 16 for panel in ("A1", "A2", "P", "C")}
    assert {row["direction_id"] for row in rows if row["transform_id"] in ("A1", "A2")} == {
        "present_to_past", "past_to_present"
    }


def test_v16_future_patch_scope_is_exactly_aligned_targets_and_p():
    encoding = tiktoken.get_encoding("gpt2")
    rows = candidate.build_rows()
    lengths = {
        panel: [(len(encoding.encode(row["base_text"])), len(encoding.encode(row["donor_text"])))
                for row in rows if row["transform_id"] == panel]
        for panel in ("A1", "A2", "P", "C")
    }
    assert all(base == donor for panel in ("A1", "A2", "P") for base, donor in lengths[panel])
    # The inherited canonical C bank is capability-only here. A future response patch must use the
    # separately aligned v15 C authority rather than silently position-patching these rows.
    assert all(base != donor for base, donor in lengths["C"])
