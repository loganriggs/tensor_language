#!/usr/bin/env python3

import aspectual_tense_cross_task_eval as cross


def test_endpoint_and_summary_contract():
    assert cross.verify_contract()


def test_authority_pairing_contract():
    pairs = cross.paired_rows()
    assert len(pairs) == 16
    assert all(left["reporter"] == right["reporter"] for left, right in pairs)
    assert all(left["direction_id"] == right["direction_id"] for left, right in pairs)
