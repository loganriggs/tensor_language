from __future__ import annotations

import torch

import correct_mlp2_cmr_v1_suffix_v2_overlaps as correction


def test_corrected_overlaps_use_numeric_tensor_values() -> None:
    supports = {}
    for offset, name in enumerate(correction.SUPPORT_NAMES):
        supports[name] = torch.arange(offset * 100, offset * 100 + 512)
    result = correction.corrected_overlaps(supports)
    assert result["SUFFIX_DERANGED"] == {
        "intersection": 412,
        "union": 612,
        "jaccard": 412 / 612,
    }
    assert result["SUFFIX_HASH_RANDOM"]["intersection"] == 12

