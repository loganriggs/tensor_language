import importlib.util
from pathlib import Path

import torch


PATH = Path(__file__).with_name("frozen_ship_oracle_v2.py")
SPEC = importlib.util.spec_from_file_location("frozen_ship_oracle_v2", PATH)
PIPELINE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PIPELINE)


def gain(mean):
    return {"global": {"mean": mean, "ci95": [mean - 0.01, mean + 0.01]}}


def test_exact_fineweb_gate_replaces_interpolated_null_gate():
    result = {"site_decisions": {}, "paired_gains": {}}
    for site in range(3):
        key = str(site)
        result["site_decisions"][key] = {
            "full_oracle_ci95_lower_gt_zero": True,
            "content_positive_both_splits": True,
            "content_beats_matched_null95_heldout": True,
        }
        heldout = {"content": gain(0.10)}
        for index in range(20):
            heldout[f"null_{index:02d}"] = gain(0.05)
        result["paired_gains"][key] = {"heldout": heldout}
    # Site 1 has one tied null: interpolated quantile may pass, exact test must fail.
    result["paired_gains"]["1"]["heldout"]["null_19"]["global"]["mean"] = 0.10

    decisions = PIPELINE.exact_fineweb_decisions(result)

    assert result["training_license_sites"] == [0, 2]
    assert decisions["0"]["exact_twenty_null_test"]["exact_one_sided_p"] == 1 / 21
    assert decisions["1"]["exact_twenty_null_test"]["passes_5pct"] is False
    assert decisions["1"]["preliminary_interpolated_null95_gate"] is True


def test_cpu_tree_detaches_tensors_without_changing_structure():
    source = {"x": (torch.arange(4), [True, 3.0]), "name": "ship"}
    copied = PIPELINE.cpu_tree(source)
    assert copied["name"] == "ship"
    assert isinstance(copied["x"], tuple)
    assert torch.equal(copied["x"][0], source["x"][0])
    source["x"][0][0] = 99
    assert int(copied["x"][0][0]) == 0


def test_authoritative_source_explicitly_upgrades_preliminary_authority():
    source = PATH.read_text()
    assert '"authority": "canonical_fineweb"' in source
    assert '"authorized_for_scored_experiments": True' in source
