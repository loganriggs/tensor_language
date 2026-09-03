from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).with_name("numbered_list_factor_localization_rung573.py")
SPEC = importlib.util.spec_from_file_location("r573", MODULE_PATH)
r573 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(r573)


def toy_tensors():
    torch.manual_seed(573)
    pattern = torch.randn(1, r573.N_HEAD, 3, 3)
    own = torch.randn(1, 3, r573.N_HEAD, r573.HEAD_D)
    cached = torch.randn_like(own)
    value = own + cached
    head_output = torch.einsum("bhqk,bkhd->bqhd", pattern, value)
    return {"pattern": pattern, "own": own, "cached": cached,
            "value": value, "head_output": head_output}


def donor_for(tensors):
    labels = []
    for source in (0, 1):
        labels.append({
            "score": torch.stack([tensors["pattern"][0, head, 2, source] + 0.7
                                  for head in r573.HEADS]),
            "value": torch.stack([tensors["value"][0, source, head] + 1.1
                                  for head in r573.HEADS]),
            "cached": torch.stack([tensors["cached"][0, source, head] + 1.3
                                   for head in r573.HEADS]),
            "own": torch.stack([tensors["own"][0, source, head] + 1.7
                                for head in r573.HEADS]),
        })
    return [{"complete": torch.randn(len(r573.HEADS), r573.HEAD_D), "labels": labels}]


def test_complete_head_arm_replaces_only_registered_heads_at_final_query():
    tensors = toy_tensors()
    donor = donor_for(tensors)
    changed = r573.modify_head_output(
        tensors["head_output"], tensors, torch.tensor([2]), [[0, 1]], donor, "complete_heads")
    for slot, head in enumerate(r573.HEADS):
        assert torch.equal(changed[0, 2, head], donor[0]["complete"][slot])
    untouched = [head for head in range(r573.N_HEAD) if head not in r573.HEADS]
    assert torch.equal(changed[0, :, untouched], tensors["head_output"][0, :, untouched])
    assert torch.equal(changed[0, :2], tensors["head_output"][0, :2])


def test_final_cached_arm_is_exact_source_term_replacement():
    tensors = toy_tensors()
    donor = donor_for(tensors)
    changed = r573.modify_head_output(
        tensors["head_output"], tensors, torch.tensor([2]), [[0, 1]], donor,
        "final_label_cached_value")
    source = 1
    for slot, head in enumerate(r573.HEADS):
        score = tensors["pattern"][0, head, 2, source]
        expected_delta = score * (donor[0]["labels"][source]["cached"][slot]
                                  - tensors["cached"][0, source, head])
        observed_delta = changed[0, 2, head] - tensors["head_output"][0, 2, head]
        assert torch.allclose(observed_delta, expected_delta, atol=1e-6, rtol=1e-6)


def test_selection_order_is_semantic_and_fixed():
    reports = {arm: {"passed": arm in {"all_label_joint", "final_label_value"}}
               for arm in r573.SELECTION_ORDER}
    assert r573.choose(reports)["selected_arm"] == "final_label_value"
