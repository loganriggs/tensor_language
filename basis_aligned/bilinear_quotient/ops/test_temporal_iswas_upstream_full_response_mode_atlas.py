#!/usr/bin/env python3
"""Focused CPU tests for complete head/module response patching."""

from types import SimpleNamespace
import torch

import run_temporal_iswas_upstream_full_response_mode_atlas_v1 as runner


def test_attention_head_isolation():
    batch=SimpleNamespace(semantic_positions=(1,2))
    arguments=(torch.zeros(2,4,18),torch.ones(1))
    donor=torch.arange(2*4*18,dtype=torch.float32).reshape(2,4,18)
    changed=runner.patch_hook(batch,donor,"L6H7",2)(None,arguments)[0]
    expected=torch.zeros_like(changed)
    expected[0,:2,14:16]=donor[0,:2,14:16]
    expected[1,:3,14:16]=donor[1,:3,14:16]
    assert torch.equal(changed,expected)
    assert torch.equal(arguments[0],torch.zeros_like(arguments[0]))


def test_complete_mlp_prefix():
    batch=SimpleNamespace(semantic_positions=(0,2))
    output=torch.zeros(2,4,5);donor=torch.arange(40,dtype=torch.float32).reshape(2,4,5)
    changed=runner.patch_hook(batch,donor,"MLP7",2)(None,(),output)
    expected=torch.zeros_like(changed);expected[0,:1]=donor[0,:1];expected[1,:3]=donor[1,:3]
    assert torch.equal(changed,expected)
    assert torch.equal(output,torch.zeros_like(output))


def test_vector_statistics():
    stats=runner.vector_stats(torch,torch.tensor([1.,2.,3.]),torch.tensor([2.,4.,6.]))
    assert abs(stats["signed_projection"]-.5)<1e-7
    assert abs(stats["cosine"]-1.)<1e-7
    assert abs(stats["norm_ratio"]-.5)<1e-7


if __name__=="__main__":
    test_attention_head_isolation();test_complete_mlp_prefix();test_vector_statistics()
    print("PASS: upstream full-response causal patch kernel")
