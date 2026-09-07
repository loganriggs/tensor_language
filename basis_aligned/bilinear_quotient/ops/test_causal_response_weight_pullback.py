#!/usr/bin/env python3
"""Focused CPU tests for canonical causal-response weight pullback."""
import torch
import causal_response_weight_pullback as p


def test_exact_replay_and_orthogonal_gauge():
    gen=torch.Generator(device="cpu").manual_seed(20260907)
    c=torch.randn(32,8,generator=gen);r=torch.randn(32,8,generator=gen)
    s=torch.linalg.qr(torch.randn(24,8,generator=gen),mode="reduced").Q
    ss=torch.linspace(.7,1.3,32);ts=torch.linspace(1.4,.8,32)
    base=p.pullback(torch,c,r,s,ss,ts,rank=8)
    assert torch.allclose(base["source_score_replay"],base["source_scores"],atol=2e-5,rtol=2e-5)
    assert torch.allclose(base["reader_score_replay"],base["reader_scores"],atol=2e-5,rtol=2e-5)
    assert torch.allclose(base["rank_response"],(c*ss[:,None])@(r*ts[:,None]).T,atol=3e-5,rtol=3e-5)
    gauge=torch.linalg.qr(torch.randn(8,8,generator=gen)).Q
    rotated=p.pullback(torch,c@gauge,r@gauge,s@gauge,ss,ts,rank=8)
    # SVD column signs are arbitrary; compare physical projectors.
    for key in ("physical_source_covectors","physical_reader_covectors"):
        qb=torch.linalg.qr(base[key],mode="reduced").Q;qr=torch.linalg.qr(rotated[key],mode="reduced").Q
        assert torch.allclose(qb@qb.T,qr@qr.T,atol=2e-5,rtol=2e-5)


if __name__=="__main__":
    test_exact_replay_and_orthogonal_gauge();print("PASS")
