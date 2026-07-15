"""Tier-2 folding for the Elriggs family (layer 0, exact).

Layer-0 attn input: h_t = rms_norm(wte[t]) — the block's (l0*x + l1*x0) scalar
is killed by the pre-attn rms_norm, so lambdas drop out at layer 0.
Per branch b, head hh: qhat_b(t) = rms_norm( (h_t @ W_qb^T)[head hh] ) in R^128
(unit-RMS rows), khat_b likewise, and

    s_b(t_q @ i, t_k @ j) = qhat(t_q)^T R_{i-j} khat(t_k) / head_dim
      = sum_f [ cos(w_f d) * (qa ka + qb kb) + sin(w_f d) * (qb ka - qa kb) ] / hd

(rotation-sign convention OPPOSITE to the tiny models: S_f = qb ka - qa kb;
the fp64 gate below verifies.) The branch's whole folded object is the factor
pair (qhat, khat) — V x 128 each; every codebook acts on factors, never V x V.
"""

import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import (load_elriggs, reference_forward, rope_tables,
                         build_eval_tokens, eval_ce)


@torch.no_grad()
def branch_factors(m, branch, dtype=torch.float64):
    """(qhat, khat): (V, n_head, head_dim) unit-RMS per-token factors, layer 0."""
    a = m.transformer.h[0].attn
    nh, hd = a.n_head, a.head_dim
    E = m.transformer.wte.weight.detach().to(dtype)
    h = F.rms_norm(E, (E.shape[-1],))
    Wq = (a.c_q if branch == 1 else a.c_q2).weight.detach().to(dtype)
    Wk = (a.c_k if branch == 1 else a.c_k2).weight.detach().to(dtype)
    qh = F.rms_norm((h @ Wq.T).view(-1, nh, hd), (hd,))
    kh = F.rms_norm((h @ Wk.T).view(-1, nh, hd), (hd,))
    return qh, kh


@torch.no_grad()
def scores_from_factors(qh, kh, tokens, hd, table_dtype='bf16'):
    """(B, n_head, T, T) branch scores from factors, via the C/S expansion."""
    B, T = tokens.shape
    Fq = qh[tokens]                                     # (B,T,nh,hd)
    Fk = kh[tokens]
    d = hd // 2
    cos, sin = rope_tables(T, hd, tokens.device, qh.dtype, table_dtype)
    # difference tables from the same per-position tables the model uses
    cosD = torch.einsum('if,jf->ijf', cos, cos) + torch.einsum('if,jf->ijf', sin, sin)
    sinD = torch.einsum('if,jf->ijf', sin, cos) - torch.einsum('if,jf->ijf', cos, sin)
    qa, qb = Fq[..., :d], Fq[..., d:]
    ka, kb = Fk[..., :d], Fk[..., d:]
    s = (torch.einsum('bihf,bjhf,ijf->bhij', qa, ka, cosD)
         + torch.einsum('bihf,bjhf,ijf->bhij', qb, kb, cosD)
         + torch.einsum('bihf,bjhf,ijf->bhij', qb, ka, sinD)
         - torch.einsum('bihf,bjhf,ijf->bhij', qa, kb, sinD))
    return s / hd


if __name__ == '__main__':
    import json
    torch.manual_seed(0)
    report = {}
    m, cfg = load_elriggs('bilin18', dtype=torch.float64)
    hd = m.transformer.h[0].attn.head_dim
    TOK = build_eval_tokens(n_chunks=2, seq_len=257)
    tokens = TOK[:, :-1].cuda()

    captured = {}

    def capture(li, s1, s2):
        if li == 0:
            captured['s1'], captured['s2'] = s1.clone(), s2.clone()
        return s1, s2

    reference_forward(m, tokens, table_dtype='bf16', score_patch=capture)
    errs = []
    for br in (1, 2):
        qh, kh = branch_factors(m, br)
        s_fold = scores_from_factors(qh, kh, tokens, hd, table_dtype='bf16')
        err = float((s_fold - captured[f's{br}']).abs().max())
        errs.append(err)
        print(f'bilin18 branch {br}: fold-vs-reference max err {err:.2e} '
              f'({"PASS" if err < 1e-10 else "FAIL"})')
    report['bilin18'] = {'branch_errs': errs,
                         'GATE': 'PASS' if max(errs) < 1e-10 else 'FAIL'}
    with open('/workspace/tensor_language/basis_aligned/qk_mdl/tier2_fold_gate.json', 'w') as fh:
        json.dump(report, fh, indent=2)
    print('saved tier2_fold_gate.json')
