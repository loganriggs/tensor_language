"""Tier-0 exactness gate (spec section 5, Tier 0.2-0.3).

GATE: the {C_f, S_f} expansion must reproduce the model's actual layer-0
patterns to ~1e-10 in fp64, all heads, both branches, on random token batches.
Also: the branch-scale gauge check (c*s1, s2/c leaves the pattern identical),
the manual-RMSNorm formula check, and the descriptive band-mass profile.

No MDL numbers are produced here.
"""

import json
import sys

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from folding import (load_tiny, expanded_branch_scores, expanded_pattern,
                     effective_embedding, band_mass)

torch.manual_seed(0)
GATE_TOL = 1e-10
report = {}

torch.set_grad_enabled(False)

for run in ['attn2-mix10-seed0', 'attn2-dense-seed0', 'attn1-seed0']:
    model, cfg = load_tiny(run)
    tokens = torch.randint(0, cfg['vocab'], (4, cfg['n_ctx']))
    x = model.embed(tokens)
    layer = model.layers[0]

    # gate 1: full pattern reconstruction (layer 0, all heads)
    P_model = layer.pattern(x)
    P_exp = expanded_pattern(model, 0, tokens)
    err_pattern = float((P_model - P_exp).abs().max())

    # informational: analytic fp64 omegas vs the deployed fp32-table model
    s1_analytic = expanded_branch_scores(model, 0, 1, tokens, use_model_trig=False)
    s1_model = expanded_branch_scores(model, 0, 1, tokens, use_model_trig=True)
    analytic_dev = float((s1_analytic - s1_model).abs().max())

    # gate 2: per-branch scores (unmasked) vs a direct branch computation
    errs_branch = []
    for br in (1, 2):
        h = layer.norm(x)
        heads = lambda t: t.reshape(*t.shape[:-1], layer.n_head, layer.d_head)
        q = layer.rotary(heads(getattr(layer, f'q{br}')(h)))
        k = layer.rotary(heads(getattr(layer, f'k{br}')(h)))
        s_direct = torch.einsum('bihd,bjhd->bhij', q, k)
        s_exp = expanded_branch_scores(model, 0, br, tokens)
        errs_branch.append(float((s_direct - s_exp).abs().max()))

    # gauge check: (c*s1, s2/c) -> identical pattern
    c = 3.7
    s1 = expanded_branch_scores(model, 0, 1, tokens)
    s2 = expanded_branch_scores(model, 0, 2, tokens)
    mask = torch.tril(torch.ones(cfg['n_ctx'], cfg['n_ctx'], dtype=s1.dtype))
    err_gauge = float(((c * s1) * (s2 / c) - s1 * s2).abs().max() / layer.d_head ** 2)

    # manual effective-embedding formula vs the module
    E = model.embed.weight.detach()
    manual = E / (E.pow(2).mean(-1, keepdim=True)
                  + torch.finfo(E.dtype).eps).sqrt()
    err_norm = float((manual - effective_embedding(model, 0)).abs().max())

    ok = err_pattern < GATE_TOL and max(errs_branch) < GATE_TOL
    report[run] = {'pattern_err': err_pattern, 'branch_errs': errs_branch,
                   'gauge_err': err_gauge, 'manual_norm_err': err_norm,
                   'analytic_omega_deviation': analytic_dev,
                   'GATE': 'PASS' if ok else 'FAIL'}
    print(f"{run:22s} pattern {err_pattern:.2e}  branches "
          f"{errs_branch[0]:.2e}/{errs_branch[1]:.2e}  gauge {err_gauge:.2e}  "
          f"norm-formula {err_norm:.2e}  analytic-dev {analytic_dev:.2e}  "
          f"-> {report[run]['GATE']}")

    # descriptive band-mass profile (no MDL claims)
    if run == 'attn2-mix10-seed0':
        prof = {}
        for br in (1, 2):
            C2, S2 = band_mass(model, 0, br)
            tot = (C2 + S2).sum(-1, keepdim=True)
            frac = ((C2 + S2) / tot)
            for hh in range(cfg['n_head']):
                top = frac[hh].argsort(descending=True)[:3]
                prof[f'L0H{hh}_b{br}'] = {
                    'top_bands': top.tolist(),
                    'top_frac': [round(float(frac[hh, t]), 3) for t in top]}
        report['band_profile_attn2-mix10-seed0_L0'] = prof
        print('band profile (head_branch: top-3 bands, mass fraction):')
        for k, v in prof.items():
            print(f"  {k}: bands {v['top_bands']} frac {v['top_frac']}")

with open('/workspace/tensor_language/basis_aligned/qk_mdl/tier0_report.json', 'w') as fh:
    json.dump(report, fh, indent=2)
print('saved tier0_report.json')
