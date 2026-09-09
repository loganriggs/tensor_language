#!/usr/bin/env python3
"""CPU-only weight diagnostic: are arbitrary token norm gains the missing operation?

Fixed before inspection: normalized joint kernel invariant residual <=.10 in >=2
of4 heads licenses investigating an explicit gain-times-equality candidate. Otherwise
close this amplitude-only explanation. No forward outcomes, fitting, head selection,
or circuit credit. The restored candidate needs separate full-output/causal tests.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path('/workspace/tensor_language')
POLY = ROOT/'basis_aligned/polynomial_causal'
sys.path[:0] = [str(ROOT), str(POLY)]


def main():
    import torch
    import equality_router_reference as R
    from hop_ablate import load
    started = time.perf_counter()
    torch.set_num_threads(2)
    os.chdir(ROOT)
    model, _ = load('attn-mlp-attn-rms-seed0')
    model = model.double().eval()
    with torch.inference_mode():
        layer = model.layers[0]
        n = layer.norm(model.embed.weight)
        norm = {name: getattr(layer, name)(n).reshape(29, 4, 32).norm(dim=-1).T
                for name in ('q1', 'q2', 'k1', 'k2')}
        query_gain = norm['q1']*norm['q2']
        key_gain = norm['k1']*norm['k2']
        assert query_gain.min() > 0 and key_gain.min() > 0
        product_gain = query_gain[:, None, :, None]*key_gain[:, None, None, :]
        kernel = R.lag_kernel(model)
        normalized = kernel/product_gain
        ids = torch.arange(29)
        orbit = R.orbit_index(ids[:, None], ids[None, :])
        restored = R.orbit_projection(normalized)[..., orbit]*product_gain
        rows = []
        for h in range(4):
            relative = lambda a, b: float((a-b).norm()/a.norm())
            e = relative(normalized[h], R.orbit_projection(normalized[h])[..., orbit])
            rows.append(dict(head=h,
                             raw_relative_residual=relative(kernel[h], R.orbit_projection(kernel[h])[..., orbit]),
                             normalized_relative_residual=e,
                             restored_relative_residual=relative(kernel[h], restored[h]),
                             query_gain_min=float(query_gain[h].min()), query_gain_max=float(query_gain[h].max()),
                             key_gain_min=float(key_gain[h].min()), key_gain_max=float(key_gain[h].max()),
                             supports_amplitude_hypothesis=e <= .10))
        support=sum(r['supports_amplitude_hypothesis'] for r in rows)
        result=dict(terminal='gain_candidate_eligible' if support>=2 else 'amplitude_only_explanation_rejected',
                    supported_heads=support, rows=rows, extra_gain_constants=2*4*29,
                    scope='CPU weight-domain diagnostic; no full-output or causal claim; native FP32 RoPE lag approximation retained only for this geometric screen.',
                    seconds=time.perf_counter()-started,
                    hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                            (ROOT/'runs_hop/attn-mlp-attn-rms-seed0/model.pt', Path(__file__),
                             POLY/'equality_router_reference.py', POLY/'EQUALITY_ROUTER_V2_ABSOLUTE_POSITION_RESULT.json')})
        (POLY/'EQUALITY_ROUTER_GAIN_V1_RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
