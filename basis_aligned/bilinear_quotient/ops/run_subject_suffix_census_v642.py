#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_exact_census pred_b_direct_route pred_c_stable_consumer
"""Five-port subject-number response census; managed queue only.

Six prefix and six suffix calls, 160 sequences per arm, zero fits/backwards.
Frozen original and historical fresh panels are OPENED. No selection or compression.
Nulls: no stable direct/individual-module reader; exact accounting still required.
"""
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
POLY = ROOT / 'basis_aligned/polynomial_causal'
OUT = ROOT / 'basis_aligned/bilinear_quotient/circuits/followups/subject_suffix_census_v642_result.json'
PREDICTIONS = dict(
    pred_a_exact_census='residual and margin effect relative closure <=1e-4, selected logits max error <=1e-4',
    pred_b_direct_route='carry aligned fraction >=.50 in every direction/template cell',
    pred_c_stable_consumer='one attention/MLP module has aligned fraction >=.25 in every cell')


def main():
    plan = dict(prefix_calls=6, suffix_calls=6, sequences_per_arm=160, fits=0,
                backwards=0, model_updates=0, execution_policy='managed_queue_only',
                predictions=PREDICTIONS, panels='original128 + historical_fresh32, opened')
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(plan)); return
    import torch
    import torch.nn.functional as F
    import circuit_fast_screen_producer as producer
    import circuit_fast_screen_candidate_subject_number_embedding_context_crossed_corrected_v1 as original
    import circuit_fast_screen_candidate_subject_number_rank1_fresh_confirmation as fresh
    import subject_number_sparse_graph_token_extraction_v1 as graph
    from disk_guard import guard_write
    if OUT.exists(): raise FileExistsError(OUT)
    torch.set_grad_enabled(False); torch.set_num_threads(4)
    tic = time.perf_counter()
    model = producer.Bilin18TorchBackend.load('cuda').model
    decoder = json.loads((POLY / 'SUBJECT_NUMBER_EMBEDDING_DECODER_FROZEN_V1_ARTIFACT.json').read_text())['frozen_decoder']
    axis = torch.tensor(decoder['axis'], device='cuda', dtype=torch.float64)
    unit = axis / axis.norm(); target_projection = float(decoder['threshold']) / float(axis.norm())
    batches = []; rows = original.build_rows()
    for start in (0, 64):
        entries = []
        for r in rows[start:start+64]:
            ans = r['native_answer_id']
            entries.append((r['token_ids'], r['subject_position'], [ans, 389 if ans == 318 else 318],
                            'original|' + r['number'] + '|' + r['template_id']))
        batches.append(entries)
    entries = []
    for r in fresh.build_rows():
        e = r['endpoints']['recipient']
        entries.append((e['ids'], len(e['ids'])-1, [e['answer_id'], e['foil_id']],
                        'historical_fresh|' + r['direction_id'] + '|' + r['template_id']))
    batches.append(entries)
    assert sum(map(len, batches)) == 160
    cells = {}; residual_errors = []; margin_errors = []; logits_errors = []
    counts = dict(prefix_calls=0, suffix_calls=0)

    def suffix(raw, x0, first, pos, answers):
        counts['suffix_calls'] += 1
        x = raw; batch = torch.arange(len(x), device=x.device)
        pieces = {'carry': x[batch, pos].double()}
        for layer in range(11, 18):
            block = model.transformer.h[layer]
            if layer > 11:
                x = block.lambdas[0]*x + block.lambdas[1]*x0
                pieces = {k: v*block.lambdas[0].double() for k, v in pieces.items()}
            a, first = block.attn(F.rms_norm(x, (x.shape[-1],)), first)
            x = x + a
            m = block.mlp(F.rms_norm(x, (x.shape[-1],)))
            x = x + m
            pieces[f'attn_{layer}'] = a[batch, pos].double()
            pieces[f'mlp_{layer}'] = m[batch, pos].double()
        endpoint = x[batch, pos]
        full = 30*torch.tanh(model.lm_head(F.rms_norm(endpoint, (endpoint.shape[-1],)))/30)
        selected = full.gather(1, answers).double()
        return endpoint.double(), pieces, selected

    for entries in batches:
        tokens = torch.tensor([e[0] for e in entries], device='cuda')
        pos = torch.tensor([e[1] for e in entries], device='cuda')
        answers = torch.tensor([e[2] for e in entries], device='cuda')
        batch = torch.arange(len(entries), device='cuda')
        assert bool((pos == tokens.shape[1]-1).all())
        initial = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)).float()
        subject = initial[batch, pos].double(); projection = subject @ unit
        orthogonal = subject - projection[:, None]*unit
        scale = ((subject.square().sum(1)-target_projection**2)/orthogonal.square().sum(1)).sqrt()
        removed = initial.clone()
        removed[batch, pos] = (target_projection*unit+scale[:, None]*orthogonal).float()
        rb, x0, first, pb, _ = graph._capture(model, initial, torch, F)
        rr, _, _, pr, _ = graph._capture(model, removed, torch, F)
        counts['prefix_calls'] += 2
        edited = rb.clone()
        edited[batch, pos] += sum((y-x)[batch, pos] for x,y in zip(pb,pr)).float()
        xb, cb, yb = suffix(rb, x0, first, pos, answers)
        xe, ce, ye = suffix(edited, x0, first, pos, answers)
        delta = xe-xb; pieces = {k: ce[k]-cb[k] for k in cb}
        residual_errors.append(float((sum(pieces.values())-delta).norm()/delta.norm()))
        eps = torch.finfo(torch.float32).eps
        sb = (xb.square().mean(-1)+eps).sqrt(); se = (xe.square().mean(-1)+eps).sqrt()
        U = model.lm_head.weight[answers].double()
        zb = torch.einsum('bvd,bd->bv', U, xb)/sb[:,None]
        ze = torch.einsum('bvd,bd->bv', U, xe)/se[:,None]
        fb = 30*torch.tanh(zb/30); fe = 30*torch.tanh(ze/30)
        logits_errors.append(float(torch.maximum((fb-yb).abs(),(fe-ye).abs()).max()))
        dz = ze-zb
        safe = torch.where(dz.abs()>1e-12, dz, torch.ones_like(dz))
        slope = torch.where(dz.abs()>1e-12, (fe-fb)/safe, 1-torch.tanh(zb/30).square())
        reader = slope[:,0,None]*U[:,0] - slope[:,1,None]*U[:,1]
        # Damage is baseline minus edited margin; every component follows this sign.
        attribution = {k: -(v*reader).sum(-1)/se for k,v in pieces.items()}
        attribution['final_normalization'] = -(xb*reader).sum(-1)*(1/se-1/sb)
        damage = (yb[:,0]-yb[:,1])-(ye[:,0]-ye[:,1])
        predicted = sum(attribution.values())
        margin_errors.append(float((predicted-damage).norm()/damage.norm()))
        for i,e in enumerate(entries):
            c = cells.setdefault(e[3], dict(target=[], contributions={k:[] for k in attribution}))
            c['target'].append(float(damage[i]))
            for k,v in attribution.items(): c['contributions'][k].append(float(v[i]))
        print('batch',len(entries),'relative closure',margin_errors[-1],flush=True)
    import numpy as np
    for c in cells.values():
        y = np.asarray(c['target']); den = float(y@y)
        c['target_rms'] = float(np.sqrt(np.mean(y*y)))
        c['stats'] = {k:dict(aligned_fraction=float(np.asarray(v)@y/max(den,1e-30)),
                              rms_ratio=float(np.linalg.norm(v)/max(np.linalg.norm(y),1e-30)))
                      for k,v in c['contributions'].items()}
    a = max(residual_errors+margin_errors) <= 1e-4 and max(logits_errors)<=1e-4 and counts=={'prefix_calls':6,'suffix_calls':6}
    b = a and all(c['stats']['carry']['aligned_fraction']>=.5 for c in cells.values())
    modules = [f'{kind}_{i}' for i in range(11,18) for kind in ('attn','mlp')]
    stable = [k for k in modules if all(c['stats'][k]['aligned_fraction']>=.25 for c in cells.values())]
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), plan=plan, counts=counts,
        predictions=dict(zip(PREDICTIONS,[bool(a),bool(b),bool(a and stable)])),
        stable_consumers=stable, cells=cells,
        instrument=dict(max_residual_relative_error=max(residual_errors),max_margin_relative_error=max(margin_errors),
                        max_selected_logit_absolute_error=max(logits_errors)),
        wall_seconds=time.perf_counter()-tic,
        scope='Native response census, opened panels. Secant reader depends on both endpoints; not an extracted predictor or causal sufficiency test.')
    payload=json.dumps(result,indent=2)+'\n'; guard_write(len(payload.encode()),str(OUT.parent),'subject suffix census')
    OUT.write_text(payload)
    print(json.dumps({k:result[k] for k in ('predictions','stable_consumers','instrument','wall_seconds')},indent=2))
    assert a, 'invalid census; preserve result and repair instrument'


if __name__ == '__main__': main()
