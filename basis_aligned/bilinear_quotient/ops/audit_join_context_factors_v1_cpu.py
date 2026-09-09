#!/usr/bin/env python3
"""Opened-case exact score/value factor diagnosis, no fitting or promotion."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import json
import signal
import sys
import time
import torch
ROOT = Path('/workspace/tensor_language'); POLY = ROOT/'basis_aligned/polynomial_causal'
sys.path[:0] = [str(ROOT), str(POLY), str(ROOT/'basis_aligned/bilinear_quotient/ops')]
import join_contribution_context_reference as R
import run_equality_router_v1 as score
from hop_ablate import load


def factors(model, tokens, masks, heads):
    x = model.embed(tokens)
    for layer in model.layers[:2]: x = layer(x)
    layer = model.layers[2]; p = layer.pattern(x)
    v = layer.v(layer.norm(x)).reshape(len(tokens), 51, 4, layer.d_head)
    return {j: (p[:, heads[j]]*mask, v[:, :, heads[j]]) for j, mask in masks.items()}


def contract(model, pairs, heads):
    layer = model.layers[2]; out = {}
    for j, (a, v) in pairs.items():
        h = heads[j]; z = torch.einsum('bts,bsd->btd', a, v)
        out[j] = (z @ layer.o.weight[:, h*layer.d_head:(h+1)*layer.d_head].T)*layer.scale
    return out


def main():
    os.chdir(ROOT); torch.set_num_threads(2); signal.alarm(300); started = time.perf_counter()
    out = POLY/'JOIN_CONTEXT_FACTOR_AUDIT_V1.json'; assert not out.exists()
    path = POLY/'JOIN_CONTRIBUTION_CONTEXT_V1_ROWS.pt'
    rows = torch.load(path, map_location='cpu', weights_only=True)
    model, _ = load('attn4-rms-seed0'); model = model.double().eval(); results = {}; replay = 0.
    with torch.inference_mode():
        for pop, worlds in R.populations().items():
            mixed = {kind: {s: [] for s in ('a','b','both')} for kind in ('donor_score','donor_value')}
            vector_parts = []
            for wi, w in enumerate(worlds):
                assert torch.equal(w['recipient'], rows[pop]['tokens'][wi*8:wi*8+8])
                donor = factors(model, w['donor'], w['masks'], w['heads'])
                own = factors(model, w['recipient'][:1], w['masks'], w['heads'])
                pairs = {'donor_score': {j:(donor[j][0],own[j][1]) for j in own},
                         'donor_value': {j:(own[j][0],donor[j][1]) for j in own}}
                native_write = contract(model, own, w['heads']); donor_write = contract(model, donor, w['heads'])
                writes = {kind: contract(model, p, w['heads']) for kind, p in pairs.items()}
                for j in own:
                    replay = max(replay, float((native_write[j]-rows[pop]['writes'][wi]['own'][j]).abs().max()),
                                 float((donor_write[j]-rows[pop]['writes'][wi]['donor'][j]).abs().max()))
                    route = 'forward' if w['heads'][j] == 1 else 'backward'
                    da = writes['donor_score'][j]-native_write[j]
                    dv = writes['donor_value'][j]-native_write[j]
                    interaction = donor_write[j]-native_write[j]-da-dv
                    vector_parts.append({'world':w['world'],'arrangement':w['arrangement'],'route':route,
                                         'score_sq':float(da.square().sum()),'value_sq':float(dv.square().sum()),
                                         'interaction_sq':float(interaction.square().sum()),
                                         'total_sq':float((donor_write[j]-native_write[j]).square().sum())})
                for i in range(0,8,4):
                    tok = w['recipient'][i:i+4]
                    for kind in mixed:
                        for suffix, selected in (('a',(0,)),('b',(1,)),('both',(0,1))):
                            with R.intervene(model,w['masks'],w['heads'],selected,writes[kind]): logits=model(tok)
                            mixed[kind][suffix].append(logits)
            native = rows[pop]['logits']['native']; metadata = rows[pop]['metadata']; grouped = {}
            for arrangement in range(2):
                select = torch.tensor([m['arrangement']==arrangement for m in metadata]); cases = {}
                for kind in mixed:
                    cases[kind] = {}
                    for suffix, values in mixed[kind].items():
                        actual = torch.cat(values)[select]; expected = native[select]
                        cut = rows[pop]['logits']['remove_'+suffix][select]
                        cases[kind][suffix] = {'all':score.distribution(actual,expected,torch),
                                              'query':score.distribution(actual[:,-1],expected[:,-1],torch),
                                              'query_effect':score.effect_error((actual-cut)[:,-1],(expected-cut)[:,-1],torch)}
                grouped[str(arrangement)] = cases
            result = {'arrangements':grouped,'write_difference_factor_terms':vector_parts}
            results[pop] = result
            print(json.dumps({'population':pop,'summary': {arr:{kind:{s:(v['query']['mean_kl'],v['query_effect']['relative_l2']) for s,v in cases.items()} for kind,cases in groups.items()} for arr,groups in grouped.items()}}),flush=True)
    assert replay <= 1e-9, replay
    receipt = {'analysis':'join_context_factor_audit_v1','status':'opened_case_diagnostic_no_promotion',
               'source_rows_sha256':score.digest(path),'code_sha256':score.digest(Path(__file__)),
               'native_write_cpu_gpu_max_error':replay,'populations':results,'wall_seconds':time.perf_counter()-started}
    out.write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
