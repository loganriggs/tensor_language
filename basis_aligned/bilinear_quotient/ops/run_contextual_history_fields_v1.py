#!/usr/bin/env python3
"""Fixed-position composite-key/payload counterfactuals on fresh documents.

pred_a_instrument: parser plus full native/folded logits1e-9; base gold P>=.8 pergroup.
pred_b_key_payload_roles: payload foil P>=.8; control foil increase<=.1; E/K gold
loss>=.5; EKP foil increase<=.1, every population/hop.
pred_c_joint_retarget: retarget foil P>=.8 and gain over EKP>=.5, every group.
pred_d_factor_specialization: fixed RMS effects .5/.1; IID nomination confirmed OOD.
Null: key/payload account fails or native QK factors do not separately encode fields.
Price:128docs x8arms x2executors,batch4,T239,29logits,1800s,tensors<256MiB.
GPU only via bqrunner. Full probability executor retains native contextual weights.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_key_payload_roles pred_c_joint_retarget pred_d_factor_specialization
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path('/workspace/tensor_language')
POLY = ROOT/'basis_aligned/polynomial_causal'
SOURCE = Path(__file__)
REF = POLY/'contextual_history_reference.py'
PREREG = POLY/'CONTEXTUAL_HISTORY_FIELDS_V1_PREREGISTRATION.md'
SCORER = ROOT/'basis_aligned/bilinear_quotient/ops/run_equality_router_v1.py'
OUT = POLY/'CONTEXTUAL_HISTORY_FIELDS_V1_RESULT.json'
EXPECTED = '46493a0be023f0b531884714d22048130cff90a56f50fdf0779f76102764a7aa'
ARMS = ('base', 'P', 'E', 'K', 'EK', 'EKP', 'RETARGET', 'CONTROL')
QUERY_POS = 178
SOURCE_POS = 115


def bound_hash():
    return hashlib.sha256(REF.read_bytes()+PREREG.read_bytes()+SCORER.read_bytes()).hexdigest()


def population(torch, kind):
    from hop_data import _fpow
    g = torch.Generator().manual_seed(5909 if kind == 'iid' else 5910)
    if kind == 'iid':
        cyc = torch.rand(64, 24, generator=g).argsort(1)
        fmap = torch.empty_like(cyc).scatter_(1, cyc, cyc.roll(-1, 1))
    else:
        fmap = torch.rand(64, 24, generator=g).argsort(1)
    order = torch.rand(64, 24, generator=g).argsort(1)
    bindings = torch.stack((order, fmap.gather(1, order)), -1).flatten(1)
    entity = torch.randint(24, (64,), generator=g)
    alternate, control = (entity+1)%24, (entity+2)%24
    hops = 2+torch.arange(64)%2
    other_hop = 5-hops
    keys = []
    for i in range(64):
        excluded = {int(e*4+k) for e in (entity[i], alternate[i]) for k in (hops[i], other_hop[i])}
        excluded.add(int(control[i]*4+hops[i]))
        allowed = torch.tensor([k for k in range(96) if k not in excluded])
        sampled = allowed[torch.randperm(len(allowed), generator=g)[:45]]
        row = torch.empty(48, dtype=torch.long)
        slots = [j for j in range(48) if j not in (15,16,32)]
        row[slots] = sampled
        row[15], row[16], row[32] = control[i]*4+hops[i], entity[i]*4+hops[i], entity[i]*4+hops[i]
        keys.append(row)
    keys = torch.stack(keys)
    qe, qk = keys//4, keys%4
    powers = _fpow(fmap, 3)
    qa = powers[torch.arange(64)[:, None], qk, qe]
    blocks = torch.stack((torch.full_like(qe, 24), qe, 25+qk, qa), -1).flatten(1)
    base = torch.cat((bindings, blocks), 1)
    gold = qa[:,32]
    other_gold = powers[torch.arange(64), other_hop, alternate]
    foil = torch.tensor([next(v for v in range(24) if v not in (int(gold[i]), int(other_gold[i]))) for i in range(64)])
    arms = {a: base.clone() for a in ARMS}
    for a in ('E', 'EK', 'EKP', 'RETARGET'):
        arms[a][:,113] = alternate
    for a in ('K', 'EK', 'EKP', 'RETARGET'):
        arms[a][:,114] = 25+other_hop
    for a in ('P', 'EKP', 'RETARGET'):
        arms[a][:,115] = foil
    arms['RETARGET'][:,177] = alternate
    arms['RETARGET'][:,178] = 25+other_hop
    arms['CONTROL'][:,111] = foil
    return arms, gold, foil, hops


def parser_checks(torch, R):
    checks = {}
    for pop in ('iid', 'ood_short_cycles'):
        arms, gold, foil, hops = population(torch, pop)
        for arm in ARMS:
            # Batch parser is also the actual executable interface.
            counts = []
            for i in range(0,64,4):
                masks, _, _, _ = R.source_masks(arms[arm][i:i+4])
                counts.append(masks['H'][:, QUERY_POS].sum(-1))
            expect = 1 if arm in ('base','P','RETARGET','CONTROL') else 0
            checks[f'{pop}_{arm}_history_count'] = bool((torch.cat(counts) == expect).all())
        checks[f'{pop}_foil_distinct'] = bool((gold != foil).all())
        checks[f'{pop}_hop_balance'] = bool((hops == 2).sum() == 32 and (hops == 3).sum() == 32)
    return dict(passed=all(checks.values()), checks=checks)


def factor_summary(fields, torch):
    records = []
    for h in range(4):
        branches = []
        for b in (1,2):
            q, k = fields['base'][f'q{b}'][:,h], fields['base'][f'k{b}'][:,h]
            den = (q*k).sum(-1).square().mean().sqrt().clamp_min(1e-8)
            effects, query_changes = {}, {}
            for arm in ('E','K'):
                dk = fields[arm][f'k{b}'][:,h]-k
                effects[arm] = float((q*dk).sum(-1).square().mean().sqrt()/den)
                query_changes[arm] = float((fields[arm][f'q{b}'][:,h]-q).norm()/q.norm().clamp_min(1e-8))
            branches.append(dict(branch=b, entity_effect=effects['E'], hop_effect=effects['K'],
                                 actual_query_relative_changes=query_changes))
        records.append(dict(head=h, branches=branches))
    qualified = []
    for row in records:
        for e, k in ((0,1),(1,0)):
            eb, kb = row['branches'][e], row['branches'][k]
            if eb['entity_effect']>=.5 and eb['hop_effect']<=.1 and kb['hop_effect']>=.5 and kb['entity_effect']<=.1:
                qualified.append([row['head'], e+1, k+1])
    return dict(records=records, qualified_head_entitybranch_hopbranch=qualified)


def run(torch, R, score):
    from hop_ablate import load
    started = time.perf_counter()
    signal.alarm(1800)
    os.chdir(ROOT)
    assert score.digest(score.CHECKPOINT) == score.EXPECTED_CHECKPOINT
    torch.backends.cuda.matmul.allow_tf32 = False
    model, _ = load('attn-mlp-attn-rms-seed0')
    model = model.to(device='cuda', dtype=torch.float64).eval()
    program = R.HistoryReadout(model).eval()
    results = {}
    with torch.inference_mode():
        for pop in ('iid', 'ood_short_cycles'):
            arms, gold, foil, hops = population(torch, pop)
            outputs, fields, fidelity = {}, {}, {}
            for arm in ARMS:
                ns, cs = [], []
                fs = {key: [] for key in ('q1','q2','k1','k2')}
                for i in range(0,64,4):
                    tok = arms[arm][i:i+4, :-1].cuda()
                    capture = {}
                    def hook(layer, args):
                        n = layer.norm(args[0])
                        for name in ('q1','q2','k1','k2'):
                            pos = QUERY_POS if name.startswith('q') else SOURCE_POS
                            z = getattr(layer,name)(n[:,pos:pos+1]).reshape(len(tok),1,layer.n_head,layer.d_head)
                            a,b = z.chunk(2,-1)
                            c,s = layer.rotary.cos_cached[:,pos:pos+1], layer.rotary.sin_cached[:,pos:pos+1]
                            capture[name] = (z*c+torch.cat((-b,a),-1)*s)[:,0].cpu()
                    handle = model.layers[-1].register_forward_pre_hook(hook)
                    try:
                        original = model(tok)
                    finally:
                        handle.remove()
                    masks, _, _, _ = R.source_masks(arms[arm][i:i+4])
                    actual = program(tok, {k:v.cuda() for k,v in masks.items()})
                    ns.append(original.cpu()); cs.append(actual.cpu())
                    for key in fs:
                        fs[key].append(capture[key])
                native, compiled = torch.cat(ns), torch.cat(cs)
                fidelity[arm] = dict(passed=bool(torch.allclose(native,compiled,atol=1e-9,rtol=1e-9)),
                                     max_abs=float((native-compiled).abs().max()),
                                     all_positions=score.distribution(compiled,native,torch),
                                     current_query=score.distribution(compiled[:,QUERY_POS],native[:,QUERY_POS],torch))
                outputs[arm] = native[:,QUERY_POS]
                fields[arm] = {k:torch.cat(v) for k,v in fs.items()}
            groups = {}
            for hop in (2,3):
                m = hops == hop
                rows = {}
                for arm in ARMS:
                    prob = outputs[arm].softmax(-1)
                    pg = prob.gather(-1,gold[:,None]).squeeze(-1)
                    pf = prob.gather(-1,foil[:,None]).squeeze(-1)
                    rows[arm] = dict(mean_gold_probability=float(pg[m].mean()), mean_foil_probability=float(pf[m].mean()),
                                     gold_top1=float((outputs[arm].argmax(-1)==gold)[m].double().mean()),
                                     foil_top1=float((outputs[arm].argmax(-1)==foil)[m].double().mean()),
                                     full_vector_change_rms=float((outputs[arm]-outputs['base'])[m].square().mean().sqrt()))
                groups[str(hop)] = rows
            results[pop] = dict(groups=groups, fidelity=fidelity, factors=factor_summary(fields,torch),
                                token_hashes={a:hashlib.sha256(t.numpy().tobytes()).hexdigest() for a,t in arms.items()})
            print(json.dumps({'population':pop,'groups':groups,'factors':results[pop]['factors']}),flush=True)
        checks = parser_checks(torch,R)
        pred_a = checks['passed'] and all(f['passed'] for r in results.values() for f in r['fidelity'].values())
        pred_b, pred_c = True, True
        for r in results.values():
            for g in r['groups'].values():
                bg, bf = g['base']['mean_gold_probability'], g['base']['mean_foil_probability']
                pred_a &= bg>=.8
                pred_b &= (g['P']['mean_foil_probability']>=.8 and g['CONTROL']['mean_foil_probability']-bf<=.1
                           and bg-g['E']['mean_gold_probability']>=.5 and bg-g['K']['mean_gold_probability']>=.5
                           and g['EKP']['mean_foil_probability']-bf<=.1)
                pred_c &= (g['RETARGET']['mean_foil_probability']>=.8
                           and g['RETARGET']['mean_foil_probability']-g['EKP']['mean_foil_probability']>=.5)
        nominees = results['iid']['factors']['qualified_head_entitybranch_hopbranch']
        confirms = results['ood_short_cycles']['factors']['qualified_head_entitybranch_hopbranch']
        pred_d = any(q in confirms for q in nominees)
        torch.cuda.synchronize()
        result = dict(predictions={'pred_a_instrument':bool(pred_a),'pred_b_key_payload_roles':bool(pred_b),
                                  'pred_c_joint_retarget':bool(pred_c),'pred_d_factor_specialization':bool(pred_d)},
                      terminal=('instrument_or_capability_failed' if not pred_a else
                                'composite_key_payload_roles_supported' if pred_b and pred_c else 'key_payload_roles_not_fully_supported'),
                      populations=results,parser_checks=checks,seconds=time.perf_counter()-started,
                      scope='New fixed-position token counterfactuals; semantic copy prediction distinct from exact full-probability native-background executor. QK factor specialization is a separate screen.',
                      hashes={str(p.relative_to(ROOT)):score.digest(p) for p in (REF,PREREG,SCORER,SOURCE,score.CHECKPOINT)})
        OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'terminal':result['terminal'],'predictions':result['predictions'],'seconds':result['seconds']}),flush=True)


def main():
    sys.path[:0]=[str(ROOT),str(POLY)]
    assert bound_hash()==EXPECTED
    import torch
    import contextual_history_reference as R
    import run_equality_router_v1 as score
    torch.set_num_threads(2)
    checks=parser_checks(torch,R)
    assert checks['passed'],checks
    assert R.controls()['passed']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'parser_checks':checks,'checkpoint_opened':False}))
        return
    run(torch,R,score)


if __name__=='__main__':
    main()
