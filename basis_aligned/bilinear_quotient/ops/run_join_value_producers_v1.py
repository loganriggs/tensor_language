#!/usr/bin/env python3
"""Exact E/Y0/Y1 payload producer factorial on fresh two-join contexts.

pred_a_instrument: closure1e-9, native accuracy>=.8, selected cut loss>=.25.
pred_b_fields: E-forward/prefix-backward recovery>=.8, complement<=.2.
pred_c_distribution: fixed joint full-output KL mean<=1e-3,p99<=1e-2 all/query.
pred_d_effects: single/joint centered RMS<=.01; absolute1e-8 below target1e-6.
Null rejects fixed semantic producer sufficiency, not the exact linear split.
All400640 native parameters retained, no fit or best-subset search.32worlds,
2arrangements,8queries,22arms,B4 FP64,1800s,<256MiB per tensor;managedGPU.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_fields pred_c_distribution pred_d_effects
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path('/workspace/tensor_language'); POLY = ROOT/'basis_aligned/polynomial_causal'
SOURCE = Path(__file__); OUT = POLY/'JOIN_VALUE_PRODUCERS_V1_RESULT.json'
ROWS = POLY/'JOIN_VALUE_PRODUCERS_V1_ROWS.pt'
CHECKPOINT = ROOT/'runs_hop/attn4-rms-seed0/model.pt'
EXPECTED_CHECKPOINT = 'c9388cc8b1ba7e95c6a043cbf4d01592987447867a50781d3babe31ce3f8358b'
BOUND = [POLY/'join_value_producer_reference.py', POLY/'join_contribution_context_reference.py',
         POLY/'JOIN_VALUE_PRODUCERS_V1_PREREGISTRATION.md',
         ROOT/'basis_aligned/bilinear_quotient/ops/run_equality_router_v1.py']
EXPECTED = '3fa1ba4c39eb039999e97a9e3c2fe66c9f87f658d59d9989e567514703103e2e'


def bound_hash():
    return hashlib.sha256(b''.join(p.read_bytes() for p in BOUND)).hexdigest()


def run(torch, R, score, checks):
    from hop_ablate import load
    os.chdir(ROOT); signal.alarm(1800); started = time.perf_counter()
    assert not OUT.exists() and not ROWS.exists() and score.digest(CHECKPOINT) == EXPECTED_CHECKPOINT
    torch.backends.cuda.matmul.allow_tf32 = False
    model, _ = load('attn4-rms-seed0'); model = model.to(device='cuda', dtype=torch.float64).eval()
    results = {}; row_data = {}; replay = 0.; producer_error = 0.; write_error = 0.; finite = True
    with torch.inference_mode():
        for pop, worlds in R.populations(seeds=(18909, 18910)).items():
            all_logits = {}; metadata = []; tokens_saved = []
            for w in worlds:
                masks = {j: m.cuda() for j, m in w['masks'].items()}; heads = w['heads']
                tokens = w['recipient'].cuda(); tokens_saved.append(w['recipient'])
                for i in range(0, 8, 4):
                    tok = tokens[i:i+4]
                    parts, error = R.producer_writes(model, tok, masks, heads)
                    producer_error = max(producer_error, error)
                    native_writes = R.contributions(model, tok, masks, heads)
                    summed = R.combine(parts)
                    write_error = max(write_error, *(float((v-native_writes[j]).abs().max()) for j, v in summed.items()))
                    arms = {'native': ((), None), 'remove_a': ((0,), None), 'remove_b': ((1,), None),
                            'remove_both': ((0, 1), None), 'all_restore_both': ((0, 1), summed)}
                    for j, name in ((0, 'a'), (1, 'b')):
                        for subset in range(8):
                            kinds = tuple(k for n, k in enumerate(R.KINDS) if subset&(1 << n))
                            arms['restore_'+name+'_'+str(subset)] = ((j,), R.combine(parts, kinds))
                    semantic = {j: parts[j]['E'] if heads[j] == 1 else parts[j]['Y0']+parts[j]['Y1'] for j in heads}
                    arms['semantic_joint'] = ((0, 1), semantic)
                    batch = {}
                    for arm, (selected, writes) in arms.items():
                        with R.intervene(model, masks, heads, selected, writes): value = model(tok)
                        finite &= bool(torch.isfinite(value).all()); batch[arm] = value
                        all_logits.setdefault(arm, []).append(value.cpu())
                    replay = max(replay, *(float((batch[a]-batch['native']).abs().max())
                                           for a in ('restore_a_7','restore_b_7','all_restore_both')))
                metadata.extend({'world': w['world'], 'arrangement': w['arrangement'], 'query': int(q),
                                 'hop': int(h), 'answer': int(a)} for q, h, a in zip(w['query'],w['hops'],w['answers']))
            logits = {a: torch.cat(v) for a, v in all_logits.items()}; native = logits['native']
            answers = torch.tensor([m['answer'] for m in metadata])
            gold = {a: v[:, -1].softmax(-1).gather(-1, answers[:, None]).squeeze(-1) for a, v in logits.items()}
            arrangements = {}
            for arrangement in range(2):
                select = torch.tensor([m['arrangement'] == arrangement for m in metadata]); fields = {}
                for j, name in ((0,'a'),(1,'b')):
                    group = torch.tensor([m['arrangement'] == arrangement and m['query'] == j and m['hop'] == 3 for m in metadata])
                    loss = float((gold['native']-gold['remove_'+name])[group].mean())
                    recovery = {str(subset): float((gold['restore_'+name+'_'+str(subset)]-gold['remove_'+name])[group].mean())/max(loss,1e-30)
                                for subset in range(8)}
                    forward = j == arrangement; chosen = 1 if forward else 6
                    fields[name] = {'direction': 'forward' if forward else 'backward', 'native_accuracy': float((native[:, -1].argmax(-1)==answers)[group].double().mean()),
                                    'cut_loss': loss, 'recovery': recovery, 'candidate_subset': chosen,
                                    'candidate_recovery': recovery[str(chosen)], 'complement_recovery': recovery[str(7-chosen)]}
                effects = {}
                for name in ('a','b','both'):
                    arm = 'semantic_joint' if name == 'both' else 'restore_'+name+'_'+str(fields[name]['candidate_subset'])
                    actual = logits[arm][select]; base = native[select]; cut = logits['remove_'+name][select]
                    effects[name] = {'all': score.effect_error(actual-cut, base-cut, torch),
                                     'query': score.effect_error((actual-cut)[:, -1], (base-cut)[:, -1], torch)}
                actual = logits['semantic_joint'][select]; base = native[select]
                arrangements[str(arrangement)] = {'fields': fields, 'effects': effects,
                                                   'joint_distribution': {'all': score.distribution(actual, base, torch),
                                                                          'query': score.distribution(actual[:, -1], base[:, -1], torch)}}
            results[pop] = {'arrangements': arrangements}
            # All positions scored above; compact per-example query vectors and token errors
            # preserve counterexamples without serializing22 complete storage views.
            lp = native.log_softmax(-1)
            token_errors = {a: {'kl': (lp.exp()*(lp-v.log_softmax(-1))).sum(-1),
                                'centered_logit_error_rms': ((v-native)-(v-native).mean(-1,keepdim=True)).square().mean(-1).sqrt()}
                            for a,v in logits.items()}
            row_data[pop] = {'tokens': torch.cat(tokens_saved), 'metadata': metadata,
                             'query_logits': {a: v[:, -1].clone() for a,v in logits.items()}, 'token_errors': token_errors}
            print(json.dumps({'population': pop, 'arrangements': arrangements}), flush=True)
    groups = [a for p in results.values() for a in p['arrangements'].values()]
    pred_a = checks['passed'] and finite and max(replay,producer_error,write_error) <= 1e-9 and all(
        f['native_accuracy'] >= .8 and f['cut_loss'] >= .25 for g in groups for f in g['fields'].values())
    pred_b = pred_a and all(f['candidate_recovery'] >= .8 and f['complement_recovery'] <= .2 for g in groups for f in g['fields'].values())
    pred_c = pred_a and all(v['passed'] for g in groups for v in g['joint_distribution'].values())
    pred_d = pred_a and all(v['passed'] for g in groups for a in g['effects'].values() for v in a.values())
    torch.save(row_data, ROWS)
    receipt = {'experiment': 'join_value_producers_v1', 'runner_sha256': score.digest(SOURCE), 'bound_sha256': EXPECTED,
               'checkpoint_sha256': EXPECTED_CHECKPOINT, 'controls': checks, 'native_parameters_retained': sum(p.numel() for p in model.parameters()),
               'closure': {'residual_producers': producer_error,'selected_write_sum': write_error,'all_term_logit_replay':replay},
               'finite': finite,'populations': results,'rows':str(ROWS.relative_to(ROOT)),'rows_sha256':score.digest(ROWS),
               'independent_worlds':32,'query_variants':512,
               'predictions':{'pred_a_instrument':bool(pred_a),'pred_b_fields':bool(pred_b),'pred_c_distribution':bool(pred_c),'pred_d_effects':bool(pred_d)},
               'terminal':'instrument_invalid' if not pred_a else ('fixed_payload_producers_supported' if pred_b and pred_c and pred_d else 'fixed_payload_producer_sufficiency_rejected'),
               'wall_seconds':time.perf_counter()-started}
    OUT.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('terminal','predictions','closure','wall_seconds')}),flush=True)


def main():
    sys.path[:0] = [str(ROOT),str(POLY)]
    assert bound_hash() == EXPECTED
    import torch
    import join_value_producer_reference as R
    import run_equality_router_v1 as score
    torch.set_num_threads(2); checks=R.controls(); assert checks['passed'],checks
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'controls':checks,'checkpoint_opened':False}));return
    run(torch,R,score,checks)


if __name__=='__main__':main()
