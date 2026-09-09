#!/usr/bin/env python3
"""Interchange two explicit join writes across unrelated binding contexts.

pred_a_instrument: controls/reuse/self replay1e-9, finite, native hop3 acc>=.8.
pred_b_selectivity: own hop3 probability loss>=.25; other query/lower hops abs<=.1.
pred_c_context_transfer: full distributions meanKL<=1e-3,p99<=1e-2 all/query.
pred_d_composition: single/joint centered effect error<=.01, absolute1e-8 below1e-6.
Null rejects context-independent selected writes, not their native causal use.
Native400640 parameters+donor prefix retained; no fit/independent extraction claim.
32worlds x2arrangements x8queries, B4 FP64,1800s,<256MiB/tensor; managed GPU.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_selectivity pred_c_context_transfer pred_d_composition
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path('/workspace/tensor_language'); POLY = ROOT/'basis_aligned/polynomial_causal'
SOURCE = Path(__file__)
OUT = POLY/'JOIN_CONTRIBUTION_CONTEXT_V1_RESULT.json'
ROWS = POLY/'JOIN_CONTRIBUTION_CONTEXT_V1_ROWS.pt'
CHECKPOINT = ROOT/'runs_hop/attn4-rms-seed0/model.pt'
EXPECTED_CHECKPOINT = 'c9388cc8b1ba7e95c6a043cbf4d01592987447867a50781d3babe31ce3f8358b'
BOUND = [POLY/'join_contribution_context_reference.py', POLY/'JOIN_CONTRIBUTION_CONTEXT_V1_PREREGISTRATION.md',
         ROOT/'basis_aligned/bilinear_quotient/ops/run_equality_router_v1.py']
EXPECTED = 'e5d8035f032b68741d2811e7fe5d367b34fcbb036bb072322b50b947ea0a1167'
ARMS = {'native': ((), None), 'remove_a': ((0,), None), 'remove_b': ((1,), None),
        'remove_both': ((0, 1), None), 'own_restore_both': ((0, 1), 'own'),
        'donor_restore_a': ((0,), 'donor'), 'donor_restore_b': ((1,), 'donor'),
        'donor_restore_both': ((0, 1), 'donor')}


def bound_hash():
    return hashlib.sha256(b''.join(p.read_bytes() for p in BOUND)).hexdigest()


def run(torch, R, score, checks):
    from hop_ablate import load
    os.chdir(ROOT); signal.alarm(1800); started = time.perf_counter()
    assert not OUT.exists() and not ROWS.exists() and score.digest(CHECKPOINT) == EXPECTED_CHECKPOINT
    torch.backends.cuda.matmul.allow_tf32 = False
    model, _ = load('attn4-rms-seed0'); model = model.to(device='cuda', dtype=torch.float64).eval()
    results = {}; row_data = {}; replay = 0.; reuse = 0.; finite = True
    with torch.inference_mode():
        for pop, worlds in R.populations().items():
            logits = {a: [] for a in ARMS}; metadata = []; token_rows = []; donor_rows = []; writes_saved = []
            for w in worlds:
                masks = {j: m.cuda() for j, m in w['masks'].items()}; heads = w['heads']
                donor = w['donor'].cuda(); tokens = w['recipient'].cuda()
                donor_writes = R.contributions(model, donor, masks, heads)
                own_writes = R.contributions(model, tokens[:1], masks, heads)
                for i in range(0, len(tokens), 4):
                    tok = tokens[i:i+4]
                    donor_forks = tok.clone(); donor_forks[:, :48] = donor[:, :48]
                    for inputs, reference in ((donor_forks, donor_writes), (tok, own_writes)):
                        observed = R.contributions(model, inputs, masks, heads)
                        reuse = max(reuse, *(float((v-reference[j]).abs().max()) for j, v in observed.items()))
                    batch = {}
                    for arm, (selected, source) in ARMS.items():
                        writes = donor_writes if source == 'donor' else (own_writes if source == 'own' else None)
                        with R.intervene(model, masks, heads, selected, writes): value = model(tok)
                        finite &= bool(torch.isfinite(value).all()); batch[arm] = value
                        logits[arm].append(value.cpu())
                    replay = max(replay, float((batch['own_restore_both']-batch['native']).abs().max()))
                token_rows.append(w['recipient']); donor_rows.append(w['donor'])
                writes_saved.append({'own': {j: v.cpu() for j, v in own_writes.items()},
                                     'donor': {j: v.cpu() for j, v in donor_writes.items()}})
                metadata.extend({'world': w['world'], 'arrangement': w['arrangement'],
                                 'query': int(q), 'hop': int(h), 'answer': int(a)}
                                for q, h, a in zip(w['query'], w['hops'], w['answers']))
            logits = {a: torch.cat(v) for a, v in logits.items()}
            answers = torch.tensor([m['answer'] for m in metadata]); native = logits['native']
            gold = {a: v[:, -1].softmax(-1).gather(-1, answers[:, None]).squeeze(-1) for a, v in logits.items()}
            arrangements = {}
            for arrangement in range(2):
                select = torch.tensor([m['arrangement'] == arrangement for m in metadata])
                behavior = {}
                for query in range(2):
                    for hop in range(4):
                        group = torch.tensor([m['arrangement'] == arrangement and m['query'] == query and m['hop'] == hop for m in metadata])
                        behavior[str(query)+'_'+str(hop)] = {
                            'n': int(group.sum()), 'query': query, 'hop': hop,
                            'native_accuracy': float((native[:, -1].argmax(-1) == answers)[group].double().mean()),
                            'remove_a_loss': float((gold['native']-gold['remove_a'])[group].mean()),
                            'remove_b_loss': float((gold['native']-gold['remove_b'])[group].mean())}
                transfer = {}; effects = {}
                for suffix in ('a', 'b', 'both'):
                    restored = logits['donor_restore_'+suffix][select]; base = native[select]
                    cut = logits['remove_'+suffix][select]
                    transfer[suffix] = {'all': score.distribution(restored, base, torch),
                                        'query': score.distribution(restored[:, -1], base[:, -1], torch)}
                    effects[suffix] = {'all': score.effect_error(restored-cut, base-cut, torch),
                                       'query': score.effect_error((restored-cut)[:, -1], (base-cut)[:, -1], torch)}
                arrangements[str(arrangement)] = {'behavior': behavior, 'transfer': transfer, 'effects': effects}
            results[pop] = {'arrangements': arrangements}
            row_data[pop] = {'tokens': torch.cat(token_rows), 'donor_tokens': torch.cat(donor_rows),
                             'metadata': metadata, 'logits': logits, 'writes': writes_saved}
            print(json.dumps({'population': pop, 'arrangements': arrangements}), flush=True)
    groups = [a for p in results.values() for a in p['arrangements'].values()]
    pred_a = checks['passed'] and finite and replay <= 1e-9 and reuse <= 1e-9 and all(
        v['native_accuracy'] >= .8 for g in groups for v in g['behavior'].values() if v['hop'] == 3)
    selective = True
    for g in groups:
        for v in g['behavior'].values():
            for query, name in ((0, 'remove_a_loss'), (1, 'remove_b_loss')):
                selective &= v[name] >= .25 if v['hop'] == 3 and v['query'] == query else abs(v[name]) <= .1
    pred_b = pred_a and selective
    pred_c = pred_a and all(v['passed'] for g in groups for a in g['transfer'].values() for v in a.values())
    pred_d = pred_a and all(v['passed'] for g in groups for a in g['effects'].values() for v in a.values())
    torch.save(row_data, ROWS)
    receipt = {'experiment': 'join_contribution_context_v1', 'runner_sha256': score.digest(SOURCE), 'bound_sha256': EXPECTED,
               'checkpoint_sha256': EXPECTED_CHECKPOINT, 'controls': checks, 'self_replay_max_abs_logit': replay,
               'query_reuse_max_abs_write': reuse, 'finite': finite, 'populations': results,
               'native_parameters_retained': sum(p.numel() for p in model.parameters()),
               'donor_prefix_required': True, 'independent_worlds': 32, 'recipient_query_variants': 512,
               'rows': str(ROWS.relative_to(ROOT)), 'rows_sha256': score.digest(ROWS),
               'predictions': {'pred_a_instrument': bool(pred_a), 'pred_b_selectivity': bool(pred_b),
                               'pred_c_context_transfer': bool(pred_c), 'pred_d_composition': bool(pred_d)},
               'terminal': 'instrument_invalid' if not pred_a else ('context_equivalence_supported' if pred_b and pred_c and pred_d else 'context_independent_join_write_not_established'),
               'wall_seconds': time.perf_counter()-started}
    OUT.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k: receipt[k] for k in ('terminal', 'predictions', 'wall_seconds')}), flush=True)


def main():
    sys.path[:0] = [str(ROOT), str(POLY)]
    assert bound_hash() == EXPECTED
    import torch
    import join_contribution_context_reference as R
    import run_equality_router_v1 as score
    torch.set_num_threads(2); checks = R.controls(); assert checks['passed'], checks
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun': True, 'controls': checks, 'checkpoint_opened': False})); return
    run(torch, R, score, checks)


if __name__ == '__main__': main()
