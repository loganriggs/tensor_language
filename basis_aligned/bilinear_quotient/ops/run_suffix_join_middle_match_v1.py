#!/usr/bin/env python3
"""Valid permutation interventions test causal middle-entity matching, frozen heads.

pred_a_instrument: structural controls, native acc>=.8 every population/orientation/case.
pred_b_joint_score_gate: mismatched RMS<=.1 smaller matched RMS, matched>1e-6.
pred_c_causal_effect_gate: same gate on centered full29-way removal effects.
pred_d_matched_use: both matched cases selected-route gold-prob loss>=.25.
Null rejects fixed middle-equality account; no factor renaming or threshold sweep.
32worlds x6orders x4cases,B4 FP64,1800s,<256MiB/tensor; managed GPU only.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_joint_score_gate pred_c_causal_effect_gate pred_d_matched_use
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import types

ROOT = Path('/workspace/tensor_language')
POLY = ROOT/'basis_aligned/polynomial_causal'
SOURCE = Path(__file__)
OUT = POLY/'SUFFIX_JOIN_MIDDLE_MATCH_V1_RESULT.json'
ROWS = POLY/'SUFFIX_JOIN_MIDDLE_MATCH_V1_ROWS.pt'
CHECKPOINT = ROOT/'runs_hop/attn4-rms-seed0/model.pt'
EXPECTED_CHECKPOINT = 'c9388cc8b1ba7e95c6a043cbf4d01592987447867a50781d3babe31ce3f8358b'
BOUND = [POLY/'suffix_join_middle_match_reference.py', POLY/'suffix_join_head_port_reference.py',
         POLY/'suffix_join_writer_reference.py', POLY/'causal_suffix_join_reference.py',
         POLY/'SUFFIX_JOIN_MIDDLE_MATCH_V1_PREREGISTRATION.md',
         ROOT/'basis_aligned/bilinear_quotient/ops/run_equality_router_v1.py']
EXPECTED = '1e0e883c5b227732272db6bd263dc2d7a45200b4ca43dd572809932976ba66c2'


def bound_hash():
    return hashlib.sha256(b''.join(p.read_bytes() for p in BOUND)).hexdigest()


def native_scores(model, tokens, mask, head):
    layer = model.layers[2]; original = layer.pattern; captured = {}
    def pattern(_self, x):
        p = original(x)
        captured['cells'] = p[:, head][mask].reshape(len(tokens), 4).detach().clone()
        return p
    layer.pattern = types.MethodType(pattern, layer)
    try:
        logits = model(tokens)
    finally:
        del layer.pattern
    return logits, captured['cells']


def run(torch, R, H, score, checks):
    from hop_ablate import load
    os.chdir(ROOT); signal.alarm(1800); started = time.perf_counter()
    assert not OUT.exists() and not ROWS.exists() and score.digest(CHECKPOINT) == EXPECTED_CHECKPOINT
    torch.backends.cuda.matmul.allow_tf32 = False
    model, _ = load('attn4-rms-seed0'); model = model.to(device='cuda', dtype=torch.float64).eval()
    results = {}; row_data = {}
    with torch.inference_mode():
        for pop, (tokens, masks, metadata) in R.populations().items():
            native, cut, cells = [], [], []
            for i in range(0, len(tokens), 4):
                assert [m['case'] for m in metadata[i:i+4]] == list(R.CASES)
                orient = metadata[i]['orientation']; head = 1 if orient == 'B2_later' else 2
                assert all(m['orientation'] == orient for m in metadata[i:i+4])
                tok, mask = tokens[i:i+4].cuda(), masks[i:i+4].cuda()
                base, values = native_scores(model, tok, mask, head)
                with H.modify(model, mask, (head,)):
                    changed = model(tok)
                native.append(base[:, 50].cpu()); cut.append(changed[:, 50].cpu()); cells.append(values.cpu())
            native, cut, cells = torch.cat(native), torch.cat(cut), torch.cat(cells)
            answers = torch.tensor([m['answer'] for m in metadata])
            prob = native.softmax(-1).gather(-1, answers[:, None]).squeeze(-1)
            cutprob = cut.softmax(-1).gather(-1, answers[:, None]).squeeze(-1)
            effect = cut-native; effect = effect-effect.mean(-1, keepdim=True)
            groups = {}
            for orient in ('B2_later', 'B3_later'):
                cases = {}
                for case in R.CASES:
                    select = torch.tensor([m['orientation'] == orient and m['case'] == case for m in metadata])
                    cases[case] = {'n_order_variants': int(select.sum()),
                                   'accuracy': float((native.argmax(-1) == answers)[select].double().mean()),
                                   'native_gold_probability': float(prob[select].mean()),
                                   'route_gold_probability_loss': float((prob-cutprob)[select].mean()),
                                   'joint_score_rms': float(cells[select].square().mean().sqrt()),
                                   'centered_removal_effect_rms': float(effect[select].square().mean().sqrt())}
                score_den = min(cases[c]['joint_score_rms'] for c in ('base', 'both'))
                effect_den = min(cases[c]['centered_removal_effect_rms'] for c in ('base', 'both'))
                ratios = {c: {'score': cases[c]['joint_score_rms']/max(score_den, 1e-30),
                              'effect': cases[c]['centered_removal_effect_rms']/max(effect_den, 1e-30)} for c in ('keys', 'values')}
                groups[orient] = {'cases': cases, 'mismatch_ratios': ratios, 'matched_score_min_rms': score_den,
                                  'matched_effect_min_rms': effect_den,
                                  'score_gate': score_den > 1e-6 and all(v['score'] <= .1 for v in ratios.values()),
                                  'effect_gate': effect_den > 1e-6 and all(v['effect'] <= .1 for v in ratios.values())}
            results[pop] = {'groups': groups, 'token_sha256': hashlib.sha256(tokens.numpy().tobytes()).hexdigest()}
            row_data[pop] = {'tokens': tokens, 'metadata': metadata, 'native_query_logits': native,
                             'cut_query_logits': cut, 'joint_score_cells': cells}
            print(json.dumps({'population': pop, 'groups': groups}), flush=True)
    groups = [g for p in results.values() for g in p['groups'].values()]
    pred_a = checks['passed'] and all(c['accuracy'] >= .8 for g in groups for c in g['cases'].values())
    pred_b = pred_a and all(g['score_gate'] for g in groups)
    pred_c = pred_a and all(g['effect_gate'] for g in groups)
    pred_d = pred_a and all(g['cases'][c]['route_gold_probability_loss'] >= .25 for g in groups for c in ('base', 'both'))
    torch.save(row_data, ROWS)
    receipt = {'experiment': 'suffix_join_middle_match_v1', 'runner_sha256': score.digest(SOURCE),
               'bound_sha256': EXPECTED, 'checkpoint_sha256': EXPECTED_CHECKPOINT, 'controls': checks,
               'populations': results, 'row_artifact': str(ROWS.relative_to(ROOT)), 'row_artifact_sha256': score.digest(ROWS),
               'independent_worlds': 32, 'correlated_order_case_variants': 768, 'native_parameters_retained': sum(p.numel() for p in model.parameters()),
               'predictions': {'pred_a_instrument': bool(pred_a), 'pred_b_joint_score_gate': bool(pred_b),
                               'pred_c_causal_effect_gate': bool(pred_c), 'pred_d_matched_use': bool(pred_d)},
               'terminal': 'instrument_invalid' if not pred_a else ('middle_entity_matching_supported' if pred_b and pred_c and pred_d else 'fixed_middle_equality_not_established'),
               'wall_seconds': time.perf_counter()-started}
    OUT.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k: receipt[k] for k in ('terminal', 'predictions', 'wall_seconds')}), flush=True)


def main():
    sys.path[:0] = [str(ROOT), str(POLY)]
    assert bound_hash() == EXPECTED
    import torch
    import suffix_join_middle_match_reference as R
    import suffix_join_head_port_reference as H
    import run_equality_router_v1 as score
    torch.set_num_threads(2); checks = R.controls()
    checks['checks'].update({'head_'+k: v for k, v in H.controls()['checks'].items()})
    checks['passed'] = all(checks['checks'].values()); assert checks['passed'], checks
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun': True, 'controls': checks, 'checkpoint_opened': False})); return
    run(torch, R, H, score, checks)


if __name__ == '__main__':
    main()
