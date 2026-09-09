#!/usr/bin/env python3
"""Contextual answer-history versus binding retrieval, full readout decomposition.

pred_a_exact_extraction: full original/folded logits at1e-9 on all5arms,3populations.
pred_b_repetition_explains_floor: higher-hop repeated acc>=.8, novel<=.15, gap>=.5;
unique-query higher-hop acc<=.15.
pred_c_selective_sources: repeated higher-hop matched gold-prob loss H>=.25,C<=.10;
novel hop1 binding loss>=.25 in IID and short-cycle OOD.
pred_d_composition: native and compiled single/joint centered vectors, and native
z_H+z_B-z0==z_HB, at1e-9. Null: history account fails; preserve exact compiler only.
Price:384docs x5arms, batch4,T239, all29 logits; 387968 program constants,
1800s, each analysis tensor<256MiB. GPU exclusively via bqrunner.
"""
# BQGATE: EXPERIMENT pred_a_exact_extraction pred_b_repetition_explains_floor pred_c_selective_sources pred_d_composition
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
PREREG = POLY/'CONTEXTUAL_HISTORY_V1_PREREGISTRATION.md'
SCORER = ROOT/'basis_aligned/bilinear_quotient/ops/run_equality_router_v1.py'
OUT = POLY/'CONTEXTUAL_HISTORY_V1_RESULT.json'
EXPECTED = '1e56a31f9eaabb039e617c98ed6cae40cea0e55800436028fddd46dfcc887ad2'
ARMS = {'native': (), 'remove_history': ('H',), 'remove_binding': ('B',),
        'remove_joint': ('H', 'B'), 'remove_control': ('C',)}


def bound_hash():
    return hashlib.sha256(REF.read_bytes()+PREREG.read_bytes()+SCORER.read_bytes()).hexdigest()


def populations(torch, data):
    iid = data.sample_docs(128, torch.Generator().manual_seed(4909))
    out = {'iid': iid}
    for kind, seed in (('ood_unique_queries', 4910), ('ood_short_cycles', 4911)):
        g = torch.Generator().manual_seed(seed)
        if kind == 'ood_unique_queries':
            cyc = torch.rand(128, 24, generator=g).argsort(1)
            fmap = torch.empty_like(cyc).scatter_(1, cyc, cyc.roll(-1, 1))
            keys = torch.rand(128, 96, generator=g).argsort(1)[:, :48]
            qe, qk = keys//4, keys%4
        else:
            fmap = torch.rand(128, 24, generator=g).argsort(1)
            qe = torch.randint(24, (128, 48), generator=g)
            qk = torch.randint(4, (128, 48), generator=g)
        order = torch.rand(128, 24, generator=g).argsort(1)
        bindings = torch.stack((order, fmap.gather(1, order)), -1).flatten(1)
        powers = data._fpow(fmap, 3)
        qa = powers[torch.arange(128)[:, None], qk, qe]
        blocks = torch.stack((torch.full_like(qe, 24), qe, 25+qk, qa), -1).flatten(1)
        out[kind] = (torch.cat((bindings, blocks), 1), qa, qk)
    return out


def group_summary(native, answers, select, positions, torch):
    count = int(select.sum())
    if count == 0:
        return {'n': 0}
    base = native['native'][:, positions]
    correct = base.argmax(-1) == answers
    pbase = base.softmax(-1).gather(-1, answers[..., None]).squeeze(-1)
    effects = {}
    for arm in ARMS:
        edited = native[arm][:, positions]
        pgold = edited.softmax(-1).gather(-1, answers[..., None]).squeeze(-1)
        p, q = base.log_softmax(-1), edited.log_softmax(-1)
        kl = (p.exp()*(p-q)).sum(-1)
        effects[arm] = dict(gold_probability_loss=float((pbase-pgold)[select].mean()),
                            accuracy=float((edited.argmax(-1) == answers)[select].double().mean()),
                            mean_kl=float(kl[select].mean()))
    return dict(n=count, accuracy=float(correct[select].double().mean()),
                mean_gold_probability=float(pbase[select].mean()), removals=effects)


def run(torch, R, score):
    from hop_ablate import load
    import hop_data
    started = time.perf_counter()
    signal.alarm(1800)
    os.chdir(ROOT)
    assert score.digest(score.CHECKPOINT) == score.EXPECTED_CHECKPOINT
    torch.backends.cuda.matmul.allow_tf32 = False
    model, cfg = load('attn-mlp-attn-rms-seed0')
    assert cfg['spec'] == ['attn', 'mlp', 'attn']
    model = model.to(device='cuda', dtype=torch.float64).eval()
    program = R.HistoryReadout(model).eval()
    results = {}
    with torch.inference_mode():
        for population, (tokens, answers, hops) in populations(torch, hop_data).items():
            # Parse in batches; never allocate a 128x239x239 GPU/CPU analysis tensor.
            native, compiled = {a: [] for a in ARMS}, {a: [] for a in ARMS}
            repeated, eligible, ha, ca = [], [], [], []
            for i in range(0, len(tokens), 4):
                masks, rep, elig, ages = R.source_masks(tokens[i:i+4])
                repeated.append(rep); eligible.append(elig)
                ha += ages['history_age_tokens']; ca += ages['control_age_tokens']
                masks = {k: v.cuda() for k, v in masks.items()}
                tok = tokens[i:i+4, :-1].cuda()
                parts = program.parts(tok, masks)
                whole = parts['R']+parts['H']+parts['B']+parts['O']
                for arm, names in ARMS.items():
                    kill = None if not names else torch.stack([masks[n] for n in names]).any(0)
                    actual = whole
                    for name in names:
                        actual = actual-parts[name]
                    original = R.native_forward(model, tok, kill)
                    native[arm].append(original.cpu()); compiled[arm].append(actual.cpu())
            native = {a: torch.cat(v) for a, v in native.items()}
            compiled = {a: torch.cat(v) for a, v in compiled.items()}
            rep, elig = torch.cat(repeated), torch.cat(eligible)
            if population == 'ood_unique_queries':
                assert not rep.any()
            metrics = {a: {s: score.distribution(compiled[a][:, idx], native[a][:, idx], torch)
                           for s, idx in (('all_positions', slice(None)), ('query_positions', hop_data.ANS_POS))}
                       for a in ARMS}
            errors = {a: score.effect_error(compiled[a]-compiled['native'], native[a]-native['native'], torch)
                      for a in ARMS if a != 'native'}
            def center(x):
                return x-x.mean(-1, keepdim=True)
            effect_closure = all(torch.allclose(center(compiled[a]-compiled['native']),
                                                center(native[a]-native['native']), atol=1e-9, rtol=1e-9)
                                 for a in ARMS)
            joint_prediction = native['remove_history']+native['remove_binding']-native['native']
            joint_closure = bool(torch.allclose(joint_prediction, native['remove_joint'], atol=1e-9, rtol=1e-9))
            groups = {}
            for hop in range(4):
                for kind, mask in (('repeated', rep), ('novel', ~rep)):
                    groups[f'hop{hop}_{kind}'] = group_summary(native, answers, (hops == hop)&mask, hop_data.ANS_POS, torch)
            for name, select in (('high_repeated', (hops >= 2)&rep), ('high_novel', (hops >= 2)&~rep),
                                 ('high_matched_repeated', (hops >= 2)&rep&elig), ('hop1_novel', (hops == 1)&~rep)):
                groups[name] = group_summary(native, answers, select, hop_data.ANS_POS, torch)
            results[population] = dict(groups=groups, distribution_fidelity=metrics, effects=errors,
                                       all_logits_close=all(torch.allclose(compiled[a], native[a], atol=1e-9, rtol=1e-9) for a in ARMS),
                                       max_abs_logit=max(float((compiled[a]-native[a]).abs().max()) for a in ARMS),
                                       effect_vectors_close=bool(effect_closure), native_joint_closure=joint_closure,
                                       joint_max_abs=float((joint_prediction-native['remove_joint']).abs().max()),
                                       matching_control_age_mean={'H': sum(ha)/max(1,len(ha)), 'C': sum(ca)/max(1,len(ca))},
                                       repeated_queries=int(rep.sum()), eligible_repeated_queries=int(elig.sum()),
                                       token_sha256=hashlib.sha256(tokens.numpy().tobytes()).hexdigest())
            print(json.dumps({'population': population, 'full_logit_closure': results[population]['all_logits_close'],
                              'high_repeated': groups['high_repeated'], 'high_novel': groups['high_novel'],
                              'high_matched': groups['high_matched_repeated']}), flush=True)
        controls = R.controls()
        pred_a = controls['passed'] and all(r['all_logits_close'] for r in results.values())
        pred_b, pred_c = True, True
        for population in ('iid', 'ood_short_cycles'):
            g = results[population]['groups']
            pred_b &= (g['high_repeated']['accuracy'] >= .8 and g['high_novel']['accuracy'] <= .15
                       and g['high_repeated']['accuracy']-g['high_novel']['accuracy'] >= .5)
            h, c = (g['high_matched_repeated']['removals'][a]['gold_probability_loss']
                    for a in ('remove_history', 'remove_control'))
            b = g['hop1_novel']['removals']['remove_binding']['gold_probability_loss']
            pred_c &= h >= .25 and c <= .1 and b >= .25
        pred_b &= results['ood_unique_queries']['groups']['high_novel']['accuracy'] <= .15
        pred_d = all(r['effect_vectors_close'] and r['native_joint_closure'] for r in results.values())
        torch.cuda.synchronize()
        result = dict(predictions={'pred_a_exact_extraction': bool(pred_a), 'pred_b_repetition_explains_floor': bool(pred_b),
                                  'pred_c_selective_sources': bool(pred_c), 'pred_d_composition': bool(pred_d)},
                      terminal=('invalid' if not(pred_a and pred_d) else 'answer_history_account_supported' if
                                pred_b and pred_c else 'answer_history_account_not_fully_supported'),
                      populations=results, controls=controls,
                      constants={'native_parameters': sum(p.numel() for p in model.parameters()),
                                 'compiled_parameters_and_fold': sum(p.numel() for p in program.parameters())+program.folded.numel(),
                                 'folded_readout': program.folded.numel(),
                                 'other_fixed_buffers': sum(b.numel() for b in program.buffers())-program.folded.numel()},
                      seconds=time.perf_counter()-started,
                      scope='Exact final-readout decomposition with explicitly native prefix and contextual Q/K/V; generic compiler folding saving, not a discovered compact router or complete bilin18 explanation.',
                      hashes={str(p.relative_to(ROOT)): score.digest(p) for p in (REF, PREREG, SCORER, SOURCE, score.CHECKPOINT)})
        OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps({'terminal': result['terminal'], 'predictions': result['predictions'],
                          'seconds': result['seconds']}), flush=True)


def main():
    sys.path[:0] = [str(ROOT), str(POLY)]
    assert bound_hash() == EXPECTED
    import torch
    import contextual_history_reference as R
    import run_equality_router_v1 as score
    torch.set_num_threads(2)
    checks = R.controls()
    assert checks['passed'], checks
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun': True, 'controls': checks, 'checkpoint_opened': False}))
        return
    run(torch, R, score)


if __name__ == '__main__':
    main()
