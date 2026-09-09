#!/usr/bin/env python3
"""Direct upstream pair-to-pair writers of the nominated causal suffix join.

pred_a_instrument: explicit replay/rescue identity1e-9 and nativehop3 acc>=.8.
pred_b_selective_writer: same layer/joint targetloss>=.5, controls<=.1 eachgroup.
pred_c_source_port_rescue: recover>=.8 jointloss and fullqueryKL1e-3/p991e-2.
pred_d_reused_port_identity: hop0 sourceports reused acrossallhops, native1e-9.
Null: close direct writer path; don't assume source nomination identifies its producer.
32worldsx6ordersx4hops,11arms,B4 FP64,1800s,<256MiB/tensor; managed GPU only.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_selective_writer pred_c_source_port_rescue pred_d_reused_port_identity
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
OUT = POLY/'SUFFIX_JOIN_WRITER_V1_RESULT.json'
CHECKPOINT = ROOT/'runs_hop/attn4-rms-seed0/model.pt'
EXPECTED_CHECKPOINT = 'c9388cc8b1ba7e95c6a043cbf4d01592987447867a50781d3babe31ce3f8358b'
BOUND = [POLY/'suffix_join_writer_reference.py', POLY/'causal_suffix_join_reference.py',
         POLY/'SUFFIX_JOIN_WRITER_V1_PREREGISTRATION.md',
         ROOT/'basis_aligned/bilinear_quotient/ops/run_equality_router_v1.py']
EXPECTED = '42218b196ba136682ede7a9ee12a51c89148fb0b4ebd6a6185553f6a65ba6f66'
LAYER_SETS = {'L0': (0,), 'L1': (1,), 'L2': (2,), 'joint': (0, 1, 2)}


def bound_hash():
    return hashlib.sha256(b''.join(p.read_bytes() for p in BOUND)).hexdigest()


def populations(torch):
    from causal_suffix_join_reference import ORDERS, SLOTS
    out = {}
    for pop, seed in (('iid', 12909), ('ood_two_12cycles', 12910)):
        g = torch.Generator().manual_seed(seed)
        rows, answers, hops, orders, worlds = [], [], [], [], []
        for world in range(16):
            perm = torch.randperm(24, generator=g); fmap = torch.empty(24, dtype=torch.long)
            if pop == 'iid':
                fmap[perm] = perm.roll(-1)
            else:
                fmap[perm[:12]] = perm[:12].roll(-1); fmap[perm[12:]] = perm[12:].roll(-1)
            e = int(torch.randint(24, (), generator=g)); chain = [e, int(fmap[e]), int(fmap[fmap[e]])]
            rest = [int(x) for x in torch.randperm(24, generator=g) if int(x) not in chain]
            for oi, order in enumerate(ORDERS):
                assigned = {s: chain[j] for s, j in zip(SLOTS, order)}; it = iter(rest)
                keys = torch.tensor([assigned[s] if s in assigned else next(it) for s in range(24)])
                bindings = torch.stack((keys, fmap[keys]), -1).flatten(); answer = e
                for hop in range(4):
                    rows.append(torch.cat((bindings, torch.tensor([24, e, 25+hop]))))
                    answers.append(answer); hops.append(hop); orders.append(oi); worlds.append(world)
                    answer = int(fmap[answer])
        out[pop] = tuple(torch.stack(rows) if i == 0 else torch.tensor(v) for i, v in enumerate((rows, answers, hops, orders, worlds)))
    return out


def run(torch, R, score, controls):
    from hop_ablate import load
    started = time.perf_counter(); signal.alarm(1800); os.chdir(ROOT)
    assert not OUT.exists() and score.digest(CHECKPOINT) == EXPECTED_CHECKPOINT
    torch.backends.cuda.matmul.allow_tf32 = False
    model, _ = load('attn4-rms-seed0'); model = model.to(device='cuda', dtype=torch.float64).eval()
    arms = ['full']+[prefix+n for n in LAYER_SETS for prefix in ('target_', 'control_')]+['rescue', 'identity']
    results = {}
    with torch.inference_mode():
        for pop, (tokens, answers, hops, orders, worlds) in populations(torch).items():
            logits = {a: [] for a in arms}; orientations = []; identity = True; reuse = True; max_identity = 0.
            for i in range(0, len(tokens), 4):
                assert torch.equal(hops[i:i+4], torch.arange(4))
                mask, control, ports, orient = R.edge_masks(tokens[i:i+4]); orientations.extend(orient)
                mask, control, ports = mask.cuda(), control.cuda(), ports.cuda(); tok = tokens[i:i+4].cuda()
                base, captured = R.native_ports(model, tok)
                shared = {n: v[:1].expand_as(v) for n, v in captured.items()}
                reuse &= all(torch.allclose(v[ports], shared[n][ports], atol=1e-9, rtol=1e-9) for n, v in captured.items())
                logits['full'].append(base[:, 50].cpu())
                for name, layers in LAYER_SETS.items():
                    for prefix, cut in (('target_', mask), ('control_', control)):
                        with R.modifications(model, cut, layers):
                            value = model(tok)
                        logits[prefix+name].append(value[:, 50].cpu())
                for name, layers in (('rescue', LAYER_SETS['joint']), ('identity', ())):
                    with R.modifications(model, mask, layers, ports, shared):
                        value = model(tok)
                    logits[name].append(value[:, 50].cpu())
                    if name == 'identity':
                        identity &= bool(torch.allclose(value, base, atol=1e-9, rtol=1e-9))
                        max_identity = max(max_identity, float((value-base).abs().max()))
            logits = {a: torch.cat(v) for a, v in logits.items()}
            probs = {a: v.softmax(-1).gather(-1, answers[:, None]).squeeze(-1) for a, v in logits.items()}
            groups = {}
            for orient in ('B2_later', 'B3_later'):
                om = torch.tensor([o == orient for o in orientations]); hg = {}
                for h in range(4):
                    select = om & (hops == h)
                    hg[str(h)] = {'n_query_variants': int(select.sum()), 'arms': {
                        a: {'accuracy': float((v.argmax(-1) == answers)[select].double().mean()),
                            'gold_probability': float(probs[a][select].mean()),
                            'gold_probability_loss': float((probs['full']-probs[a])[select].mean())} for a, v in logits.items()},
                        'rescue_distribution': score.distribution(logits['rescue'][select], logits['full'][select], torch)}
                passing = [n for n in LAYER_SETS if hg['3']['arms']['target_'+n]['gold_probability_loss'] >= .5
                           and abs(hg['3']['arms']['control_'+n]['gold_probability_loss']) <= .1
                           and all(abs(hg[str(h)]['arms']['target_'+n]['gold_probability_loss']) <= .1 for h in range(3))]
                loss = hg['3']['arms']['target_joint']['gold_probability_loss']
                recovered = loss-hg['3']['arms']['rescue']['gold_probability_loss']
                groups[orient] = {'hops': hg, 'passing_writers': passing, 'joint_loss': loss, 'rescue_fraction': recovered/max(loss, 1e-30)}
            results[pop] = {'groups': groups, 'identity': identity, 'ports_reused_exactly': reuse, 'identity_max_abs': max_identity,
                            'token_sha256': hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
                            'rows': [{'world': int(worlds[i]), 'order': int(orders[i]), 'hop': int(hops[i]), 'orientation': orientations[i],
                                      'answer': int(answers[i]), 'query_logits': {a: logits[a][i].tolist() for a in arms}} for i in range(len(tokens))]}
            print(json.dumps({'population': pop, 'groups': groups, 'identity_max_abs': max_identity}), flush=True)
    groups = [g for p in results.values() for g in p['groups'].values()]
    common = sorted(set.intersection(*(set(g['passing_writers']) for g in groups)))
    pred_a = controls['passed'] and all(g['hops']['3']['arms']['full']['accuracy'] >= .8 for g in groups)
    pred_b = pred_a and bool(common)
    pred_c = pred_a and all(g['joint_loss'] >= .5 and g['rescue_fraction'] >= .8 and g['hops']['3']['rescue_distribution']['passed'] for g in groups)
    pred_d = all(p['identity'] and p['ports_reused_exactly'] for p in results.values())
    receipt = {'experiment': 'suffix_join_writer_v1', 'runner_sha256': score.digest(SOURCE), 'bound_sha256': EXPECTED,
               'checkpoint_sha256': EXPECTED_CHECKPOINT, 'controls': controls, 'populations': results,
               'common_selective_writers': common, 'independent_worlds': 32, 'correlated_query_variants': 768,
               'native_parameters_retained': sum(p.numel() for p in model.parameters()),
               'predictions': {'pred_a_instrument': bool(pred_a), 'pred_b_selective_writer': bool(pred_b),
                               'pred_c_source_port_rescue': bool(pred_c), 'pred_d_reused_port_identity': bool(pred_d)},
               'terminal': 'instrument_invalid' if not pred_a else ('direct_join_writer_nominated' if pred_b else 'direct_join_writer_not_selective'),
               'wall_seconds': time.perf_counter()-started}
    OUT.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k: receipt[k] for k in ('terminal', 'predictions', 'common_selective_writers', 'wall_seconds')}), flush=True)


def main():
    sys.path[:0] = [str(ROOT), str(POLY)]
    assert bound_hash() == EXPECTED
    import torch
    import suffix_join_writer_reference as R
    import run_equality_router_v1 as score
    torch.set_num_threads(2); checks = R.controls(); assert checks['passed'], checks
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun': True, 'controls': checks, 'checkpoint_opened': False})); return
    run(torch, R, score, checks)


if __name__ == '__main__':
    main()
