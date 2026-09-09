#!/usr/bin/env python3
"""L2 writing heads and all8 final source-port restoration subsets.

pred_a_instrument: controls, source reuse/allport query closure1e-9, nativeacc>=.8.
pred_b_localized_heads: same qualifying head perorientation across populations.
pred_c_directional_ports: forward V-only>=.8/KK<=.1; backward KK>=.8/V<=.1 rescue.
pred_d_exact_joint_ports: all-port query rescue1e-9 and complete subset outcomes.
Null retains coupled representation; no port-name semantics or threshold tuning.
32worlds x6orders x4hops x17arms,B4 FP64,1800s,<256MiB/tensor; managed GPU only.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_localized_heads pred_c_directional_ports pred_d_exact_joint_ports
import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path('/workspace/tensor_language')
POLY = ROOT/'basis_aligned/polynomial_causal'
SOURCE = Path(__file__)
OUT = POLY/'SUFFIX_JOIN_HEAD_PORT_V1_RESULT.json'
ROWS = POLY/'SUFFIX_JOIN_HEAD_PORT_V1_QUERY_LOGITS.pt'
CHECKPOINT = ROOT/'runs_hop/attn4-rms-seed0/model.pt'
EXPECTED_CHECKPOINT = 'c9388cc8b1ba7e95c6a043cbf4d01592987447867a50781d3babe31ce3f8358b'
BOUND = [POLY/'suffix_join_head_port_reference.py', POLY/'suffix_join_writer_reference.py',
         POLY/'causal_suffix_join_reference.py', POLY/'SUFFIX_JOIN_HEAD_PORT_V1_PREREGISTRATION.md',
         ROOT/'basis_aligned/bilinear_quotient/ops/run_equality_router_v1.py']
EXPECTED = 'bdff72eab5911c5ed3186e46667b6f8ed580fcc54f6bc3c5434c4bdb109d9a8c'
PORT_SETS = {('ports_all' if len(s) == 3 else 'ports_' + ('+'.join(s) or 'none')): s
             for k in range(4) for s in itertools.combinations(('k1', 'k2', 'v'), k)}


def bound_hash():
    return hashlib.sha256(b''.join(p.read_bytes() for p in BOUND)).hexdigest()


def populations(torch):
    from causal_suffix_join_reference import ORDERS, SLOTS
    out = {}
    for pop, seed in (('iid', 13909), ('ood_two_12cycles', 13910)):
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
    assert not OUT.exists() and not ROWS.exists() and score.digest(CHECKPOINT) == EXPECTED_CHECKPOINT
    torch.backends.cuda.matmul.allow_tf32 = False
    model, _ = load('attn4-rms-seed0'); model = model.to(device='cuda', dtype=torch.float64).eval()
    arms = ['full']+[prefix+str(h) for h in range(4) for prefix in ('target_h', 'control_h')]+list(PORT_SETS)
    results = {}; row_data = {}
    with torch.inference_mode():
        for pop, (tokens, answers, hops, orders, worlds) in populations(torch).items():
            logits = {a: [] for a in arms}; orientations = []; exact = True; reused = True; max_error = 0.
            for i in range(0, len(tokens), 4):
                assert torch.equal(hops[i:i+4], torch.arange(4))
                mask, control, ports, orient = R.edge_masks(tokens[i:i+4]); orientations.extend(orient)
                mask, control, ports = mask.cuda(), control.cuda(), ports.cuda(); tok = tokens[i:i+4].cuda()
                base, captured = R.native_ports(model, tok)
                shared = {n: v[:1].expand_as(v) for n, v in captured.items()}
                reused &= all(torch.allclose(v[ports], shared[n][ports], atol=1e-9, rtol=1e-9) for n, v in captured.items())
                logits['full'].append(base[:, 50].cpu())
                for h in range(4):
                    for prefix, cut in (('target_h', mask), ('control_h', control)):
                        with R.modify(model, cut, (h,)):
                            value = model(tok)
                        logits[prefix+str(h)].append(value[:, 50].cpu())
                for arm, names in PORT_SETS.items():
                    with R.modify(model, mask, (0, 1, 2, 3), ports, {n: shared[n] for n in names}):
                        value = model(tok)
                    logits[arm].append(value[:, 50].cpu())
                    if arm == 'ports_all':
                        exact &= bool(torch.allclose(value[:, 50], base[:, 50], atol=1e-9, rtol=1e-9))
                        max_error = max(max_error, float((value[:, 50]-base[:, 50]).abs().max()))
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
                        'all_port_distribution': score.distribution(logits['ports_all'][select], logits['full'][select], torch)}
                passing = [h for h in range(4) if hg['3']['arms']['target_h'+str(h)]['gold_probability_loss'] >= .5
                           and abs(hg['3']['arms']['control_h'+str(h)]['gold_probability_loss']) <= .1
                           and all(abs(hg[str(k)]['arms']['target_h'+str(h)]['gold_probability_loss']) <= .1 for k in range(3))]
                loss = hg['3']['arms']['ports_none']['gold_probability_loss']
                fractions = {a: (loss-hg['3']['arms'][a]['gold_probability_loss'])/max(loss, 1e-30) for a in PORT_SETS}
                groups[orient] = {'hops': hg, 'passing_heads': passing, 'all_head_cut_loss': loss, 'port_rescue_fractions': fractions}
            results[pop] = {'groups': groups, 'all_port_query_exact': exact, 'source_ports_reused_exactly': reused,
                            'all_port_query_max_abs': max_error, 'token_sha256': hashlib.sha256(tokens.numpy().tobytes()).hexdigest()}
            row_data[pop] = {'query_logits': logits, 'answers': answers, 'hops': hops, 'orders': orders, 'worlds': worlds,
                             'orientations': orientations, 'scope': 'intervention reference outputs; not an extracted runtime cache'}
            print(json.dumps({'population': pop, 'summary': {o: {'heads': g['passing_heads'], 'port_fractions': g['port_rescue_fractions']} for o, g in groups.items()}, 'all_port_error': max_error}), flush=True)
    groups = [g for p in results.values() for g in p['groups'].values()]
    common = {o: sorted(set.intersection(*(set(p['groups'][o]['passing_heads']) for p in results.values()))) for o in ('B2_later', 'B3_later')}
    pred_d = all(p['all_port_query_exact'] for p in results.values())
    pred_a = controls['passed'] and pred_d and all(p['source_ports_reused_exactly'] for p in results.values()) and all(g['hops']['3']['arms']['full']['accuracy'] >= .8 for g in groups)
    pred_b = pred_a and all(common.values())
    pred_c = pred_a and all(g['all_head_cut_loss'] >= .5 and
        (g['port_rescue_fractions']['ports_v'] >= .8 and g['port_rescue_fractions']['ports_k1+k2'] <= .1 if o == 'B2_later' else
         g['port_rescue_fractions']['ports_k1+k2'] >= .8 and g['port_rescue_fractions']['ports_v'] <= .1)
        for p in results.values() for o, g in p['groups'].items())
    torch.save(row_data, ROWS)
    receipt = {'experiment': 'suffix_join_head_port_v1', 'runner_sha256': score.digest(SOURCE), 'bound_sha256': EXPECTED,
               'checkpoint_sha256': EXPECTED_CHECKPOINT, 'controls': controls, 'populations': results,
               'common_heads_by_orientation': common, 'independent_worlds': 32, 'correlated_query_variants': 768,
               'query_logits_artifact': str(ROWS.relative_to(ROOT)), 'query_logits_sha256': score.digest(ROWS),
               'native_parameters_retained': sum(p.numel() for p in model.parameters()),
               'predictions': {'pred_a_instrument': bool(pred_a), 'pred_b_localized_heads': bool(pred_b),
                               'pred_c_directional_ports': bool(pred_c), 'pred_d_exact_joint_ports': bool(pred_d)},
               'terminal': 'instrument_invalid' if not pred_a else ('join_heads_nominated' if pred_b else 'no_single_head_join_localization'),
               'wall_seconds': time.perf_counter()-started}
    OUT.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k: receipt[k] for k in ('terminal', 'predictions', 'common_heads_by_orientation', 'wall_seconds')}), flush=True)


def main():
    sys.path[:0] = [str(ROOT), str(POLY)]
    assert bound_hash() == EXPECTED
    import torch
    import suffix_join_head_port_reference as R
    import run_equality_router_v1 as score
    torch.set_num_threads(2); checks = R.controls(); assert checks['passed'], checks
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun': True, 'controls': checks, 'checkpoint_opened': False})); return
    run(torch, R, score, checks)


if __name__ == '__main__':
    main()
