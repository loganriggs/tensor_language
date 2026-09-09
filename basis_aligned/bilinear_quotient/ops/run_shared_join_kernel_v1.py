#!/usr/bin/env python3
"""Replace whole L2H1/H2 routing with the frozen shared semantic equality kernel.

pred_a_instrument: planted executor, exact folded baseline, seen cells, finite.
pred_b_distribution: every arm/population KL mean<=1e-3,p99<=1e-2, all/query.
pred_c_interventions: centered removal relative RMS<=.01 (absolute1e-8 below1e-6).
pred_d_structure: physically remove32768 QK weights, add6576, total361776.
Null closes this fixed local-record kernel; no feature/head/lag/iteration sweep.
256 calibration,64IID/64OOD/32fixed-point held;8 ALS passes;B4 FP64;1800s;
each analysis tensor<256MiB. Coefficients frozen before held data generation.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_distribution pred_c_interventions pred_d_structure
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
OUT = POLY/'SHARED_JOIN_KERNEL_V1_RESULT.json'
COEFFICIENTS = POLY/'SHARED_JOIN_KERNEL_V1_COEFFICIENTS.pt'
ROWS = POLY/'SHARED_JOIN_KERNEL_V1_ROWS.pt'
CHECKPOINT = ROOT/'runs_hop/attn4-rms-seed0/model.pt'
EXPECTED_CHECKPOINT = 'c9388cc8b1ba7e95c6a043cbf4d01592987447867a50781d3babe31ce3f8358b'
BOUND = [POLY/'shared_join_kernel_reference.py', POLY/'contextual_history_reference.py',
         POLY/'SHARED_JOIN_KERNEL_V1_PREREGISTRATION.md',
         ROOT/'basis_aligned/bilinear_quotient/ops/run_equality_router_v1.py']
EXPECTED = '5d5abe656d40c2b6eadac4a37f5de8fc5f696f950fdd45fce9a86d2913604c92'
ARMS = {'full': (), 'remove_h1': (1,), 'remove_h2': (2,), 'remove_both': (1, 2)}


def bound_hash():
    return hashlib.sha256(b''.join(p.read_bytes() for p in BOUND)).hexdigest()


def population(torch, kind, count, seed):
    g = torch.Generator().manual_seed(seed)
    rows, answers, hops = [], [], []
    for _ in range(count):
        perm = torch.randperm(24, generator=g); fmap = torch.empty(24, dtype=torch.long)
        if kind == 'ood_two_12cycles':
            fmap[perm[:12]] = perm[:12].roll(-1); fmap[perm[12:]] = perm[12:].roll(-1)
        else:
            fmap[perm] = perm.roll(-1)
        entity = int(torch.randint(24, (), generator=g))
        hop = int(torch.randint(4, (), generator=g)) if kind != 'fixed_point' else 3
        if kind == 'fixed_point':
            predecessor = int(torch.where(fmap == entity)[0][0]); old = int(fmap[entity])
            fmap[entity] = entity; fmap[predecessor] = old
        order = torch.randperm(24, generator=g)
        answer = entity
        for _ in range(hop):
            answer = int(fmap[answer])
        rows.append(torch.cat((torch.stack((order, fmap[order]), -1).flatten(), torch.tensor([24, entity, 25+hop]))))
        answers.append(answer); hops.append(hop)
    return torch.stack(rows), torch.tensor(answers), torch.tensor(hops)


def calibrate(torch, R, model):
    tokens, _, _ = population(torch, 'iid', 256, 15909)
    samples = [[] for _ in range(4)]
    for i in range(0, len(tokens), 4):
        tok = tokens[i:i+4].cuda(); x = model.embed(tok)
        for layer in model.layers[:2]:
            x = layer(x)
        layer = model.layers[2]; pattern = layer.pattern(x)
        gain = R.rms_gain(x, layer.norm)
        feature = gain.square()[:, :, None]*gain.square()[:, None, :]
        for h in range(2):
            group, middle, match = R.indices(tok, h)
            for output, data in zip(samples, (group[match], middle[match], feature[match], pattern[:, h+1][match])):
                output.append(data)
    group, middle, feature, target = [torch.cat(x) for x in samples]
    assert all(x.numel()*x.element_size() < 256*1024**2 for x in (group, middle, feature, target))
    theta, gamma, seen = R.fit(group, middle, feature, target)
    estimate = feature*theta[group]*gamma[middle]
    diagnostic = {'matching_samples': len(group), 'seen_role_lag_cells': int(seen.sum()),
                  'calibration_token_sha256': hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
                  'matched_score_relative_rms': float((estimate-target).square().sum().sqrt()/target.square().sum().sqrt())}
    torch.save({'theta': theta.cpu(), 'gamma': gamma.cpu(), 'seen': seen.cpu(),
                'calibration': diagnostic, 'checkpoint_sha256': EXPECTED_CHECKPOINT,
                'bound_sha256': EXPECTED, 'passes': 8}, COEFFICIENTS)
    return theta, gamma, seen, diagnostic


def run(torch, R, score, checks):
    from hop_ablate import load
    from contextual_history_reference import HistoryReadout
    os.chdir(ROOT); signal.alarm(1800); started = time.perf_counter()
    assert not any(p.exists() for p in (OUT, COEFFICIENTS, ROWS))
    assert score.digest(CHECKPOINT) == EXPECTED_CHECKPOINT
    torch.backends.cuda.matmul.allow_tf32 = False
    model, _ = load('attn4-rms-seed0'); model = model.to(device='cuda', dtype=torch.float64).eval()
    results, row_data = {}, {}; exact_error = 0.; finite = True; all_seen = True
    with torch.inference_mode():
        theta, gamma, seen, calibration = calibrate(torch, R, model)
        frozen_sha = score.digest(COEFFICIENTS)
        program = R.JoinKernelProgram(model, theta, gamma, seen)
        baseline = HistoryReadout(model)
        # First construction/access of held populations happens after the coefficient file is frozen.
        for pop, count, seed in (('iid', 64, 15910), ('ood_two_12cycles', 64, 15911), ('fixed_point', 32, 15912)):
            tokens, answers, hops = population(torch, pop, count, seed)
            missing = set()
            for h in range(2):
                group, _, match = R.indices(tokens, h)
                missing.update(group[match][~seen.cpu()[group[match]]].unique().tolist())
            all_seen &= not missing
            panel = {'n': count, 'seed': seed, 'unseen_matched_cells': sorted(missing),
                     'token_sha256': hashlib.sha256(tokens.numpy().tobytes()).hexdigest()}
            results[pop] = panel
            if missing:
                row_data[pop] = {'tokens': tokens, 'answers': answers, 'hops': hops}
                print(json.dumps({'population': pop, 'instrument': 'unseen_matched_cells', 'count': len(missing)}), flush=True)
                continue
            native = {a: [] for a in ARMS}; candidate = {a: [] for a in ARMS}
            for i in range(0, count, 4):
                tok = tokens[i:i+4].cuda()
                for arm, heads in ARMS.items():
                    with R.cut_heads(model, heads), R.cut_heads(program, heads), R.cut_heads(baseline, heads):
                        teacher = model(tok); actual = R.full(program, tok); exact = R.full(baseline, tok)
                    exact_error = max(exact_error, float((exact-teacher).abs().max()))
                    finite &= bool(torch.isfinite(teacher).all() and torch.isfinite(actual).all())
                    native[arm].append(teacher.cpu()); candidate[arm].append(actual.cpu())
            native = {a: torch.cat(v) for a, v in native.items()}
            candidate = {a: torch.cat(v) for a, v in candidate.items()}
            panel['distribution'] = {a: {'all': score.distribution(candidate[a], native[a], torch),
                                              'query': score.distribution(candidate[a][:, -1], native[a][:, -1], torch)} for a in ARMS}
            panel['effects'] = {a: score.effect_error(candidate[a]-candidate['full'], native[a]-native['full'], torch)
                                for a in ARMS if a != 'full'}
            panel['native_accuracy'] = float((native['full'][:, -1].argmax(-1) == answers).double().mean())
            panel['candidate_accuracy'] = float((candidate['full'][:, -1].argmax(-1) == answers).double().mean())
            row_data[pop] = {'tokens': tokens, 'answers': answers, 'hops': hops, 'native': native, 'candidate': candidate}
            print(json.dumps({'population': pop, 'full': panel['distribution']['full'], 'effects': panel['effects']}), flush=True)
    assert score.digest(COEFFICIENTS) == frozen_sha
    raw_count = sum(p.numel() for p in model.parameters())
    baseline_count = sum(p.numel() for p in baseline.parameters())+baseline.folded.numel()
    count = sum(p.numel() for p in program.parameters())+program.folded.numel()+theta.numel()+gamma.numel()
    removed = all(getattr(program.background.layers[2].native, n).weight.numel()*2 == getattr(model.layers[2], n).weight.numel()
                  for n in ('q1', 'k1', 'q2', 'k2'))
    pred_a = checks['passed'] and all_seen and finite and exact_error <= 1e-9
    pred_b = pred_a and all(v['passed'] for p in results.values() for a in p['distribution'].values() for v in a.values())
    pred_c = pred_a and all(e['passed'] for p in results.values() for e in p['effects'].values())
    pred_d = removed and count == 361776 and baseline_count == 387968 and raw_count == 400640
    torch.save(row_data, ROWS)
    receipt = {'experiment': 'shared_join_kernel_v1', 'runner_sha256': score.digest(SOURCE), 'bound_sha256': EXPECTED,
               'checkpoint_sha256': EXPECTED_CHECKPOINT, 'controls': checks, 'calibration': calibration,
               'coefficients_sha256': frozen_sha, 'coefficients': str(COEFFICIENTS.relative_to(ROOT)),
               'rows': str(ROWS.relative_to(ROOT)), 'rows_sha256': score.digest(ROWS), 'populations': results,
               'exact_baseline_max_abs_logit': exact_error, 'finite': finite,
               'price': {'original': raw_count, 'generic_folded': baseline_count, 'candidate': count,
                         'qk_rows_physically_removed': removed, 'semantic_coefficients': theta.numel()+gamma.numel()},
               'predictions': {'pred_a_instrument': bool(pred_a), 'pred_b_distribution': bool(pred_b),
                               'pred_c_interventions': bool(pred_c), 'pred_d_structure': bool(pred_d)},
               'terminal': 'instrument_invalid' if not pred_a else ('shared_kernel_supported' if pred_b and pred_c and pred_d else 'fixed_local_record_kernel_rejected'),
               'wall_seconds': time.perf_counter()-started}
    OUT.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k: receipt[k] for k in ('terminal', 'predictions', 'price', 'wall_seconds')}), flush=True)


def main():
    sys.path[:0] = [str(ROOT), str(POLY)]
    assert bound_hash() == EXPECTED
    import torch
    import shared_join_kernel_reference as R
    import run_equality_router_v1 as score
    torch.set_num_threads(2); checks = R.controls()
    checks['checks'].update({'executor_'+k: v for k, v in R.executor_controls()['checks'].items()})
    checks['passed'] = all(checks['checks'].values()); assert checks['passed'], checks
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun': True, 'controls': checks, 'checkpoint_opened': False})); return
    run(torch, R, score, checks)


if __name__ == '__main__':
    main()
