#!/usr/bin/env python3
"""Fixed shared hop-state q1/k1 compiler: full output and joint interventions.

pred_a_instrument: CPU controls, native fold1e-9, no unseen semantic states.
pred_b_prediction: all29-way KL mean<=1e-3/p99<=1e-2, every panel/arm/subset.
pred_c_manipulation: centered removal error<=.01 relative or1e-8 silent absolute.
pred_d_structure: physically fewer constants, charge books/native background.
Null closes the fixed code, no head/rank/feature sweep. Batch4 FP64, <256MiB tensor,
64 fit plus48 held docs,1800s guard. GPU exclusively through managed bqrunner.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_prediction pred_c_manipulation pred_d_structure
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
BOUND = [POLY/'hop_state_reference.py', POLY/'contextual_history_reference.py',
         POLY/'HOP_STATE_COMPILER_V1_PREREGISTRATION.md',
         ROOT/'basis_aligned/bilinear_quotient/ops/run_equality_router_v1.py']
EXPECTED = '2bdde0753b66c9135b62ebf01ea9cb3313e6411387461687173706c1e4ddc049'
OUT = POLY/'HOP_STATE_COMPILER_V1_RESULT.json'
BOOKS = POLY/'HOP_STATE_COMPILER_V1_BOOKS.pt'
ARMS = {'native': (), 'remove_h0': (0,), 'remove_h3': (3,), 'remove_joint': (0, 3)}


def digest_bound():
    return hashlib.sha256(b''.join(p.read_bytes() for p in BOUND)).hexdigest()


def populations(torch, data):
    panels = {'iid': data.sample_docs(16, torch.Generator().manual_seed(6910))[0]}
    g = torch.Generator().manual_seed(6911)
    fmap = torch.rand(16, 24, generator=g).argsort(1)
    order = torch.rand(16, 24, generator=g).argsort(1)
    bindings = torch.stack((order, fmap.gather(1, order)), -1).flatten(1)
    qe = torch.randint(24, (16, 48), generator=g)
    qk = torch.randint(4, (16, 48), generator=g)
    qa = data._fpow(fmap, 3)[torch.arange(16)[:, None], qk, qe]
    blocks = torch.stack((torch.full_like(qe, 24), qe, 25+qk, qa), -1).flatten(1)
    panels['ood_short_cycles'] = torch.cat((bindings, blocks), 1)
    panels['ood_short_history'] = data.sample_docs(16, torch.Generator().manual_seed(6912))[0][:, :112]
    return panels


def run(torch, R, score, controls):
    from hop_ablate import load
    import hop_data
    signal.alarm(1800)
    started = time.perf_counter()
    os.chdir(ROOT)
    assert score.digest(score.CHECKPOINT) == score.EXPECTED_CHECKPOINT
    torch.backends.cuda.matmul.allow_tf32 = False
    model, cfg = load('attn-mlp-attn-rms-seed0')
    model = model.to(device='cuda', dtype=torch.float64).eval()
    baseline = R.HistoryReadout(model).eval()
    fit = hop_data.sample_docs(64, torch.Generator().manual_seed(6909))[0]
    sums = {n: torch.zeros(30, 2, 32, device='cuda', dtype=torch.float64) for n in ('q1', 'k1')}
    count = torch.zeros(30, device='cuda', dtype=torch.float64)
    with torch.inference_mode():
        for i in range(0, len(fit), 4):
            tok = fit[i:i+4, :-1].cuda()
            state = R.codes(tok).flatten()
            x = model.embed(tok)
            for layer in model.layers[:-1]:
                x = layer(x)
            final = model.layers[-1]
            h = final.norm(x)
            count.index_add_(0, state, torch.ones_like(state, dtype=torch.float64))
            for name in sums:
                values = getattr(final, name)(h).reshape(-1, 4, 32)[:, [0, 3]]
                sums[name].index_add_(0, state, values)
        seen = count > 0
        books = {n: sums[n]/count.clamp_min(1)[:, None, None] for n in sums}
        # Seal before constructing any held population, never fit to logits.
        assert not BOOKS.exists() and not OUT.exists()
        torch.save({'books': {n: b.cpu() for n, b in books.items()}, 'counts': count.cpu(),
                    'bound_sha256': EXPECTED, 'fit_seed': 6909}, BOOKS)
        frozen_sha = score.digest(BOOKS)
        program = R.HopStateReadout(model, books, seen).eval()
        results = {}
        for pop, tokens in populations(torch, hop_data).items():
            native = {a: [] for a in ARMS}; compiled = {a: [] for a in ARMS}
            fold_max = 0.; fold_close = True; states_seen = True
            for i in range(0, len(tokens), 4):
                tok = tokens[i:i+4, :-1].cuda()
                states_seen &= bool(seen[R.codes(tok)].all())
                if not states_seen:
                    break
                ref = model(tok)
                folded = R.full(baseline, tok)
                fold_max = max(fold_max, float((ref-folded).abs().max()))
                fold_close &= bool(torch.allclose(ref, folded, atol=1e-9, rtol=1e-9))
                for arm, heads in ARMS.items():
                    with R.remove_heads(model, heads), R.remove_heads(program, heads):
                        native[arm].append(model(tok).cpu())
                        compiled[arm].append(R.full(program, tok).cpu())
            if not states_seen:
                results[pop] = {'states_seen': False, 'fold_close': fold_close}
                continue
            native = {a: torch.cat(v) for a, v in native.items()}
            compiled = {a: torch.cat(v) for a, v in compiled.items()}
            qp = torch.arange(50, tokens.shape[1]-1, 4)
            fidelity = {a: {'all': score.distribution(compiled[a], native[a], torch),
                            'query': score.distribution(compiled[a][:, qp], native[a][:, qp], torch)}
                        for a in ARMS}
            effects = {a: score.effect_error(compiled[a]-compiled['native'], native[a]-native['native'], torch)
                       for a in ARMS if a != 'native'}
            joint = native['remove_h0']+native['remove_h3']-native['native']
            results[pop] = {'states_seen': True, 'fold_close': fold_close, 'fold_max_abs': fold_max,
                            'distribution': fidelity, 'effects': effects,
                            'native_joint_identity_max_abs': float((joint-native['remove_joint']).abs().max()),
                            'token_sha256': hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
                            'native_query_accuracy': float((native['native'][:, qp].argmax(-1) == tokens[:, qp+1]).double().mean())}
            print(json.dumps({'population': pop, 'query': fidelity['native']['query'], 'joint': effects['remove_joint']}), flush=True)
    charged = sum(p.numel() for p in program.parameters())+program.folded.numel()+sum(b.numel() for b in books.values())
    pred_a = controls['passed'] and all(r['states_seen'] and r['fold_close'] for r in results.values())
    pred_b = pred_a and all(s['passed'] for r in results.values() for a in r['distribution'].values() for s in a.values())
    pred_c = pred_a and all(e['passed'] for r in results.values() for e in r['effects'].values()) and all(
        r['effects'][a]['target_rms'] > 1e-6 for r in results.values() for a in ('remove_h0', 'remove_h3'))
    pred_d = charged < 387968 and all(getattr(program.background.layers[-1], n).weight.numel() == 8192 for n in ('q1', 'k1'))
    receipt = {'experiment': 'hop_state_compiler_v1', 'runner_sha256': score.digest(SOURCE),
               'bound_sha256': EXPECTED, 'checkpoint_sha256': score.EXPECTED_CHECKPOINT,
               'books_sha256': frozen_sha, 'controls': controls, 'fit_counts': count.cpu().tolist(),
               'populations': results, 'arbitrary_constants': charged, 'prior_folded_constants': 387968,
               'remaining_fixed_buffer_scalars': sum(b.numel() for n, b in program.named_buffers() if 'book' not in n and n != 'folded'),
               'predictions': {'pred_a_instrument': bool(pred_a), 'pred_b_prediction': bool(pred_b),
                               'pred_c_manipulation': bool(pred_c), 'pred_d_structure': bool(pred_d)},
               'terminal': 'instrument_invalid' if not pred_a else ('shared_hop_state_supported' if pred_b and pred_c and pred_d else 'fixed_hop_state_rejected'),
               'wall_seconds': time.perf_counter()-started}
    OUT.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k: receipt[k] for k in ('terminal', 'predictions', 'arbitrary_constants', 'wall_seconds')}), flush=True)


def main():
    sys.path[:0] = [str(ROOT), str(POLY)]
    assert digest_bound() == EXPECTED
    import torch
    import hop_state_reference as R
    import run_equality_router_v1 as score
    torch.set_num_threads(2)
    checks = R.controls()
    assert checks['passed'], checks
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun': True, 'controls': checks, 'checkpoint_opened': False}))
        return
    run(torch, R, score, checks)


if __name__ == '__main__':
    main()
