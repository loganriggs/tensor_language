#!/usr/bin/env python3
"""pred_a exact saved-state replay; pred_b bothconverged; pred_c both5%gain.
BQGATE:0forwards0seq. Full-U structured continuation, 1800seconds/start.
"""
import json
import os
from pathlib import Path
import signal
import sys
import time

from run_structured_bilinear_native_v2 import P, CK, digest, diagnostics
import torch
from structured_bilinear_bank_v1 import StructuredBank

PREFIX = 'STRUCTURED_BILINEAR_CONTINUE_V1'


def main():
    binding = json.loads((P / f'{PREFIX}_BINDING.json').read_text())
    assert all(digest(path) == sha for path, sha in binding.items())
    previous = json.loads((P / 'STRUCTURED_BILINEAR_NATIVE_V2_RESULT.json').read_text())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False, corpus_access=False,
            body_forwards=0, coefficients=276480, products=4608,
            seeds=[0, 937], additional_seconds_per_seed=1800)))
        return
    out = P / f'{PREFIX}_RESULT.json'
    assert not out.exists()
    signal.alarm(4500)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.perf_counter()
    sd = torch.load(CK, map_location='cpu', weights_only=True, mmap=True)
    u = sd['lm_head.weight'].double().cuda()
    whitener = torch.linalg.cholesky(u.T @ u).T
    del u
    l, r, d = [sd[f'transformer.h.17.mlp.{key}.weight'].double().cuda()
               for key in ('Left', 'Right', 'Down')]
    native = l, r, whitener @ d
    total = json.loads((P / 'FULLU_OUTPUT_FUNCTIONS_V1_AUDIT.json').read_text())['native_total']
    reports = []
    for prior in previous['starts']:
        seed = prior['seed']
        assert digest(prior['cache']['path']) == prior['cache']['sha256']
        saved = torch.load(prior['cache']['path'], map_location='cpu', weights_only=True)
        model = StructuredBank([2] * 7 + [3, 3], branches=4).cuda()
        model.load_state_dict(saved['model'])
        optimizer = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=1,
            max_eval=10, history_size=20, tolerance_grad=0., tolerance_change=0.,
            line_search_fn='strong_wolfe')
        optimizer.load_state_dict(saved['optimizer'])
        old_state = next(iter(saved['optimizer']['state'].values()))
        state = optimizer.state[next(iter(model.parameters()))]
        counters = {k: int(state[k]) for k in ('n_iter', 'func_evals')}
        assert all(counters[k] == old_state[k] for k in counters)
        assert len(state['old_dirs']) == len(old_state['old_dirs'])
        evaluations = 0

        def closure():
            nonlocal evaluations
            model.zero_grad(set_to_none=True)
            loss, detail = model.coefficient_loss(native, whitener, total)
            assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters())
            closure.detail = detail
            evaluations += 1
            return loss

        loss = closure()
        initial = diagnostics(model, loss, closure.detail)
        errors = {key: abs(initial[key] - prior['final'][key]) / max(abs(prior['final'][key]), 1e-30)
                  for key in ('loss', 'relative_stationarity')}
        assert max(errors.values()) <= 1e-8, errors
        replay = dict(errors=errors, optimizer_counters=counters, history_length=len(state['old_dirs']))
        print(json.dumps(dict(seed=seed, replay=replay, initial=initial)), flush=True)
        history = [dict(step=0, seconds=0., **initial)]
        fit_start = time.perf_counter()
        last_save = fit_start
        converged = False
        stop = 'step_limit'
        stalled = 0
        last_loss = initial['loss']
        cache = Path(f'/dev/shm/bilin18_structured_continue_v1_s{seed}.pt')
        assert not cache.exists()

        def save_state(status):
            temporary = cache.with_suffix('.tmp')
            torch.save(dict(model={k: v.cpu() for k, v in model.state_dict().items()},
                optimizer=optimizer.state_dict(), seed=seed, order=saved['order'],
                history=history, binding=binding, prior_cache=prior['cache']), temporary)
            temporary.replace(cache)
            report = dict(seed=seed, replay=replay, initial=initial, final=history[-1],
                history=history, converged=converged, stop=status,
                additional_seconds=time.perf_counter() - fit_start, evaluations=evaluations,
                capture_gain=history[-1]['capture'] - initial['capture'],
                cache=dict(path=str(cache), sha256=digest(cache), bytes=cache.stat().st_size,
                           ephemeral=True), prior_cache=prior['cache'])
            receipt = P / f'{PREFIX}_SEED_{seed}.json'
            temp_receipt = receipt.with_suffix('.tmp')
            temp_receipt.write_text(json.dumps(report, indent=2) + '\n')
            temp_receipt.replace(receipt)
            return report

        for step in range(1, 3001):
            optimizer.step(closure)
            elapsed = time.perf_counter() - fit_start
            checkpoint_due = time.perf_counter() - last_save >= 300
            if step % 5 == 0 or elapsed >= 1800 or checkpoint_due or step == 3000:
                # Recompute the accepted point, not a stale line-search trial.
                loss = closure()
                current = float(loss)
                stalled = stalled + 1 if abs(current - last_loss) <= 1e-15 else 0
                last_loss = current
                row = dict(step=step, seconds=elapsed, **diagnostics(model, loss, closure.detail))
                history.append(row)
                progress = abs(row['loss'] - history[-5]['loss']) / max(row['capture'], 1e-12) if len(history) >= 5 else None
                row['five_check_relative_change'] = progress
                converged = row['relative_stationarity'] <= 1e-4 and row['gradient_max'] <= 1e-7 and progress is not None and progress <= 1e-5
                print(json.dumps(dict(seed=seed, **row)), flush=True)
                if converged:
                    stop = 'converged'
                    break
                if elapsed >= 1800 or stalled >= 10:
                    stop = 'time_limit' if elapsed >= 1800 else 'objective_stall'
                    break
            if checkpoint_due:
                save_state('running')
                last_save = time.perf_counter()
        report = save_state(stop)
        reports.append(report)
        del model, optimizer, state, old_state, saved
    result = dict(predictions={'pred_a_replay': True,
        'pred_b_both_converged': all(r['converged'] for r in reports),
        'pred_c_both_capture_gain': all(r['final']['capture'] >= 1.05 * .08634383041327387 for r in reports)},
        starts=reports, wall_seconds=time.perf_counter() - started, binding=binding,
        corpus_access=False, body_forwards=0,
        scope='Same structured weight objective continued; convergence is local, not circuit identification.')
    with out.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result['predictions']), flush=True)


if __name__ == '__main__':
    main()
