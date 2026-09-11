#!/usr/bin/env python3
"""A exact replay<=1e-8; B fullU128capture>=.08698912596; C fullU1024>=.50."""
# BQGATE: 0forwards0seq; weight-only native product dictionary OLS, two metrics.
import os, sys, json, hashlib, signal, time
from pathlib import Path
RUNNER = Path(__file__).resolve()
P = RUNNER.parents[3] / 'basis_aligned/polynomial_causal'
sys.path.insert(0, str(P))
import torch
from shared_dictionary_ols_v1 import select
from joint_quadratic_fit_v1 import product_cross
CK = Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
BUDGETS = [32, 64, 128, 256, 512, 1024]


def conditional(gram, cross, support, total):
    g = gram[support][:, support]
    c = cross[:, support]
    chol = torch.linalg.cholesky((g + g.T) / 2)
    writer = torch.cholesky_solve(c.T, chol).T
    captured = float((writer * c).sum() / total)
    residual = float((writer @ g - c).norm() / c.norm())
    ev = torch.linalg.eigvalsh(g)
    return dict(capture=captured, solve_relative_residual=residual,
                gram_condition=float(ev[-1] / ev[0]))


def main():
    binding = json.loads((P / 'NATIVE_DICTIONARY_OLS_V1_BINDING.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p,h in binding.items())
    assert json.loads((P/'SHARED_DICTIONARY_OLS_V1_CONTROL.json').read_text())['instrument_passed']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(model_loaded=False, gpu_accessed=False, body_forwards=0,
            corpus_access=False, metrics=['full', 'centered'], budgets=BUDGETS,
            scalar_parameters_at_1024=3*1152*1024, large_checkpoint=False)))
        return
    out = P/'NATIVE_DICTIONARY_OLS_V1_RESULT.json'
    assert not out.exists()
    signal.alarm(900)
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    sd = torch.load(CK, weights_only=True, mmap=True, map_location='cpu')
    u = sd['lm_head.weight'].double().cuda()
    l,r,d = [sd[f'transformer.h.17.mlp.{key}.weight'].double().cuda()
             for key in ['Left','Right','Down']]
    assert u.shape == (50304,1152) and l.shape == r.shape == (4608,1152)
    original_gram = product_cross(l,r,l,r)
    scale = original_gram.diag().sqrt()
    gram = original_gram / scale[:,None] / scale[None,:]
    ugram = u.T @ u
    mean = u.mean(0)
    centered_ugram = ugram - len(u) * mean[:,None] * mean[None,:]
    random_order = torch.randperm(4608, generator=torch.Generator().manual_seed(944)).tolist()
    full_reference = json.loads((P/'FULLU_OUTPUT_FUNCTIONS_V1_AUDIT.json').read_text())
    arms = {}
    errors = []
    for name, metric in [('full', ugram), ('centered', centered_ugram)]:
        arm_start = time.perf_counter()
        root = torch.linalg.cholesky((metric + metric.T)/2)
        native_writer = root.T @ d
        raw_cross = native_writer @ original_gram
        total = float((raw_cross * native_writer).sum())
        cross = raw_cross / scale[None,:]
        covariance = raw_cross @ native_writer.T
        ev = torch.linalg.eigvalsh((covariance+covariance.T)/2).flip(0)
        expected = full_reference['native_total'] if name == 'full' else 92104252412.19983
        trace_error = abs(total/expected - 1)
        errors.append(trace_error)
        support, gains = select(gram, cross, 1024)
        assert len(support) == 1024
        rows = {}
        for count in BUDGETS:
            fitted = conditional(gram, cross, support[:count], total)
            random_fit = conditional(gram, cross, random_order[:count], total)
            gain_capture = sum(gains[:count])/total
            gain_error = abs(fitted['capture']-gain_capture)
            ceiling = float(ev[:count].sum()/total)
            errors.extend([fitted['solve_relative_residual'],random_fit['solve_relative_residual'],
                           gain_error,max(0.,fitted['capture']-ceiling)])
            rows[str(count)] = dict(selected=fitted,random=random_fit,
                                   output_rank_upper_bound=ceiling,gain_capture=gain_capture,
                                   absolute_gain_replay_error=gain_error,
                                   exported_reader_writer_numbers=3*1152*count)
            print(json.dumps(dict(metric=name,count=count,**rows[str(count)])),flush=True)
        arms[name] = dict(native_total=total,native_trace_relative_error=trace_error,
                         support=support,rows=rows,wall_seconds=time.perf_counter()-arm_start)
    valid = all(e <= 1e-8 for e in errors) and all(torch.isfinite(torch.tensor(errors)).tolist())
    result = dict(predictions={'pred_a_instrument':valid,
        'pred_b_small_reference':valid and arms['full']['rows']['128']['selected']['capture']>=.08698912596,
        'pred_c_broader_capacity':valid and arms['full']['rows']['1024']['selected']['capture']>=.50},
        arms=arms, maximum_instrument_error=max(errors),
        price=dict(body_forwards=0,corpus_access=False,large_checkpoint=False),
        checkpoint=str(CK),checkpoint_sha256='680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3',
        wall_seconds=time.perf_counter()-start,
        scope='Weight-only native dictionary OLS and exact conditional writers. Greedy support, no global optimum or circuit identification.')
    with out.open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(predictions=result['predictions'],wall_seconds=result['wall_seconds'])),flush=True)
    assert valid


if __name__ == '__main__':
    main()
