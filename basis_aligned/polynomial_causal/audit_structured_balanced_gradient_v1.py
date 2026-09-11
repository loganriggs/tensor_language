"""CPU seed0 same-objective gradient redistribution after exact stage balancing.
This is not a speed benchmark or an optimizer convergence claim.
"""
import hashlib
import json
from pathlib import Path
import time
import torch
from structured_bilinear_bank_v1 import StructuredBank
from chunked_bilinear_coefficient_v1 import value_gradient

P = Path(__file__).resolve().parent
CK = Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def diagnostic(model, capture):
    return dict(relative_stationarity=max(float(p.grad.norm()*p.detach().norm().clamp_min(1)) for p in model.parameters())/capture,
                gradient_max=max(float(p.grad.abs().max()) for p in model.parameters()))


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    started = time.perf_counter()
    previous = json.loads((P/'STRUCTURED_BILINEAR_NATIVE_V2_SEED_0.json').read_text())
    balanced = json.loads((P/'STRUCTURED_STAGE_BALANCE_V1_AUDIT.json').read_text())['starts'][0]
    models = []
    for cache in (previous['cache'], balanced['cache']):
        assert digest(cache['path']) == cache['sha256']
        saved = torch.load(cache['path'], map_location='cpu', weights_only=True)
        m = StructuredBank([2]*7+[3,3], branches=4)
        m.load_state_dict(saved['model'])
        models.append(m)
    sd = torch.load(CK, map_location='cpu', weights_only=True, mmap=True)
    u = sd['lm_head.weight'].double()
    whitener = torch.linalg.cholesky(u.T@u).T
    del u
    l, r, d = [sd[f'transformer.h.17.mlp.{key}.weight'].double() for key in ('Left','Right','Down')]
    native = l, r, whitener@d
    total = json.loads((P/'FULLU_OUTPUT_FUNCTIONS_V1_AUDIT.json').read_text())['native_total']
    first = models[0].factors()
    white = whitener@first[2]
    loss, gradients, details = value_gradient(*native, first[0], first[1], white, total, .01, 256)
    torch.autograd.backward((first[0], first[1], white), gradients)
    original = diagnostic(models[0],1-details['residual'])
    second = models[1].factors()
    errors = [float((a.detach()-b.detach()).norm()/a.detach().norm()) for a,b in zip(first,second)]
    torch.autograd.backward((second[0], second[1], whitener@second[2]), gradients)
    after = diagnostic(models[1],1-details['residual'])
    replay = dict(loss=abs(float(loss)-previous['final']['loss'])/previous['final']['loss'],
        stationarity=abs(original['relative_stationarity']-previous['final']['relative_stationarity'])/previous['final']['relative_stationarity'],
        factor_error=max(errors))
    ratios = {key:original[key]/after[key] for key in original}
    result = dict(predictions={
        'pred_a_replay':max(replay['loss'],replay['stationarity'])<=1e-6 and replay['factor_error']<=1e-10,
        'pred_b_stationarity_halved':ratios['relative_stationarity']>=2,
        'pred_c_max_gradient_halved':ratios['gradient_max']>=2},
        replay=replay, original=original, balanced=after, reduction_factors=ratios,
        loss=float(loss),capture=1-details['residual'], seconds=time.perf_counter()-started,
        source_sha256=digest(__file__), inputs=[previous['cache'],balanced['cache']],
        gpu_access=False,corpus_access=False,
        scope='Same function and factor-space gradient, chain rule through alternate stages; no optimization-speed inference.')
    with (P/'STRUCTURED_BALANCED_GRADIENT_V1_AUDIT.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
