"""Execute NORMALIZED_ROUTER_FOLD_V1_PREREGISTRATION.md on CPU only."""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import torch
import folded_normalized_router_v1 as f

ROOT = Path(__file__).resolve().parent
CHECKPOINT = Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
EXPECTED = '680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def error(a, b):
    return {'relative_l2': float((a-b).norm() / b.norm().clamp_min(1e-30)),
            'max_abs': float((a-b).abs().max())}


def main():
    out = ROOT / 'NORMALIZED_ROUTER_FOLD_V1_RESULT.json'
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(2)
    start = time.perf_counter()
    assert digest(CHECKPOINT) == EXPECTED
    state = torch.load(CHECKPOINT, weights_only=True, mmap=True, map_location='cpu')
    names = ['c_q', 'c_k', 'c_q2', 'c_k2']
    weights = [state[f'transformer.h.0.attn.{name}.weight'][:128].double() for name in names]
    assert all(w.shape == (128, 1152) for w in weights)
    generator = torch.Generator().manual_seed(9111260)
    x, y = [torch.randn(64, 1152, generator=generator, dtype=torch.float64) for _ in range(2)]
    x, y = [v / v.square().mean(-1, keepdim=True).sqrt() for v in (x,y)]
    half = torch.cat((torch.full((32,), 2.), torch.full((32,), .5))).double()
    scale = torch.cat((half, half))
    signs = torch.cat((torch.ones(32), -torch.ones(32))).double().repeat(2)
    scaled, signed = f.paired_change(weights, scale), f.paired_change(weights, signs)
    cells = []
    for t, s in [(0,0), (1,0), (31,7), (127,32)]:
        sig, scaled_sig, signed_sig = [f.fold(w, t, s) for w in (weights, scaled, signed)]
        score, numerator = f.evaluate(sig, x, y)
        changed, changed_numerator = f.evaluate(scaled_sig, x, y)
        sign_score, _ = f.evaluate(signed_sig, x, y)
        cell = {'t': t, 's': s, 'n':64,
                'fp64_bridge': error(score, f.direct(weights,x,y,t,s)),
                'fp32_bridge': error(f.direct([w.float() for w in weights],x.float(),y.float(),t,s).double(),score),
                'scaled_numerator_matrices': [error(a[0], b[0]) for a,b in zip(scaled_sig,sig)],
                'scaled_numerator_values': error(changed_numerator,numerator),
                'scaled_router': error(changed,score),
                'signed_numerator_matrices': [error(a[0], b[0]) for a,b in zip(signed_sig,sig)],
                'signed_router': error(sign_score,score),
                'native_router_rms':float(score.square().mean().sqrt()),
                'scaled_router_rms':float(changed.square().mean().sqrt())}
        if t == s == 0:
            same, _ = f.evaluate(sig,x,x)
            same_scaled, _ = f.evaluate(scaled_sig,x,x)
            cell['same_token_bridge'] = error(same,f.direct(weights,x,x,t,s))
            cell['same_token_scaled_router'] = error(same_scaled,same)
        cells.append(cell)
        del sig, scaled_sig, signed_sig
    within = lambda e, bar: e['relative_l2'] <= bar and e['max_abs'] <= bar
    predictions = {
        'pred_a_fp64_fp32_bridges': all(within(c['fp64_bridge'],1e-10) and within(c['fp32_bridge'],1e-5) for c in cells),
        'pred_b_identical_numerator_different_router': all(all(e['relative_l2'] <= 1e-10 for e in c['scaled_numerator_matrices']) and c['scaled_router']['relative_l2'] >= .01 for c in cells),
        'pred_c_valid_sign_gauge': all(all(e['relative_l2'] <= 1e-10 for e in c['signed_numerator_matrices']) and c['signed_router']['relative_l2'] <= 1e-10 for c in cells),
        'pred_d_same_token_bridge': within(cells[0]['same_token_bridge'],1e-10)}
    result = {'schema':'normalized_router_fold.v1','predictions':predictions,'cells':cells,
              'checkpoint_sha256':EXPECTED,'site':'attn0.head0','seed':9111260,
              'domain':'local continuous normalized residual pairs, not native text',
              'epsilon':f.EPS,'model_forwards':0,'gpu_accessed':False,'native_parameters_retained':545902902,
              'folded_signature_scalars':6*1152**2,'original_four_maps_scalars':4*128*1152,
              'per_matrix_bytes':1152**2*8,'fitted_parameters':0,
              'hashes':{p.name:digest(p) for p in [Path(__file__),Path(f.__file__),ROOT/'NORMALIZED_ROUTER_FOLD_V1_PREREGISTRATION.md',ROOT.parents[1]/'jacclust/tt_model.py']},
              'seconds':time.perf_counter()-start,'finished_utc':datetime.now(timezone.utc).isoformat()}
    with out.open('x') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({'predictions':predictions,'seconds':result['seconds'],
                      'scale_errors':[c['scaled_router']['relative_l2'] for c in cells],
                      'fp64_max':max(c['fp64_bridge']['max_abs'] for c in cells),
                      'fp32_max':max(c['fp32_bridge']['max_abs'] for c in cells)}))


if __name__ == '__main__':
    main()
