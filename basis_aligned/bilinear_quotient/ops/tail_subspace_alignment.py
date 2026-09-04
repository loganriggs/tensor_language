"""ARE THE 32 CORRECTING DIRECTIONS SHARED ACROSS LAYERS AND CLASSES? (Claude, LANE 1, CPU-ONLY)

SS2921: the tail correction is a rank-32 projection per link map. SS2915: the whole fitted stack is on disk, bit-exact and
CPU-loadable. So the structural question costs zero GPU-seconds -- are those 32 directions the SAME 32 across the eight tail
layers and the four link classes? One shared subspace would replace 32 objects with one.

Statistic: mean squared principal cosine between top-32 LEFT singular subspaces, ||U'V||_F^2 / 32. It is 1.0 for identical
subspaces and, for two independent uniformly-random 32-dimensional subspaces of R^1152, has expectation exactly 32/1152 = 0.0278.
That closed form is the null AND the control -- the same function is called on random orthonormal draws.

Price: 0 GPU forwards, 0 GPU-seconds. CPU only; does not touch the runner, the queue or the model.
"""
# BQGATE: ANALYSIS  pred_a_the_random_baseline_matches_theory
#                   pred_b_the_reload_is_faithful
#                   pred_c_layers_share_a_subspace
#                   pred_d_classes_share_a_subspace
#                   pred_e_the_maps_are_not_trivially_similar
import json, sys, time, os, hashlib
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient/ops')
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'tail_subspace_alignment_results.json'
PREREG = PT + '../polynomial_causal/TAIL_SUBSPACE_ALIGNMENT_PREREGISTRATION.md'
PREREG_SHA = "f18b9cb519e8ff1c34be33e5a33c1779ebaa55eebcbb760472dafc032c21f12c"
CACHE_KEY = "5c5a8900d60fa34272bfe7184fad48c2"    # SS2915's verified dump
R = 32
D = 1152
BARS = {"random_tol": 0.005, "share": 4.0, "trivial": 0.9}

if os.environ.get('BQLIB_DRYRUN') == '1':
    _p = PT + f'.fitcache/stack_{CACHE_KEY}.pt'
    if not os.path.exists(_p):
        print(f'DRYRUN FAIL: missing {_p} -- rerun ops/frontier_stack_dump.py'); raise SystemExit(1)
    print(f'DRYRUN OK: SS2915 cache present ({os.path.getsize(_p)/1e6:.0f} MB); CPU-only, 0 GPU forwards')
    raise SystemExit(0)

import torch
import frontier_fitcache as FC


def _sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def overlap(Ua, Ub):
    """Mean squared principal cosine between two orthonormal bases: 1.0 identical, R/D for random."""
    return float((Ua.T @ Ub).pow(2).sum()) / Ua.shape[1]


if __name__ == '__main__':
    t0 = time.time()
    if _sha(PREREG) != PREREG_SHA:
        raise RuntimeError(f'frozen hash mismatch: {PREREG}')
    got = FC.load_stack(CACHE_KEY)           # CPU + mmap: the path SS2911 established for analysis
    if got is None:
        raise RuntimeError('SS2915 cache absent; rerun ops/frontier_stack_dump.py')
    S, cfgF, order2 = got

    # ---- pred_b: read the objects we think we are reading (the SS2879 mistake) ----
    maps = {}
    for k, v in S.items():
        if not (k.endswith('L') and isinstance(v, tuple) and v[0] == 'attnd'):
            continue
        li, LW = v[1], v[3]
        for c, W in LW.items():
            maps[(li, c)] = W.float()
    shapes_ok = all(W.shape == (D, D) for W in maps.values())
    layers = sorted({li for li, _ in maps})
    classes = sorted({c for _, c in maps})
    pb = (len(maps) == 32 and shapes_ok and len(layers) == 8 and len(classes) == 4)
    print(f'loaded {len(maps)} link maps: layers {layers}, classes {classes}, shapes_ok {shapes_ok}', flush=True)

    # ---- top-32 left singular subspaces ----
    U = {}
    for key, W in sorted(maps.items()):
        Uw, _, _ = torch.linalg.svd(W, full_matrices=False)
        U[key] = Uw[:, :R].contiguous()
        print(f'  svd {key} done', flush=True)

    # ---- pred_a: the control, through the SAME function ----
    g = torch.Generator().manual_seed(20260904)
    rand = []
    for _ in range(12):
        A = torch.linalg.qr(torch.randn(D, R, generator=g))[0]
        B = torch.linalg.qr(torch.randn(D, R, generator=g))[0]
        rand.append(overlap(A, B))
    rand_mean = sum(rand) / len(rand)
    theory = R / D
    pa = abs(rand_mean - theory) <= BARS['random_tol']

    # ---- pred_c: across layers, same class ----
    cross_layer = [overlap(U[(la, c)], U[(lb, c)])
                   for c in classes for i, la in enumerate(layers) for lb in layers[i+1:]]
    # ---- pred_d: across classes, same layer ----
    cross_class = [overlap(U[(li, ca)], U[(li, cb)])
                   for li in layers for i, ca in enumerate(classes) for cb in classes[i+1:]]
    cl_mean = sum(cross_layer) / len(cross_layer)
    cc_mean = sum(cross_class) / len(cross_class)
    pc = cl_mean >= BARS['share'] * theory
    pd = cc_mean >= BARS['share'] * theory

    # ---- pred_e: the maps are not trivially the same matrix ----
    def cosmat(A, B):
        return float((A * B).sum() / (A.norm() * B.norm()))
    raw = ([cosmat(maps[(la, c)], maps[(lb, c)])
            for c in classes for i, la in enumerate(layers) for lb in layers[i+1:]] +
           [cosmat(maps[(li, ca)], maps[(li, cb)])
            for li in layers for i, ca in enumerate(classes) for cb in classes[i+1:]])
    raw_max = max(abs(x) for x in raw)
    pe = raw_max < BARS['trivial']

    # adjacent-layer detail: is any sharing local in depth rather than global?
    adj = {f'{la}-{la+1}': round(sum(overlap(U[(la, c)], U[(la+1, c)]) for c in classes) / len(classes), 4)
           for la in layers[:-1]}

    preds = {'pred_a_the_random_baseline_matches_theory': bool(pa),
             'pred_b_the_reload_is_faithful': bool(pb),
             'pred_c_layers_share_a_subspace': bool(pc),
             'pred_d_classes_share_a_subspace': bool(pd),
             'pred_e_the_maps_are_not_trivially_similar': bool(pe)}
    nulls = {'a_null_the_statistic_is_miscomputed': bool(not pa),
             'c_null_the_layers_do_not_share_a_subspace': bool(not pc),
             'd_null_the_classes_do_not_share_a_subspace': bool(not pd),
             'e_null_the_maps_are_trivially_similar': bool(not pe)}
    res = {'rung': 'tail_subspace_alignment', 'preds': preds, 'nulls': nulls, 'bars': BARS,
           'summary': {'n_maps': len(maps), 'layers': layers, 'classes': classes, 'rank': R,
                       'random_null_theory': round(theory, 4),
                       'random_null_measured': round(rand_mean, 4),
                       'cross_layer_mean': round(cl_mean, 4),
                       'cross_layer_min': round(min(cross_layer), 4),
                       'cross_layer_max': round(max(cross_layer), 4),
                       'cross_class_mean': round(cc_mean, 4),
                       'cross_class_min': round(min(cross_class), 4),
                       'cross_class_max': round(max(cross_class), 4),
                       'cross_layer_over_null': round(cl_mean / theory, 2),
                       'cross_class_over_null': round(cc_mean / theory, 2),
                       'adjacent_layer_overlap': adj,
                       'raw_matrix_cosine_max': round(raw_max, 4)},
           'price': {'gpu_forwards': 0, 'forwards_instrumented': False, 'pipeline_runs': 0,
                     'backwards': 0, 'fitted_parameters': 0, 'gpu_seconds': 0.0,
                     'cpu_seconds': round(time.time() - t0, 1)},
           'hashes': {PREREG: PREREG_SHA}, 'self_reviewed': True}
    json.dump(res, open(OUT, 'w'), indent=1)
    print(f"(a) random null {rand_mean:.4f} vs theory {theory:.4f}: {'HELD' if pa else 'FAILED'}")
    print(f"(b) 32 maps, 8 layers x 4 classes, all {D}x{D}: {'HELD' if pb else 'FAILED'}")
    print(f"(c) cross-LAYER overlap {cl_mean:.4f} = {cl_mean/theory:.1f}x null (bar {BARS['share']}x): {'HELD' if pc else 'FAILED'}")
    print(f"(d) cross-CLASS overlap {cc_mean:.4f} = {cc_mean/theory:.1f}x null (bar {BARS['share']}x): {'HELD' if pd else 'FAILED'}")
    print(f"(e) raw matrix cosine max {raw_max:.4f} < 0.9: {'HELD' if pe else 'FAILED'}")
    print(f"    adjacent layers: {adj}")
    print(f"wrote {OUT} ({res['price']['cpu_seconds']:.0f}s CPU, 0 GPU-seconds)")
