"""HOW MANY DIRECTIONS IS THE WHOLE TAIL OVER-LARGE ALONG? (Claude, LANE 1, CPU-ONLY)

SS2923 adopted a rank-32 projection PER tail link map -- 32 objects. SS2924 showed their subspaces are partly shared (6-8x the
random null) but far from identical (0.17-0.23 overlap), so one shared rank-32 projection cannot replace 32. Pairwise overlap
cannot say how big the UNION is: 32 subspaces can be pairwise 20% aligned and span anywhere from 32 to 1024 dimensions.

M = [U1 | ... | U32] is 1152x1024 with ||M||_F^2 = 1024 exactly. f(k) = sum_{i<=k} sigma_i^2 / 1024 is the fraction of all 32
correcting subspaces lying in the top k directions of their union; k90 is the smallest k with f(k) >= 0.90. The control is the
SAME computation on 32 independent random 32-dimensional subspaces.

Price: 0 GPU forwards, 0 GPU-seconds. CPU only; does not touch the runner, the queue or the model.
"""
# BQGATE: ANALYSIS  pred_a_the_total_energy_is_exactly_1024
#                   pred_b_the_reload_is_faithful
#                   pred_c_the_union_is_more_concentrated_than_chance
#                   pred_d_the_union_is_substantially_smaller_than_1024
#                   pred_e_the_union_is_stable_across_depth
import json, sys, time, os, hashlib
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient/ops')
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'tail_subspace_union_results.json'
PREREG = PT + '../polynomial_causal/TAIL_SUBSPACE_UNION_PREREGISTRATION.md'
PREREG_SHA = "a7c07675d384bf06e9a9fcbb11d1e14b340d02848b1202e8a91b348acd56ec6b"
CACHE_KEY = "5c5a8900d60fa34272bfe7184fad48c2"
R, D, K = 32, 1152, 128
BARS = {"energy_tol": 0.5, "concentration": 1.5, "k90_frac": 0.7, "stability": 1.5}

if os.environ.get('BQLIB_DRYRUN') == '1':
    _p = PT + f'.fitcache/stack_{CACHE_KEY}.pt'
    if not os.path.exists(_p):
        print(f'DRYRUN FAIL: missing {_p}'); raise SystemExit(1)
    print('DRYRUN OK: SS2915 cache present; CPU-only, 0 GPU forwards'); raise SystemExit(0)

import torch
import frontier_fitcache as FC


def _sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def profile(bases):
    """Energy profile of a union of orthonormal bases. Returns (total, f(k) list, k90, top-K basis)."""
    M = torch.cat(bases, dim=1)
    total = float(M.pow(2).sum())
    U, sv, _ = torch.linalg.svd(M, full_matrices=False)
    e = sv.pow(2)
    cum = torch.cumsum(e, 0) / e.sum()
    k90 = int((cum >= 0.90).nonzero()[0]) + 1
    return total, cum, k90, U[:, :K].contiguous(), M.shape


def overlap(Ua, Ub):
    return float((Ua.T @ Ub).pow(2).sum()) / Ua.shape[1]


if __name__ == '__main__':
    t0 = time.time()
    if _sha(PREREG) != PREREG_SHA:
        raise RuntimeError(f'frozen hash mismatch: {PREREG}')
    got = FC.load_stack(CACHE_KEY)
    if got is None:
        raise RuntimeError('SS2915 cache absent; rerun ops/frontier_stack_dump.py')
    S, _, _ = got
    maps = {}
    for k, v in S.items():
        if k.endswith('L') and isinstance(v, tuple) and v[0] == 'attnd':
            for c, W in v[3].items():
                maps[(v[1], c)] = W.float()
    layers = sorted({li for li, _ in maps})
    classes = sorted({c for _, c in maps})
    U = {}
    for key, W in sorted(maps.items()):
        Uw, _, _ = torch.linalg.svd(W, full_matrices=False)
        U[key] = Uw[:, :R].contiguous()
    print(f'{len(maps)} maps, layers {layers}, classes {classes}', flush=True)

    total, cum, k90, Utop, shape = profile([U[k] for k in sorted(U)])
    fK = float(cum[K-1])

    # ---- the control: 32 independent random 32-dim subspaces, SAME functions ----
    g = torch.Generator().manual_seed(20260904)
    rb = [torch.linalg.qr(torch.randn(D, R, generator=g))[0] for _ in range(len(maps))]
    rtotal, rcum, rk90, rUtop, _ = profile(rb)
    rfK = float(rcum[K-1])

    # ---- pred_e: the half-split across depth ----
    loA, loB = layers[:4], layers[4:]
    _, _, _, UA, _ = profile([U[(li, c)] for li in loA for c in classes])
    _, _, _, UB, _ = profile([U[(li, c)] for li in loB for c in classes])
    stab = overlap(UA, UB)
    rand_stab = overlap(rUtop[:, :K], torch.linalg.qr(torch.randn(D, K, generator=g))[0])

    pa = abs(total - 1024.0) <= BARS['energy_tol']
    pb = (len(maps) == 32 and len(layers) == 8 and len(classes) == 4
          and tuple(shape) == (D, 1024) and all(W.shape == (D, D) for W in maps.values()))
    pc = fK >= BARS['concentration'] * rfK
    pd = k90 <= BARS['k90_frac'] * rk90
    pe = stab >= BARS['stability'] * (K / D)

    preds = {'pred_a_the_total_energy_is_exactly_1024': bool(pa),
             'pred_b_the_reload_is_faithful': bool(pb),
             'pred_c_the_union_is_more_concentrated_than_chance': bool(pc),
             'pred_d_the_union_is_substantially_smaller_than_1024': bool(pd),
             'pred_e_the_union_is_stable_across_depth': bool(pe)}
    nulls = {'a_null_the_bases_are_not_orthonormal': bool(not pa),
             'b_null_the_reload_is_not_faithful': bool(not pb),
             'c_null_the_union_is_no_more_concentrated_than_chance': bool(not pc),
             'd_null_the_union_fills_the_available_space': bool(not pd),
             'e_null_the_shared_structure_does_not_survive_a_half_split': bool(not pe)}
    res = {'rung': 'tail_subspace_union', 'preds': preds, 'nulls': nulls, 'bars': BARS,
           'summary': {'n_maps': len(maps), 'M_shape': list(shape), 'rank_per_map': R,
                       'total_energy': round(total, 2), 'total_energy_expected': 1024,
                       'f_128': round(fK, 4), 'f_128_random': round(rfK, 4),
                       'concentration_ratio': round(fK / rfK, 2),
                       'f_32': round(float(cum[31]), 4), 'f_32_random': round(float(rcum[31]), 4),
                       'f_256': round(float(cum[255]), 4), 'f_256_random': round(float(rcum[255]), 4),
                       'f_512': round(float(cum[511]), 4), 'f_512_random': round(float(rcum[511]), 4),
                       'k90': k90, 'k90_random': rk90, 'k90_ratio': round(k90 / rk90, 3),
                       'half_split_stability': round(stab, 4),
                       'half_split_random_reference': round(rand_stab, 4),
                       'half_split_null': round(K / D, 4),
                       'layers_first_half': loA, 'layers_second_half': loB},
           'price': {'gpu_forwards': 0, 'forwards_instrumented': False, 'pipeline_runs': 0,
                     'backwards': 0, 'fitted_parameters': 0, 'gpu_seconds': 0.0,
                     'cpu_seconds': round(time.time() - t0, 1)},
           'hashes': {PREREG: PREREG_SHA}, 'self_reviewed': True}
    json.dump(res, open(OUT, 'w'), indent=1)
    print(f"(a) total energy {total:.2f} vs 1024: {'HELD' if pa else 'FAILED'}")
    print(f"(b) 32 maps, M {tuple(shape)}: {'HELD' if pb else 'FAILED'}")
    print(f"(c) f(128) {fK:.4f} vs random {rfK:.4f} = {fK/rfK:.2f}x (bar 1.5x): {'HELD' if pc else 'FAILED'}")
    print(f"(d) k90 {k90} vs random {rk90} (bar <= {BARS['k90_frac']}x = {BARS['k90_frac']*rk90:.0f}): {'HELD' if pd else 'FAILED'}")
    print(f"(e) half-split stability {stab:.4f} vs null {K/D:.4f} (bar {BARS['stability']*K/D:.4f}): {'HELD' if pe else 'FAILED'}")
    print(f"    energy profile: f(32) {float(cum[31]):.3f} | f(128) {fK:.3f} | f(256) {float(cum[255]):.3f} | f(512) {float(cum[511]):.3f}")
    print(f"    random profile: f(32) {float(rcum[31]):.3f} | f(128) {rfK:.3f} | f(256) {float(rcum[255]):.3f} | f(512) {float(rcum[511]):.3f}")
    print(f"wrote {OUT} ({res['price']['cpu_seconds']:.0f}s CPU, 0 GPU-seconds)")
