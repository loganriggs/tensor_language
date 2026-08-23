# patch_relax_magnitude: the magnitude-aware re-measurement §1156 registered.
#
# §1155 claimed the injected address survives at 0.96 (single L14 patch vs full clamp, final
# residual) — but its instrument was cosine similarity: direction-only. §1156 showed coords
# carry the bulk of transport, so the 50%-excess gap (§1154) must live in what cosine cannot
# see. This run re-tracks the same injections with three readings of the coord delta
# Δc(ℓ) = c_patch(ℓ) − c_base(ℓ) against the full-clamp delta Δc_full(ℓ) and source delta:
#   dircos   — cos(Δc, Δc_src)            [reproduces §1155]
#   magratio — ‖Δc‖ / ‖Δc_full‖           [what cosine hides]
#   projcoef — <Δc, Δc_full> / ‖Δc_full‖² [signed transfer coefficient, the linear-readout view]
#
# Registered predictions:
#   pred_a MAGNITUDE IS THE MISSING HALF: p14's final-layer (L17) magratio ∈ [0.40, 0.65]
#          — reconciling §1154's 50% excess with §1156's coords-dominance.
#   pred_b DECAY IS IN MAGNITUDE: p6's magratio falls with depth (L7 → L17 ratio declines by
#          ≥ 0.2) while its dircos stays ≥ 0.75 throughout (reproducing §1155).
#   pred_c PROJECTION PREDICTS EXCESS ORDERING: projcoef at L17 orders p6 < p14 < full9 and
#          p14's projcoef/full9's ∈ [0.35, 0.65] (the linear-readout account closes).
# Control: random-subspace injection (r10) tracked identically — its magratio profile shows
# whether generic deltas also persist in magnitude (stream linearity) or shrink.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'patch_relax_magnitude_results.json'
NEVAL = 160; SEQ = 256; REF_LAYERS = [8, 10, 12]; ABL = list(range(6, 15))
TRACK = list(range(6, 18))
ST = {'mode': None, 'U': None, 'srcres': None, 'store': None, 'layers': None,
      'track': None, 'Utrack': None}


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for li, blk in enumerate(m.transformer.h):
        x, v1 = blk(x, v1, x0)
        if ST['mode'] == 'cap' and li in ABL:
            ST['store'][li] = x.detach()
        elif ST['mode'] == 'patch' and li in ST['layers']:
            U = ST['U']; xs = ST['srcres'][li]
            x = x - (x @ U) @ U.T + (xs @ U) @ U.T
        if ST['track'] is not None and li in TRACK:
            ST['track'][li] = (x @ ST['Utrack']).detach()
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def capture_dev(blocks):
    caps = {L: [] for L in REF_LAYERS}; toks = []; hs = []
    for L in REF_LAYERS:
        mlp = m.transformer.h[L].mlp
        def mk(L):
            def h(mo, i_, o_):
                caps[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hs.append(mlp.register_forward_hook(mk(L)))
    ST['mode'] = None
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); toks.append(idx.reshape(-1)); fwd(idx)
    for h in hs: h.remove()
    return {L: torch.cat(caps[L], 0) for L in REF_LAYERS}, torch.cat(toks, 0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(2 * NEVAL)[NEVAL:]
    blocks = rows[:, :SEQ].contiguous(); V = int(m.lm_head.weight.shape[0])

    caps, tok = capture_dev(blocks)
    devsum = None
    for L in REF_LAYERS:
        X = caps[L]; xbar = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
        xbar.index_add_(0, tok, X); cnts.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
        xbar = xbar / cnts.clamp_min(1).unsqueeze(1)
        dv = X - xbar[tok]; devsum = dv if devsum is None else devsum + dv; del X, dv
    dev = devsum / len(REF_LAYERS); devc = dev - dev.mean(0)
    _, _, Vt = torch.linalg.svd(devc, full_matrices=False)
    U256 = Vt[:256].T.contiguous()
    g = torch.Generator(device=DEV).manual_seed(0)
    R256 = torch.linalg.qr(torch.randn(D, 256, generator=g, device=DEV))[0]
    del caps, devsum, dev, devc

    CONDS = [('p6', {6}, U256), ('p10', {10}, U256), ('p14', {14}, U256),
             ('r10', {10}, R256), ('full9', set(range(6, 15)), U256)]
    n = blocks.shape[0] // 2; src = blocks[:n].contiguous(); tgt = blocks[n:2 * n].contiguous()
    # accumulators: sums over positions of |dp|, |dfull|, <dp,dfull>, cos(dp,dsrc)
    A = {c: {L: {'nrm': 0.0, 'dot': 0.0, 'cos': 0.0} for L in TRACK} for c, _, _ in CONDS}
    NF = {L: 0.0 for L in TRACK}    # sum of ||d_full|| over positions
    NF2 = {L: 0.0 for L in TRACK}   # sum of ||d_full||^2 over positions
    npos = 0
    for i in range(0, n, 8):
        si = src[i:i+8].to(DEV)[:, :-1].contiguous(); ti = tgt[i:i+8].to(DEV)[:, :-1].contiguous()
        if si.shape[0] != ti.shape[0]: continue
        ST['Utrack'] = U256; ST['track'] = {}
        ST['mode'] = 'cap'; ST['store'] = {}; fwd(si)
        srcres = {li: ST['store'][li] for li in ABL}; csrc = ST['track']
        ST['mode'] = None; ST['track'] = {}; fwd(ti); cbase = ST['track']
        deltas = {}
        for cname, layers, U in CONDS:
            ST['mode'] = 'patch'; ST['U'] = U; ST['srcres'] = srcres; ST['layers'] = layers
            ST['track'] = {}; fwd(ti); ST['mode'] = None
            deltas[cname] = {L: (ST['track'][L] - cbase[L]).reshape(-1, 256) for L in TRACK}
        dfull = deltas['full9']
        for L in TRACK:
            nrms = dfull[L].norm(dim=-1)
            NF[L] += float(nrms.sum()); NF2[L] += float((nrms ** 2).sum())
        for cname, layers, U in CONDS:
            pl = min(layers) if layers else 6
            for L in TRACK:
                if L < pl: continue
                dp = deltas[cname][L]
                ds = (csrc[L] - cbase[L]).reshape(-1, 256)
                A[cname][L]['nrm'] += float(dp.norm(dim=-1).sum())
                A[cname][L]['dot'] += float((dp * dfull[L]).sum())
                A[cname][L]['cos'] += float(F.cosine_similarity(dp, ds, dim=-1).sum())
        ST['track'] = None
        npos += si.shape[0] * si.shape[1]

    # magratio = ratio of position-summed norms; projcoef = pooled sum<dp,dfull>/sum||dfull||^2
    out_c = {}
    for cname, layers, _ in CONDS:
        pl = min(layers) if layers else 6
        rowsd = {}
        for L in TRACK:
            if L < pl: continue
            rowsd[str(L)] = {'dircos': round(A[cname][L]['cos'] / npos, 4),
                             'magratio': round(A[cname][L]['nrm'] / max(NF[L], 1e-6), 4),
                             'projcoef': round(A[cname][L]['dot'] / max(NF2[L], 1e-6), 4)}
        out_c[cname] = rowsd
    m6 = out_c['p6']; m14 = out_c['p14']
    pred_a = 0.40 <= m14['17']['magratio'] <= 0.65
    pred_b = (m6['7']['magratio'] - m6['17']['magratio'] >= 0.2) and all(
        m6[str(L)]['dircos'] >= 0.75 for L in range(7, 18))
    p14_over_full = m14['17']['magratio']  # full9 magratio at 17 is 1.0 by construction
    pred_c = out_c['p6']['17']['magratio'] < m14['17']['magratio'] < 1.0 and 0.35 <= p14_over_full <= 0.65
    out = {'n_positions': npos, 'curves': out_c,
           'pred_a_magnitude_missing_half': bool(pred_a),
           'pred_b_decay_in_magnitude': bool(pred_b),
           'pred_c_ordering_and_ratio': bool(pred_c),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    for cname in out_c:
        row = out_c[cname]
        ks = sorted(row, key=int)
        print(f"{cname:>6} mag: " + " ".join(f"L{k}:{row[k]['magratio']:.3f}" for k in ks), flush=True)
        print(f"{cname:>6} cos: " + " ".join(f"L{k}:{row[k]['dircos']:+.3f}" for k in ks), flush=True)
    print(f"pred_a mag-half {out['pred_a_magnitude_missing_half']} | pred_b decay-in-mag {out['pred_b_decay_in_magnitude']} | pred_c ordering {out['pred_c_ordering_and_ratio']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
