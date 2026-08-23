# patch_relaxation: measure the RELAXATION CONSTANT of the content clamp (§1154 follow-up).
#
# §1154: the transported address persists across unpatched blocks (depth-uniform), but excess
# grows with patch density — so each unpatched block must pull the coords PARTWAY back toward
# the target's own topic (the 30% created-in-flight, §1125-27). This experiment measures that
# pull directly: patch the content coords at ONE layer only, then track how the injected
# address decays through the remaining blocks.
#
# Protocol: §1150-54 harness (K=256, full coverage, fresh rows). Patch at a single layer
# L0p ∈ {6, 10, 14}; capture content coords c(ℓ) = x(ℓ)·U at every layer ℓ ∈ 6..17 in three
# runs (source, target-base, target-patched). Retention at layer ℓ > L0p:
#     S(ℓ) = cos( c_patch(ℓ) − c_base(ℓ),  c_src(ℓ) − c_base(ℓ) )   [per position, meaned]
# S = 1 right after the patch by construction; each subsequent block multiplies it by the
# survival factor.
#
# Registered predictions:
#   pred_a RELAXATION IS PARTIAL, NOT COLLAPSE: one block later S ≥ 0.6 (the §1122 transport
#          R² 0.92 says most of the state carries linearly); after 8 blocks (L6 patch read at
#          L14) S still ≥ 0.3 (else §1154's early3 ≈ late3 could not hold).
#   pred_b RATE IS DEPTH-UNIFORM: per-block survival ratio S(ℓ+1)/S(ℓ) for the L6 and L10
#          injections agrees within 0.1 at overlapping depths (same dynamics everywhere —
#          matches §1154's interchangeable thirds).
#   pred_c RECONCILES §1154 QUANTITATIVELY: S at L17 (what the readout sees) for the L14
#          patch, divided by S at L17 for the full-9 patch (≈1), is within ±0.15 of the
#          observed excess ratio 0.115/0.230 = 0.50.
# Control: same S computed for a RANDOM-subspace single-layer patch at L10 (should decay
# faster / carry no privileged persistence if persistence is a content-subspace property;
# alternative: persistence is generic stream linearity — report which).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'patch_relaxation_results.json'
NEVAL = 160; SEQ = 256; REF_LAYERS = [8, 10, 12]; ABL = list(range(6, 15))
TRACK = list(range(6, 18))
ST = {'mode': None, 'U': None, 'srcres': None, 'store': None, 'patch_layer': None,
      'track': None, 'Utrack': None}


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for li, blk in enumerate(m.transformer.h):
        x, v1 = blk(x, v1, x0)
        if ST['mode'] == 'cap' and li in ABL:
            ST['store'][li] = x.detach()
        elif ST['mode'] == 'patch' and li == ST['patch_layer']:
            U = ST['U']; xs = ST['srcres'][li]
            x = x - (x @ U) @ U.T + (xs @ U) @ U.T
        elif ST['mode'] == 'patch_all' and li in ABL:
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
    blocks = rows[:, :SEQ].contiguous()

    caps, tok = capture_dev(blocks)
    V = int(m.lm_head.weight.shape[0])
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

    # conditions: (name, patch mode, patch layer(s), subspace)
    CONDS = [('p6', 'patch', 6, U256), ('p10', 'patch', 10, U256), ('p14', 'patch', 14, U256),
             ('r10', 'patch', 10, R256), ('full9', 'patch_all', None, U256)]
    n = blocks.shape[0] // 2; src = blocks[:n].contiguous(); tgt = blocks[n:2 * n].contiguous()
    Ssum = {c: {L: 0.0 for L in TRACK} for c, _, _, _ in CONDS}; npos = 0
    for i in range(0, n, 8):
        si = src[i:i+8].to(DEV)[:, :-1].contiguous(); ti = tgt[i:i+8].to(DEV)[:, :-1].contiguous()
        if si.shape[0] != ti.shape[0]: continue
        # source run: capture residuals for patching AND its own tracked coords
        ST['Utrack'] = U256; ST['track'] = {}
        ST['mode'] = 'cap'; ST['store'] = {}; fwd(si)
        srcres = {li: ST['store'][li] for li in ABL}; csrc = ST['track']
        ST['mode'] = None; ST['track'] = {}; fwd(ti); cbase = ST['track']
        for cname, mode, pl, U in CONDS:
            ST['mode'] = mode; ST['U'] = U; ST['srcres'] = srcres; ST['patch_layer'] = pl
            ST['track'] = {}; fwd(ti); cp = ST['track']; ST['mode'] = None
            for L in TRACK:
                if pl is not None and L < pl: continue
                d_p = (cp[L] - cbase[L]).reshape(-1, 256)
                d_s = (csrc[L] - cbase[L]).reshape(-1, 256)
                Ssum[cname][L] += float(F.cosine_similarity(d_p, d_s, dim=-1).sum())
        ST['track'] = None
        npos += si.shape[0] * si.shape[1]

    S = {c: {str(L): round(v / npos, 4) for L, v in d.items() if v != 0.0} for c, d in Ssum.items()}
    s6 = S['p6']; s10 = S['p10']
    # per-block survival ratios at overlapping depths (L11..14 for both p6 and p10)
    ratios6 = {str(L): round(s6[str(L + 1)] / max(s6[str(L)], 1e-6), 3) for L in range(11, 14)}
    ratios10 = {str(L): round(s10[str(L + 1)] / max(s10[str(L)], 1e-6), 3) for L in range(11, 14)}
    rate_gap = max(abs(ratios6[k] - ratios10[k]) for k in ratios6)
    s17_p14 = S['p14']['17']; s17_full = S['full9']['17']
    recon = s17_p14 / max(s17_full, 1e-6)
    out = {'n_positions': npos, 'S_curves': S,
           'survival_ratios_L11_14': {'p6': ratios6, 'p10': ratios10, 'max_gap': round(rate_gap, 3)},
           'reconcile': {'S17_p14_over_full9': round(recon, 4), 'observed_excess_ratio_1154': 0.50},
           'pred_a_partial_relaxation': bool(s6['7'] >= 0.6 and s6['14'] >= 0.3),
           'pred_b_rate_depth_uniform': bool(rate_gap <= 0.1),
           'pred_c_reconciles_1154': bool(abs(recon - 0.50) <= 0.15),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    for c in S:
        print(f"{c:>6}: " + " ".join(f"L{L}:{S[c][L]:+.3f}" for L in sorted(S[c], key=int)), flush=True)
    print(f"survival ratios p6 {ratios6} | p10 {ratios10} | gap {rate_gap}")
    print(f"S17(p14)/S17(full9) = {out['reconcile']['S17_p14_over_full9']} vs excess ratio 0.50")
    print(f"pred_a partial {out['pred_a_partial_relaxation']} | pred_b uniform {out['pred_b_rate_depth_uniform']} | pred_c reconciles {out['pred_c_reconciles_1154']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
