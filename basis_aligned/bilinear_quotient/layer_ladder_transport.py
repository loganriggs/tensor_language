# layer_ladder_transport: the DEPTH dimension of the transport arc (§1150-53 covered the
# position dimension; this covers which LAYERS the address is read from).
#
# §1150-53: the topic address is carried per-position and read almost entirely locally.
# All runs so far patch at ALL nine deep blocks (L6-14). Question: where along depth does
# the patched address actually get consumed? If the main consumer is the exit path (the
# skeleton result: deep MLPs are minor content consumers, main reader = readout, §1114-18),
# then patching only the LAST deep blocks should retain most transport (later blocks would
# otherwise re-derive/overwrite earlier patches); patching only EARLY blocks should fade as
# subsequent unpatched blocks rebuild the target's own content on top.
#
# Harness: identical to §1150-53 (K=256, full position coverage, fresh rows, alignment+KL).
# Conditions = which block subset gets patched:
#   early3  — L6-8      mid3 — L9-11      late3 — L12-14
#   early6  — L6-11     late6 — L9-14
#   full9   — L6-14 (reference, 0.8994)
#   last1   — L14 only (the cheapest possible patch)
#   r256    — full-position, full-9-layer random-subspace null (0.6693)
#
# Registered predictions (excess = alignment − r256):
#   pred_a LATE DOMINATES: excess(late3) > 2 × excess(early3) — early patches are largely
#          overwritten by the unpatched blocks after them (construction keeps running),
#          late patches sit closest to the reader.
#   pred_b LAST LAYER IS NOT ENOUGH: excess(last1) < 0.5 × excess(late3) — the address is
#          consumed across several late blocks (and mid-block MLPs read some content,
#          13% §1114-18), not at a single point.
#   pred_c MONOTONE IN DEPTH at fixed count: align(late3) > align(mid3) > align(early3).
# Alternative: if early3 ≈ late3, the address persists once written (value-residual-style
# persistence §1076/§1122) and depth position is irrelevant — report plainly.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'layer_ladder_transport_results.json'
NEVAL = 160; SEQ = 256; REF_LAYERS = [8, 10, 12]; ABL = list(range(6, 15))
ST = {'mode': None, 'U': None, 'srcres': None, 'store': None, 'layers': None}


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for li, blk in enumerate(m.transformer.h):
        x, v1 = blk(x, v1, x0)
        if ST['mode'] == 'cap' and li in ABL:
            ST['store'][li] = x.detach()
        elif ST['mode'] == 'patch' and li in ST['layers']:
            U = ST['U']; xs = ST['srcres'][li]
            x = x - (x @ U) @ U.T + (xs @ U) @ U.T
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

    SUBSETS = {'early3': [6, 7, 8], 'mid3': [9, 10, 11], 'late3': [12, 13, 14],
               'early6': list(range(6, 12)), 'late6': list(range(9, 15)),
               'full9': list(range(6, 15)), 'last1': [14], 'r256': list(range(6, 15))}
    n = blocks.shape[0] // 2; src = blocks[:n].contiguous(); tgt = blocks[n:2 * n].contiguous()
    acc = {c: {'kl': 0.0, 'al': 0.0} for c in SUBSETS}; npos = 0
    for i in range(0, n, 8):
        si = src[i:i+8].to(DEV)[:, :-1].contiguous(); ti = tgt[i:i+8].to(DEV)[:, :-1].contiguous()
        if si.shape[0] != ti.shape[0]: continue
        ST['mode'] = 'cap'; ST['store'] = {}; ls = fwd(si).float(); ST['mode'] = None
        srcres = {li: ST['store'][li] for li in ABL}
        lb = fwd(ti).float(); base = F.log_softmax(lb, -1)
        for cname, layers in SUBSETS.items():
            ST['mode'] = 'patch'; ST['U'] = R256 if cname == 'r256' else U256
            ST['srcres'] = srcres; ST['layers'] = set(layers)
            lp = fwd(ti).float(); ST['mode'] = None
            patch = F.log_softmax(lp, -1)
            kl = (patch.exp() * (patch - base)).sum(-1)
            cos = F.cosine_similarity((lp - lb).reshape(-1, V), (ls - lb).reshape(-1, V), dim=-1)
            acc[cname]['kl'] += float(kl.sum()); acc[cname]['al'] += float(cos.sum())
        npos += si.shape[0] * si.shape[1]

    res = {c: {'kl': round(a['kl']/npos, 4), 'alignment': round(a['al']/npos, 4)}
           for c, a in acc.items()}
    al = {k: v['alignment'] for k, v in res.items()}; r = al['r256']
    exc = {k: round(v - r, 4) for k, v in al.items()}
    out = {'n_positions': npos, 'conds': res, 'excess_over_r256': exc,
           'pred_a_late_dominates': bool(exc['late3'] > 2 * exc['early3']),
           'pred_b_last1_not_enough': bool(exc['last1'] < 0.5 * exc['late3']),
           'pred_c_monotone_depth': bool(al['late3'] > al['mid3'] > al['early3']),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    for c in SUBSETS:
        print(f"{c:>7}: KL {res[c]['kl']:7.3f} | align {res[c]['alignment']:+.4f} | excess {exc[c]:+.4f}", flush=True)
    print(f"pred_a late {out['pred_a_late_dominates']} | pred_b last1 {out['pred_b_last1_not_enough']} | pred_c monotone {out['pred_c_monotone_depth']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
