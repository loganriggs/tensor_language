# coverage_curve_transport: is transport per-position-additive-but-diluted, or does it
# require configuration consistency across (nearly) all positions?
#
# §1150: position-shuffled source coords lose all excess. §1151: HALF-coverage patching
# (any position class) has zero net excess — but the global alignment readout cannot
# distinguish "per-position transport diluted by the unpatched half" from "consistency
# required". This experiment separates them with a coverage curve + contiguity contrast.
#
# Harness: identical to content_grain_ladder.py (§1150) — residual patch after blocks
# L6-14, K=256 content basis, position-aligned source coords, alignment+KL vs r256 null,
# fresh rows. Mask variants:
#   scat25 / scat50 / scat75 / scat90  — random scattered positions at coverage p
#   prefix50 / suffix50                — contiguous first/second half of the sequence
#   full                               — p=1.0 reference (expect 0.8994)
#   r256                               — full-position random-subspace null (0.6693)
#
# Registered predictions (excess = alignment − r256):
#   pred_a CONSISTENCY REQUIRED: scattered excess stays ≈0 (<0.05) through p=0.75 and
#          only rises at p=0.90 — i.e., strongly convex, not linear in p.
#          Alternative (per-position additive): excess ≈ p × 0.23, so scat50 ≈ +0.11.
#   pred_b CONTIGUITY HELPS: prefix50 excess > scat50 excess + 0.05 — a contiguous
#          coherent source PREFIX transports (downstream positions read a consistent
#          early topic) even though scattered-half does not.
#   pred_c PREFIX > SUFFIX: prefix50 > suffix50 (suffix-patched positions sit atop an
#          unpatched target prefix that contaminates their reads; causal attention makes
#          early positions the address's root).
# Controls: r256 null; full reference; scat50 replicates §1151's rand_c/rand_f (~0.64).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'coverage_curve_transport_results.json'
NEVAL = 160; SEQ = 256; REF_LAYERS = [8, 10, 12]; ABL = list(range(6, 15))
ST = {'mode': None, 'U': None, 'srcres': None, 'store': None, 'mask': None}


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for li, blk in enumerate(m.transformer.h):
        x, v1 = blk(x, v1, x0)
        if ST['mode'] == 'cap' and li in ABL:
            ST['store'][li] = x.detach()
        elif ST['mode'] == 'patch' and li in ABL:
            U = ST['U']; xs = ST['srcres'][li]
            xn = x - (x @ U) @ U.T + (xs @ U) @ U.T
            x = torch.where(ST['mask'], xn, x)
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
    rows = cl.fineweb_rows(2 * NEVAL)[NEVAL:]              # same fresh half as §1150-51
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

    n = blocks.shape[0] // 2; src = blocks[:n].contiguous(); tgt = blocks[n:2 * n].contiguous()
    CONDS = ['scat25', 'scat50', 'scat75', 'scat90', 'prefix50', 'suffix50', 'full', 'r256']
    acc = {c: {'kl': 0.0, 'al': 0.0} for c in CONDS}; npos = 0
    gp = torch.Generator(device=DEV).manual_seed(1)
    for i in range(0, n, 8):
        si = src[i:i+8].to(DEV)[:, :-1].contiguous(); ti = tgt[i:i+8].to(DEV)[:, :-1].contiguous()
        if si.shape[0] != ti.shape[0]: continue
        ST['mode'] = 'cap'; ST['store'] = {}; ls = fwd(si).float(); ST['mode'] = None
        srcres = {li: ST['store'][li] for li in ABL}
        lb = fwd(ti).float(); base = F.log_softmax(lb, -1)
        B, T = ti.shape; half = T // 2
        MASKS = {}
        for pname, p in (('scat25', .25), ('scat50', .5), ('scat75', .75), ('scat90', .9)):
            mk = torch.zeros(B, T, dtype=torch.bool, device=DEV)
            for b in range(B):
                perm = torch.randperm(T, generator=gp, device=DEV)
                mk[b, perm[:int(p * T)]] = True
            MASKS[pname] = mk
        pre = torch.zeros(B, T, dtype=torch.bool, device=DEV); pre[:, :half] = True
        MASKS['prefix50'] = pre; MASKS['suffix50'] = ~pre
        MASKS['full'] = torch.ones(B, T, dtype=torch.bool, device=DEV); MASKS['r256'] = MASKS['full']
        for cname in CONDS:
            ST['mode'] = 'patch'; ST['U'] = R256 if cname == 'r256' else U256
            ST['srcres'] = srcres; ST['mask'] = MASKS[cname].unsqueeze(-1)
            lp = fwd(ti).float(); ST['mode'] = None
            patch = F.log_softmax(lp, -1)
            kl = (patch.exp() * (patch - base)).sum(-1)
            cos = F.cosine_similarity((lp - lb).reshape(-1, V), (ls - lb).reshape(-1, V), dim=-1)
            acc[cname]['kl'] += float(kl.sum()); acc[cname]['al'] += float(cos.sum())
        npos += B * T

    res = {c: {'kl': round(a['kl']/npos, 4), 'alignment': round(a['al']/npos, 4)}
           for c, a in acc.items()}
    al = {k: v['alignment'] for k, v in res.items()}; r = al['r256']
    exc = {k: round(v - r, 4) for k, v in al.items()}
    out = {'n_positions': npos, 'conds': res, 'excess_over_r256': exc,
           'pred_a_consistency_required': bool(exc['scat75'] < 0.05 and exc['scat50'] < 0.05),
           'pred_b_contiguity_helps': bool(exc['prefix50'] > exc['scat50'] + 0.05),
           'pred_c_prefix_over_suffix': bool(al['prefix50'] > al['suffix50']),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    for c in CONDS:
        print(f"{c:>9}: KL {res[c]['kl']:7.3f} | align {res[c]['alignment']:+.4f} | excess {exc[c]:+.4f}", flush=True)
    print(f"pred_a consistency {out['pred_a_consistency_required']} | pred_b contiguity {out['pred_b_contiguity_helps']} | "
          f"pred_c prefix>suffix {out['pred_c_prefix_over_suffix']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
