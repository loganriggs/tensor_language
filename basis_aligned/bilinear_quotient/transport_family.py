# transport_family: does the transport law (§1150-1157) hold in the sibling model?
#
# The bilin18 result: topic patching works (0.90 alignment), is position-local (a patched
# position's own prediction moves fully; neighbors only get a fading leak), and the address
# is nearly frozen once written (single-layer patch = ~90% of the final coordinate state,
# raw-KL share 73%). Content info is family-universal (CCA 0.95-0.97 §1061; API atoms 7/8
# §1119; generation phenotypes §1137). Registered test: the TRANSPORT LAW is too.
#
# Model: swiglu18 (independently trained, same depth/width, gated-SwiGLU MLPs instead of
# bilinear). Same harness translated: content basis = top-256 PCA of pooled L8-12 MLP-input
# per-token deviations; patch residual after blocks L6-14; alignment + KL readout; fresh rows.
# Conditions: full9, last1 (L14 only), scat50 (random half positions, with per-position
# patched/unpatched split), r256 (random-subspace null).
#
# Registered predictions:
#   pred_a TRANSPORT REPLICATES: align(full9) > align(r256) + 0.10 and KL(full9) > 2×KL(r256).
#   pred_b POSITION-LOCAL REPLICATES: in scat50, patched-position alignment >= 0.85 ×
#          align(full9) while unpatched positions sit below 0.6 × align(full9).
#   pred_c FROZEN CLAMP REPLICATES: raw-KL share KL(last1)/KL(full9) >= 0.55 (bilin18: 0.73).
# Null: r256. If pred_a fails, the law is bilin18-specific — report plainly.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs
import census_lib as cl

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'transport_family_results.json'
NEVAL = 160; SEQ = 256; REF_LAYERS = [8, 10, 12]; ABL = list(range(6, 15))
DEV = 'cuda'
mdl, cfg = load_elriggs('swiglu18'); mdl = mdl.to(DEV).eval()
D = cfg['n_embd']
ST = {'mode': None, 'U': None, 'srcres': None, 'store': None, 'layers': None, 'mask': None}


def fwd(idx):
    x = F.rms_norm(mdl.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for li, blk in enumerate(mdl.transformer.h):
        x, v1 = blk(x, v1, x0)
        if ST['mode'] == 'cap' and li in ABL:
            ST['store'][li] = x.detach()
        elif ST['mode'] == 'patch' and li in ST['layers']:
            U = ST['U']; xs = ST['srcres'][li]
            xn = x - (x @ U) @ U.T + (xs @ U) @ U.T
            x = torch.where(ST['mask'], xn, x) if ST['mask'] is not None else xn
    return 30.0*torch.tanh(mdl.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def capture_dev(blocks):
    caps = {L: [] for L in REF_LAYERS}; toks = []; hs = []
    for L in REF_LAYERS:
        mlp = mdl.transformer.h[L].mlp
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
    blocks = rows[:, :SEQ].contiguous(); V = int(mdl.lm_head.weight.shape[0])

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

    L69 = set(range(6, 15))
    CONDS = [('full9', L69, U256, 'full'), ('last1', {14}, U256, 'full'),
             ('scat50', L69, U256, 'scat'), ('r256', L69, R256, 'full')]
    n = blocks.shape[0] // 2; src = blocks[:n].contiguous(); tgt = blocks[n:2 * n].contiguous()
    acc = {c: {'kl': 0.0, 'al': 0.0} for c, _, _, _ in CONDS}
    scat_p = scat_u = 0.0; scat_pn = scat_un = 0
    npos = 0
    gp = torch.Generator(device=DEV).manual_seed(1)
    for i in range(0, n, 8):
        si = src[i:i+8].to(DEV)[:, :-1].contiguous(); ti = tgt[i:i+8].to(DEV)[:, :-1].contiguous()
        if si.shape[0] != ti.shape[0]: continue
        ST['mode'] = 'cap'; ST['store'] = {}; ls = fwd(si).float(); ST['mode'] = None
        srcres = {li: ST['store'][li] for li in ABL}
        lb = fwd(ti).float(); base = F.log_softmax(lb, -1)
        B, T = ti.shape
        scat = torch.zeros(B, T, dtype=torch.bool, device=DEV)
        for b in range(B):
            perm = torch.randperm(T, generator=gp, device=DEV)
            scat[b, perm[:T // 2]] = True
        for cname, layers, U, mtype in CONDS:
            ST['mode'] = 'patch'; ST['U'] = U; ST['srcres'] = srcres; ST['layers'] = layers
            ST['mask'] = scat.unsqueeze(-1) if mtype == 'scat' else None
            lp = fwd(ti).float(); ST['mode'] = None; ST['mask'] = None
            patch = F.log_softmax(lp, -1)
            kl = (patch.exp() * (patch - base)).sum(-1)
            cos = F.cosine_similarity((lp - lb), (ls - lb), dim=-1)
            acc[cname]['kl'] += float(kl.sum()); acc[cname]['al'] += float(cos.sum())
            if cname == 'scat50':
                scat_p += float(cos[scat].sum()); scat_pn += int(scat.sum())
                scat_u += float(cos[~scat].sum()); scat_un += int((~scat).sum())
        npos += B * T

    res = {c: {'kl': round(a['kl']/npos, 4), 'alignment': round(a['al']/npos, 4)}
           for c, a in acc.items()}
    al = {k: v['alignment'] for k, v in res.items()}
    own = scat_p / max(scat_pn, 1); other = scat_u / max(scat_un, 1)
    klshare = res['last1']['kl'] / max(res['full9']['kl'], 1e-6)
    out = {'model': 'swiglu18', 'n_positions': npos, 'conds': res,
           'scat50_patched_pos': round(own, 4), 'scat50_unpatched_pos': round(other, 4),
           'last1_kl_share': round(klshare, 4),
           'bilin18_refs': {'full9': 0.8994, 'r256': 0.6693, 'patched_pos': 0.8914,
                            'unpatched_pos': 0.3579, 'last1_kl_share': 0.73},
           'pred_a_transport_replicates': bool(al['full9'] > al['r256'] + 0.10 and
                                               res['full9']['kl'] > 2 * res['r256']['kl']),
           'pred_b_position_local': bool(own >= 0.85 * al['full9'] and other < 0.6 * al['full9']),
           'pred_c_frozen_clamp': bool(klshare >= 0.55),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    for c, _, _, _ in CONDS:
        print(f"{c:>7}: KL {res[c]['kl']:7.3f} | align {res[c]['alignment']:+.4f}", flush=True)
    print(f"scat50 patched {out['scat50_patched_pos']} vs unpatched {out['scat50_unpatched_pos']} | last1 KL share {out['last1_kl_share']}")
    print(f"pred_a transport {out['pred_a_transport_replicates']} | pred_b local {out['pred_b_position_local']} | pred_c frozen {out['pred_c_frozen_clamp']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
