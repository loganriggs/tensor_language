# late_leak: is the cross-position transport leak pooled by the READOUT BAND's attention?
#
# §1159: freezing ALL attention in the patch band (L7-14) barely dents the leak (unpatched
# positions 0.3057 vs 0.3505). The frozen-clamp law (§1155) means patched coords persist to
# L15-17 — where attention was left live. Hypothesis: the leak is pooled late, by the
# readout band's attention, not inside the deep band at all.
#
# Same path-freezing harness as leak_carrier.py (freeze = replace attention outputs with
# unpatched-base-run captures; freeze-identity null exact 0). Scat50 patching (K=256,
# L6-14, fresh rows), attention frozen over different bands:
#   scat50            — reference (patched 0.891 / unpatched 0.351)
#   frz7_14           — §1159 replication (unpatched ≈ 0.306)
#   frz15_17          — ONLY readout-band attention frozen
#   frz7_17           — all attention after the first patch frozen
#   full9 / r256      — anchors
#
# Registered predictions:
#   pred_a LATE BAND IS THE MAIN CARRIER: unpatched(frz15_17) < 0.5 × unpatched(scat50) AND
#          unpatched(frz7_17) < 0.15.
#   pred_b OWN-POSITION TERM IS ATTENTION-FREE: patched(frz7_17) >= 0.8 × patched(scat50) —
#          a position's own reading of its own coords needs no attention at all after L6.
#   pred_c BAND SHARES ADD: unpatched(scat50) − unpatched(frz7_17) ≈ [unpatched(scat50) −
#          unpatched(frz7_14)] + [unpatched(scat50) − unpatched(frz15_17)] within 0.05
#          (deep-band and late-band leak contributions roughly additive).
# Null/controls: r256; freeze-identity re-checked (frz on unpatched run, KL must be 0).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'late_leak_results.json'
NEVAL = 160; SEQ = 256; REF_LAYERS = [8, 10, 12]; ABL = list(range(6, 15))
ALLFRZ = list(range(7, 18))
ST = {'mode': None, 'U': None, 'srcres': None, 'store': None, 'mask': None,
      'freeze_band': None, 'frozen_out': None, 'capture_out': None}


def mk_hook(li):
    def h(mo, i_, o_):
        out = o_[0] if isinstance(o_, tuple) else o_
        if ST['capture_out'] is not None:
            ST['capture_out'][li] = out.detach()
            return None
        fb = ST['freeze_band']
        if fb is not None and li in fb:
            rep = ST['frozen_out'][li]
            return (rep,) + tuple(o_[1:]) if isinstance(o_, tuple) else rep
        return None
    return h

HOOKS = [m.transformer.h[li].attn.register_forward_hook(mk_hook(li)) for li in ALLFRZ]


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for li, blk in enumerate(m.transformer.h):
        x, v1 = blk(x, v1, x0)
        if ST['mode'] == 'cap' and li in ABL:
            ST['store'][li] = x.detach()
        elif ST['mode'] == 'patch' and li in ABL:
            U = ST['U']; xs = ST['srcres'][li]
            xn = x - (x @ U) @ U.T + (xs @ U) @ U.T
            x = torch.where(ST['mask'], xn, x) if ST['mask'] is not None else xn
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

    B714 = set(range(7, 15)); B1517 = {15, 16, 17}; B717 = set(range(7, 18))
    CONDS = [('scat50', U256, 'scat', None), ('frz7_14', U256, 'scat', B714),
             ('frz15_17', U256, 'scat', B1517), ('frz7_17', U256, 'scat', B717),
             ('full9', U256, 'full', None), ('r256', R256, 'full', None)]
    SCATS = [c for c, _, mt, _ in CONDS if mt == 'scat']
    n = blocks.shape[0] // 2; src = blocks[:n].contiguous(); tgt = blocks[n:2 * n].contiguous()
    acc = {c: {'kl': 0.0, 'al': 0.0} for c, _, _, _ in CONDS}
    pp = {c: [0.0, 0] for c in SCATS}; uu = {c: [0.0, 0] for c in SCATS}
    freeze_null_kl = -1.0
    npos = 0
    gp = torch.Generator(device=DEV).manual_seed(1)
    for i in range(0, n, 8):
        si = src[i:i+8].to(DEV)[:, :-1].contiguous(); ti = tgt[i:i+8].to(DEV)[:, :-1].contiguous()
        if si.shape[0] != ti.shape[0]: continue
        ST['mode'] = 'cap'; ST['store'] = {}; ls = fwd(si).float(); ST['mode'] = None
        srcres = {li: ST['store'][li] for li in ABL}
        ST['capture_out'] = {}; lb = fwd(ti).float()
        frozen = ST['capture_out']; ST['capture_out'] = None
        base = F.log_softmax(lb, -1)
        if freeze_null_kl < 0:
            ST['freeze_band'] = B717; ST['frozen_out'] = frozen
            l0 = fwd(ti).float(); ST['freeze_band'] = None; ST['frozen_out'] = None
            p0 = F.log_softmax(l0, -1)
            freeze_null_kl = float((p0.exp() * (p0 - base)).sum(-1).mean())
        B, T = ti.shape
        scat = torch.zeros(B, T, dtype=torch.bool, device=DEV)
        for b in range(B):
            perm = torch.randperm(T, generator=gp, device=DEV)
            scat[b, perm[:T // 2]] = True
        for cname, U, mtype, band in CONDS:
            ST['mode'] = 'patch'; ST['U'] = U; ST['srcres'] = srcres
            ST['mask'] = scat.unsqueeze(-1) if mtype == 'scat' else None
            ST['freeze_band'] = band; ST['frozen_out'] = frozen if band else None
            lp = fwd(ti).float()
            ST['mode'] = None; ST['mask'] = None; ST['freeze_band'] = None; ST['frozen_out'] = None
            patch = F.log_softmax(lp, -1)
            kl = (patch.exp() * (patch - base)).sum(-1)
            cos = F.cosine_similarity((lp - lb), (ls - lb), dim=-1)
            acc[cname]['kl'] += float(kl.sum()); acc[cname]['al'] += float(cos.sum())
            if cname in pp:
                pp[cname][0] += float(cos[scat].sum()); pp[cname][1] += int(scat.sum())
                uu[cname][0] += float(cos[~scat].sum()); uu[cname][1] += int((~scat).sum())
        npos += B * T

    res = {c: {'kl': round(a['kl']/npos, 4), 'alignment': round(a['al']/npos, 4)}
           for c, a in acc.items()}
    P = {c: round(v[0] / max(v[1], 1), 4) for c, v in pp.items()}
    Uu = {c: round(v[0] / max(v[1], 1), 4) for c, v in uu.items()}
    ref_u = Uu['scat50']
    drop_deep = ref_u - Uu['frz7_14']; drop_late = ref_u - Uu['frz15_17']; drop_all = ref_u - Uu['frz7_17']
    out = {'n_positions': npos, 'conds': res, 'freeze_null_kl': round(freeze_null_kl, 6),
           'scat50_patched': P, 'scat50_unpatched': Uu,
           'leak_drops': {'deep_band': round(drop_deep, 4), 'late_band': round(drop_late, 4),
                          'all_bands': round(drop_all, 4)},
           'pred_a_late_main_carrier': bool(Uu['frz15_17'] < 0.5 * ref_u and Uu['frz7_17'] < 0.15),
           'pred_b_own_term_attention_free': bool(P['frz7_17'] >= 0.8 * P['scat50']),
           'pred_c_band_shares_add': bool(abs(drop_all - (drop_deep + drop_late)) <= 0.05),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    for c, _, _, _ in CONDS:
        print(f"{c:>9}: KL {res[c]['kl']:7.3f} | align {res[c]['alignment']:+.4f}", flush=True)
    print(f"patched   {P}")
    print(f"unpatched {Uu} | drops deep {out['leak_drops']['deep_band']} late {out['leak_drops']['late_band']} all {out['leak_drops']['all_bands']}")
    print(f"freeze-null KL {out['freeze_null_kl']}")
    print(f"pred_a late-carrier {out['pred_a_late_main_carrier']} | pred_b own-attn-free {out['pred_b_own_term_attention_free']} | pred_c additive {out['pred_c_band_shares_add']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
