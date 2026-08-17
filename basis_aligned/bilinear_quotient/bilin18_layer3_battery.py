"""Phase C: layer 3 -- with layers 2-3 jointly carrying 64% of the middle's fair
share (§27) and layer 4 reading their output (§28), these are the highest-value
unexplored targets. Same battery as layers 0/16, at the same depth. Generated from bilin18_layer0_battery.py with LI=16 and
the writer tracker swapped: at depth 16 the exact writers are grouped coarsely as
  E     the embedding path (with every block's lambda re-injection)
  ATT   the sum of all 17 attention outputs so far (incl. block 16's)
  MLP   the sum of all 16 previous MLP outputs
tracked exactly through the recurrence (components scale by lambda0 each block, E
re-injects lambda1*x0, writes add to their group), then divided by the rms scalar.
Head split is over block 16's own attention.

Layer 0 is the second-most important MLP (delete cost 1.80 nats, §9) and the shallowest:
its input is rmsnorm(embedding + attn0 out), so there are only THREE writer pairs and
the embedding fold-in is nearly the whole story by construction. The §16 lesson is
baked in from the start: the basis is fit on the large corpus (153,900 positions,
rows disjoint from evaluation), and every "share of layer" number is reported as
row-group-relative.

Stages (each gated, results in one JSON):
  B1 Shapley over the top-32 output directions, 20 permutations. Concentration verdict.
  B2 exact writer folding (emb, attn0) for the top-3 Shapley leaders.
  B3 data structure: spectrum, coefficient kurtosis, document ICC, hierarchy probe.
  B4 naming: emb-only curvature over the full vocab (weights-only), measured excitation
     with permutation nulls; head-pair unfold of the attn0 x attn0 term if it matters.
  B5 causal MDL ladder for the leader: delete / story surrogate a(u.x)^2+b / rank-2
     whitened, plus the random-direction control. Fit on fit rows only.
"""

import json
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
import tiktoken
from bilin18_joint_removal import fwd, held, orth, m, FW, DEV, PATCH, B0, HELD
from bilin18_identifiable import form_for_direction
from tier2_model import rope_tables, apply_rot

LI = 3
NDIR = 32
N_PERM = 20
NH, HD, D = 9, 128, 1152
enc = tiktoken.get_encoding('gpt2')
OUT = ('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
       'bilin18_layer3_battery_results.json')


@torch.no_grad()
def collect0(seqs):
    accs = []
    for i in range(0, seqs.shape[0], 6):
        acc = []
        fwd(seqs[i:i + 6].to(DEV), collect=LI, acc=acc)
        accs.append(acc[0])
    return torch.cat(accs, 0)


def value0(Q, cols, Ybar, base):
    if len(cols) == 0:
        return 0.0
    Qs = Q[:, cols]
    PATCH[LI] = (Qs, Ybar @ Qs)
    try:
        return float((held() - base).mean())
    finally:
        PATCH.pop(LI)


@torch.no_grad()
def tracked0(idx):
    """components of layer-16's MLP input, coarse writers (E / all-attn / all-mlp),
    tracked exactly through the recurrence."""
    B, T = idx.shape
    x0 = F.rms_norm(m.transformer.wte(idx), (D,))
    E = x0.clone(); Aw = torch.zeros_like(x0); Mw = torch.zeros_like(x0)
    x = x0; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(LI + 1):
        blk = m.transformer.h[li]
        E = blk.lambdas[0] * E + blk.lambdas[1] * x0
        Aw = blk.lambdas[0] * Aw; Mw = blk.lambdas[0] * Mw
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        hcur = F.rms_norm(x, (D,))

        def qk(l):
            z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,))
            return apply_rot(z, cosb, sinb)

        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        q, k1_, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k1_) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        ctx = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        ao = a.c_proj(ctx.reshape(B, T, -1))
        Aw = Aw + ao; x = x + ao
        if li == LI:
            ctx16 = ctx
            break
        xhat = F.rms_norm(x, (D,)); mlp = blk.mlp
        mo = mlp.Down(mlp.Left(xhat) * mlp.Right(xhat)) + mlp.Down_bias
        Mw = Mw + mo; x = x + mo
    r = x.norm(dim=-1, keepdim=True) / D ** 0.5
    xhat = F.rms_norm(x, (D,))
    Wp = m.transformer.h[LI].attn.c_proj.weight.detach()
    A0h = torch.einsum('bqhd,ehd->bqeh', ctx16,
                       Wp.view(D, NH, HD).to(ctx16.dtype)) / r[..., None]
    return ((E / r).reshape(-1, D).float(),
            ((Aw + Mw) / r).reshape(-1, D).float(),
            xhat.reshape(-1, D).float(), A0h.reshape(-1, D, NH).float(),
            (Aw / r).reshape(-1, D).float(), (Mw / r).reshape(-1, D).float())


COEFF_FN = None


@torch.no_grad()
def fwd_hook(idx):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(len(m.transformer.h)):
        blk = m.transformer.h[li]; x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        hcur = F.rms_norm(x, (D,))

        def qk(l):
            z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,))
            return apply_rot(z, cosb, sinb)

        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        q, k1_, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k1_) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
        xhat = F.rms_norm(x, (D,)); mlp = blk.mlp
        mo = mlp.Down(mlp.Left(xhat) * mlp.Right(xhat)) + mlp.Down_bias
        if li == LI and COEFF_FN is not None:
            mo = COEFF_FN(xhat, mo)
        x = x + mo
    logits = 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)
    Vv = logits.shape[-1]
    return F.cross_entropy(logits[:, :-1].reshape(-1, Vv).float(),
                           idx[:, 1:].reshape(-1), reduction='none').view(B, T - 1)


def ce_hooked(hook):
    global COEFF_FN
    COEFF_FN = hook
    try:
        tot = 0.0; n = 0
        for i in range(0, HELD.shape[0], B0):
            ce = fwd_hook(HELD[i:i + B0].to(DEV))
            tot += float(ce.sum()); n += ce.numel()
        return tot / n
    finally:
        COEFF_FN = None


def spearman(a, b):
    ra = a.argsort().argsort().double(); rb = b.argsort().argsort().double()
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return float((ra @ rb) / (ra.norm() * rb.norm()).clamp_min(1e-30))


def main():
    t0 = time.time()
    base = held(); BASE = float(base.mean())
    out = {'base_ce': BASE, 'layer': LI}
    print(f'base CE {BASE:.4f}\n')

    # basis on big corpus, disjoint from eval rows
    Y = collect0(FW[0:300, :513])
    Ybar = Y.mean(0)
    Yc = (Y - Ybar).float()
    _, Sv, Vh = torch.linalg.svd(Yc, full_matrices=False)
    Q = orth(Vh[:NDIR].T)

    # ===== B1: Shapley =====
    import os
    cache = OUT.replace('.json', '_phi.pt')
    print(f'== B1: Shapley over the top-{NDIR} output directions ==')
    v_all = value0(Q, list(range(NDIR)), Ybar, base)
    if os.path.exists(cache):
        phi = torch.load(cache)
        est = phi.mean(1); se = phi.std(1) / phi.shape[1] ** 0.5
        print('  (loaded cached permutations)')
    else:
        g = torch.Generator().manual_seed(0)
        phi = torch.zeros(NDIR, N_PERM, dtype=torch.float64)
        for p in range(N_PERM):
            perm = torch.randperm(NDIR, generator=g).tolist()
            prev = 0.0; cur = []
            for pos, i in enumerate(perm):
                cur.append(i)
                v = v_all if pos == NDIR - 1 else value0(Q, cur, Ybar, base)
                phi[i, p] = v - prev; prev = v
            if (p + 1) % 5 == 0:
                print(f'  permutation {p+1}/{N_PERM}', flush=True)
        est = phi.mean(1); se = phi.std(1) / N_PERM ** 0.5
        torch.save(phi, cache)
    pr = float(est.sum() ** 2 / (est ** 2).sum())
    srt, order = est.sort(descending=True)
    tot_phi = float(est.sum())
    out['b1'] = {'v_all': v_all, 'phi': est.tolist(), 'se': se.tolist(),
                 'participation_ratio': pr, 'leader_share': float(srt[0] / tot_phi),
                 'top4_share': float(srt[:4].sum() / tot_phi),
                 'leaders': [int(i) for i in order[:6]]}
    print(f'  joint effect of the span {v_all:+.4f} | participation ratio {pr:.1f} '
          f'of {NDIR} | leader {100*srt[0]/tot_phi:.0f}% | top-4 '
          f'{100*srt[:4].sum()/tot_phi:.0f}%')
    print(f'  leaders: {[int(i) for i in order[:6]]}\n')

    # ===== B2 + B4: folding, heads, naming for the top 3 leaders =====
    E_l, A_l, X_l, Ah_l, Aw_l, Mw_l = [], [], [], [], [], []
    for i in range(0, 96, 6):
        E_, A_, X_, Ah_, Awx, Mwx = tracked0(FW[i:i + 6, :513].to(DEV))
        E_l.append(E_); A_l.append(A_); X_l.append(X_); Ah_l.append(Ah_)
        Aw_l.append(Awx); Mw_l.append(Mwx)
    E = torch.cat(E_l); A0 = torch.cat(A_l); Xh = torch.cat(X_l)
    A0h = torch.cat(Ah_l); AwT = torch.cat(Aw_l); MwT = torch.cat(Mw_l)
    gate = float((E + A0 - Xh).norm() / Xh.norm())
    wte = m.transformer.wte.weight.detach().float()
    wte_n = F.rms_norm(wte, (D,))
    mlp0 = m.transformer.h[LI].mlp
    cur_tok = FW[0:96, :513].reshape(-1).to(DEV)
    print(f'== B2/B4: writers and names (decomposition gate {gate:.1e}) ==')
    out['b2'] = []
    for r_ in range(3):
        di = int(order[r_])
        d = Q[:, di].float()
        M = form_for_direction(mlp0, d).float()
        c = torch.einsum('ni,ij,nj->n', Xh, M, Xh)
        var_c = float(c.var())
        rows = []
        for a_, Ta in (('emb', E), ('attn', AwT), ('mlps', MwT)):
            for b_, Tb in (('emb', E), ('attn', AwT), ('mlps', MwT)):
                if a_ > b_:
                    continue
                t = torch.einsum('ni,ij,nj->n', Ta, M, Tb)
                if a_ != b_:
                    t = 2 * t
                cov = float(((t - t.mean()) * (c - c.mean())).mean())
                rows.append((f'{a_}x{b_}', cov / max(var_c, 1e-30)))
        rows.sort(key=lambda x: -abs(x[1]))
        # head split of attn0 x attn0
        u = torch.einsum('nd,de->ne', A0, M)
        Thh = torch.einsum('ndh,nd->nh', A0h, u)
        hshare = []
        z = torch.einsum('ndh,de->neh', A0h, M)
        Tfull = torch.einsum('ndh,ndg->nhg', A0h, z)
        cm = c - c.mean()
        for h in range(NH):
            t = Tfull[:, h, h]
            hshare.append(float(((t - t.mean()) * cm).mean()) / max(var_c, 1e-30))
        # emb-only curvature naming (weights only)
        s_t = torch.einsum('vi,ij,vj->v', wte_n, M, wte_n)
        top = [enc.decode([t]) for t in s_t.argsort(descending=True)[:10].tolist()]
        bot = [enc.decode([t]) for t in s_t.argsort()[:10].tolist()]
        # measured excitation by current token
        a2 = c ** 2
        uniq, cnt = cur_tok.unique(return_counts=True)
        keep = uniq[cnt >= 30]
        exc = torch.stack([a2[cur_tok == t].mean() for t in keep])
        nm = s_t[keep].abs()
        rho = spearman(nm, exc)
        gp = torch.Generator().manual_seed(0)
        null = sorted(abs(spearman(nm[torch.randperm(keep.numel(), generator=gp)],
                                   exc)) for _ in range(100))
        p95 = null[95]
        topx = [enc.decode([int(t)]) for t in keep[exc.argsort(descending=True)[:8]]]
        rec = {'rank': r_ + 1, 'dir': di, 'share': float(est[di] / est.sum()),
               'pairs': rows, 'head_diag_shares': hshare,
               'emb_curv_top': top, 'emb_curv_bot': bot,
               'fires_on': topx, 'rho': rho, 'null_p95': p95}
        out['b2'].append(rec)
        print(f'  leader #{r_+1} (dir {di}, {100*float(est[di]/est.sum()):.0f}% of layer):')
        print(f'    writer pairs: ' + ', '.join(f'{n} {100*v:+.0f}%' for n, v in rows))
        hmax = int(torch.tensor(hshare).argmax())
        print(f'    biggest head-squared term: head {hmax} at {100*hshare[hmax]:+.0f}%')
        print(f'    emb curvature +: {top[:6]}')
        print(f'    emb curvature -: {bot[:6]}')
        print(f'    fires on: {topx}  (rho {rho:+.3f}, null p95 {p95:.3f})',
              flush=True)

    # ===== B3: data structure =====
    C = Yc.T @ Yc / Yc.shape[0]
    ev = torch.linalg.eigvalsh(C.double()).flip(0).clamp_min(0)
    cum = (ev / ev.sum()).cumsum(0)
    er = float(ev.sum() ** 2 / (ev ** 2).sum())
    cpc = Yc @ Vh[:32].T

    def kurt(cx):
        cx = cx - cx.mean(0, keepdim=True)
        return ((cx ** 4).mean(0) / (cx ** 2).mean(0).clamp_min(1e-30) ** 2) - 3

    kp = kurt(cpc)
    n = Yc.shape[0]
    doc = torch.arange(n, device=DEV) // 512
    icc = []
    for j in range(4):
        cx = cpc[:, j]
        dm = torch.zeros(int(doc.max()) + 1, device=DEV).index_add_(0, doc, cx)
        ct = torch.zeros_like(dm).index_add_(0, doc, torch.ones_like(cx))
        icc.append(float((dm[doc] / ct[doc] - cx.mean()).pow(2).mean() /
                         max(float(cx.var()), 1e-30)))
    out['b3'] = {'effective_rank': er,
                 'dims_50_90': [int((cum < .5).sum()) + 1, int((cum < .9).sum()) + 1],
                 'kurtosis_top8': [round(float(v), 1) for v in kp[:8]],
                 'icc_top4': [round(v, 2) for v in icc]}
    print(f'\n== B3: structure ==  eff-rank {er:.0f}, dims for 50/90%: '
          f'{out["b3"]["dims_50_90"]}, kurtosis(top8) {out["b3"]["kurtosis_top8"]}, '
          f'ICC(top4) {out["b3"]["icc_top4"]}')

    # ===== B5: causal MDL ladder for the leader =====
    print(f'\n== B5: causal MDL ladder for the layer-0 leader ==')
    d0 = Q[:, int(order[0])].float()
    M0 = form_for_direction(mlp0, d0).float()
    c_fit = torch.einsum('ni,ij,nj->n', Xh, M0, Xh)
    cbar = float(c_fit.mean())
    S = (Xh.T @ Xh / Xh.shape[0]).double()
    evs, Us = torch.linalg.eigh(S)
    kd = evs > 1e-8 * evs.max()
    Sh = (Us[:, kd] * evs[kd].sqrt()) @ Us[:, kd].T
    Sih = (Us[:, kd] * evs[kd].rsqrt()) @ Us[:, kd].T
    Mw = Sh @ M0.double() @ Sh
    ew, Uw = torch.linalg.eigh(Mw)
    iw = ew.abs().argmax()
    u1 = (Sih @ Uw[:, iw]).float(); u1 = u1 / u1.norm()
    p2 = (Xh @ u1) ** 2
    Af = torch.stack([p2, torch.ones_like(p2)], 1)
    co = torch.linalg.lstsq(Af, c_fit[:, None]).solution.squeeze()
    idx2 = ew.abs().argsort(descending=True)[:2]
    M2 = (Sih @ (Uw[:, idx2] * ew[idx2]) @ Uw[:, idx2].T @ Sih).float()
    b2c = float((c_fit - torch.einsum('ni,ij,nj->n', Xh, M2, Xh)).mean())
    gr = torch.Generator(device=DEV).manual_seed(0)
    ur = torch.randn(D, device=DEV, generator=gr); ur = ur / ur.norm()
    pr_ = (Xh @ ur) ** 2
    cr = torch.linalg.lstsq(torch.stack([pr_, torch.ones_like(pr_)], 1),
                            c_fit[:, None]).solution.squeeze()

    def hook_del(xhat, mo):
        c = mo.float() @ d0
        return mo + ((cbar - c)[..., None] * d0).to(mo.dtype)

    def mk_sur(uv, av, bv):
        def hook(xhat, mo):
            c = mo.float() @ d0
            chat = av * (xhat.float() @ uv) ** 2 + bv
            return mo + ((chat - c)[..., None] * d0).to(mo.dtype)
        return hook

    def hook_r2(xhat, mo):
        c = mo.float() @ d0
        xf = xhat.float()
        chat = torch.einsum('...i,ij,...j->...', xf, M2, xf) + b2c
        return mo + ((chat - c)[..., None] * d0).to(mo.dtype)

    ce_del = ce_hooked(hook_del)
    ce_sur = ce_hooked(mk_sur(u1, float(co[0]), float(co[1])))
    ce_rk2 = ce_hooked(hook_r2)
    ce_rnd = ce_hooked(mk_sur(ur, float(cr[0]), float(cr[1])))
    span = max(ce_del - BASE, 1e-9)
    out['b5'] = {'ce_delete': ce_del, 'ce_surrogate': ce_sur, 'ce_rank2': ce_rk2,
                 'ce_random_u': ce_rnd,
                 'repair_surrogate': 1 - (ce_sur - BASE) / span,
                 'repair_rank2': 1 - (ce_rk2 - BASE) / span,
                 'repair_random': 1 - (ce_rnd - BASE) / span}
    print(f'  delete leader        +{ce_del-BASE:.4f}')
    print(f'  a(u.x)^2+b (1,154p)  +{ce_sur-BASE:.4f}  repairs '
          f'{100*out["b5"]["repair_surrogate"]:.1f}%')
    print(f'  rank-2 (2,308p)      +{ce_rk2-BASE:.4f}  repairs '
          f'{100*out["b5"]["repair_rank2"]:.1f}%')
    print(f'  random-u control     +{ce_rnd-BASE:.4f}  repairs '
          f'{100*out["b5"]["repair_random"]:.1f}%')

    out['runtime_s'] = time.time() - t0
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
