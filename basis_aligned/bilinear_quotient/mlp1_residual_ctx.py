# mlp1_residual_ctx: THE DIRECT PROOF (OR REFUTATION) OF WITHIN-TOKEN CONTEXT, WITH NO
# BOUND ARGUMENTS LEFT. §1330 retracted the ceiling gate (the mean table is L2-optimal,
# not CE-optimal) and reinstated mlp1's context increment as genuine on the strength of
# its data-scaling. This script removes the remaining indirection: instead of comparing
# coarse (tok16 x ctx16) tables against nulls, fit context ON TOP of the FULL token
# table and ask whether it adds anything.
#
#   Stand-in family: out_hat = tok50k_table[token] + delta[tokcls16, ctx16]
#   where delta is fit on the RESIDUALS (true output - tok50k_table[token]) over the fit
#   rows, at (16 token-classes x 16 context-classes) grain. If context genuinely selects
#   among per-token output variants, the residual table is nonzero where it matters and
#   recovery EXCEEDS the token-table ceiling. The within-token null (labels resampled
#   from P(ctx|token), §1328 machinery) prices the same construction with token-matched
#   chance labels; its delta should converge to ~0 cellwise.
#
# Registered predictions:
#   pred_a CONTEXT ADDS ON TOP OF THE FULL TABLE at mlp1: rec(tok50k + residual-ctx)
#          >= rec(tok50k) + 0.02.
#   pred_b IT IS THE LABEL, NOT THE CELLS: the real arm beats the null arm by >= 0.015.
#   pred_c THE TOP HAS MORE TO ADD: at mlp17 the same construction clears its ceiling by
#          >= 0.05 (its contextual half is bigger at every grain).
# Diagnostic: L2 variance of residuals explained by the real vs null partition — if CE
# gain is large while L2 explained is tiny, §1330's nonlinear-readout account is
# confirmed in the same run.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp1_residual_ctx_results.json'
NFIT = 3840; NR = 480; V = 50257
KTOK = 16; KCTX = 16
LAYERS = (1, 17)
H = m.transformer.h
CUR = {'toks': None, 'mode': None, 'mean': None, 'tab': None, 'tokcls': None,
       'ctxC': None, 'shuf': None, 'delta': None}


def mlp_hook(mod, args, out):
    mo = CUR['mode']
    if mo is None:
        return out
    if mo == 'mean':
        return CUR['mean'].to(out.dtype).expand_as(out)
    toks = CUR['toks']
    y = CUR['tab'][toks].to(out.dtype)
    if mo == 'tab':
        return y
    B2, T2 = toks.shape
    k = CUR['tokcls'][toks].reshape(-1)
    if mo == 'tab_null':
        c = CUR['shuf'].reshape(-1)
    else:                                  # 'tab_ctx'
        f = args[0].float().reshape(-1, D)
        c = torch.cdist(f, CUR['ctxC']).argmin(1)
    return y + CUR['delta'][k * KCTX + c].reshape(B2, T2, D).to(out.dtype)


@torch.no_grad()
def fwd(idx):
    CUR['toks'] = idx
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


def kmeans(X, K, w=None, iters=25, seed=41):
    g = torch.Generator(device='cpu').manual_seed(seed)
    C = X[torch.randperm(X.shape[0], generator=g)[:K].to(X.device)].clone()
    for _ in range(iters):
        a = torch.cdist(X, C).argmin(1)
        for k in range(K):
            mk = a == k
            if mk.any():
                C[k] = ((X[mk] * w[mk].unsqueeze(1)).sum(0) / w[mk].sum()) if w is not None \
                    else X[mk].mean(0)
    return a, C


@torch.no_grad()
def one_layer(LI, FITR, EVR, gen):
    hk = H[LI].mlp.register_forward_hook(mlp_hook)
    caps = {}
    def cap_hook(mod, args, out):
        caps['in'] = args[0].detach().float(); caps['out'] = out.detach().float()
        return out
    hk2 = H[LI].mlp.register_forward_hook(cap_hook)

    # pass 1: token table + input subsample for ctx centroids
    sums = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
    sub = []
    CUR['mode'] = None
    stride = max(1, (NFIT * T) // 60000)
    for i in range(0, NFIT, 8):
        idx = FITR[i:i + 8, :-1].to(DEV).contiguous()
        fwd(idx)
        ft = idx.reshape(-1)
        sums.index_add_(0, ft, caps['out'].reshape(-1, D))
        cnts.index_add_(0, ft, torch.ones_like(ft, dtype=torch.float))
        sub.append(caps['in'].reshape(-1, D)[::stride].clone())
    gmean = sums.sum(0) / cnts.sum()
    tab = torch.where(cnts.unsqueeze(1) > 0, sums / cnts.clamp_min(1).unsqueeze(1),
                      gmean.unsqueeze(0))
    seen = cnts > 0; tok_ids = torch.nonzero(seen).squeeze(1)
    a, _ = kmeans(tab[seen], KTOK, cnts[seen])
    tokcls = torch.zeros(V, dtype=torch.long, device=DEV); tokcls[tok_ids] = a
    _, ctxC = kmeans(torch.cat(sub), KCTX, seed=17)
    del sub

    # pass 2: residual tables (real + null) + P(ctx|token) + L2 diagnostics
    Sr = torch.zeros(KTOK * KCTX, D, device=DEV); Nr = torch.zeros(KTOK * KCTX, device=DEV)
    JT = torch.zeros(V, KCTX, device=DEV)
    res_sq = 0.0; res_n = 0
    for i in range(0, NFIT, 8):
        idx = FITR[i:i + 8, :-1].to(DEV).contiguous()
        fwd(idx)
        r = (caps['out'] - tab[idx]).reshape(-1, D)
        c = torch.cdist(caps['in'].reshape(-1, D), ctxC).argmin(1)
        kc = tokcls[idx].reshape(-1) * KCTX + c
        Sr.index_add_(0, kc, r); Nr.index_add_(0, kc, torch.ones_like(kc, dtype=torch.float))
        JT.index_put_((idx.reshape(-1), c), torch.ones_like(c, dtype=torch.float),
                      accumulate=True)
        res_sq += float((r * r).sum()); res_n += r.shape[0]
    delta_real = torch.where(Nr.unsqueeze(1) > 0, Sr / Nr.clamp_min(1).unsqueeze(1),
                             torch.zeros(1, D, device=DEV))
    PC = torch.where(JT.sum(1, keepdim=True) > 0, JT / JT.sum(1, keepdim=True).clamp_min(1),
                     torch.full_like(JT, 1.0 / KCTX))

    def shuf_labels(rows_tok):
        flat = rows_tok.reshape(-1)
        return torch.multinomial(PC[flat], 1, generator=gen).squeeze(1).view(rows_tok.shape)

    Sn = torch.zeros(KTOK * KCTX, D, device=DEV); Nn = torch.zeros(KTOK * KCTX, device=DEV)
    for i in range(0, NFIT, 8):
        idx = FITR[i:i + 8, :-1].to(DEV).contiguous()
        fwd(idx)
        r = (caps['out'] - tab[idx]).reshape(-1, D)
        c = shuf_labels(idx).reshape(-1)
        kc = tokcls[idx].reshape(-1) * KCTX + c
        Sn.index_add_(0, kc, r); Nn.index_add_(0, kc, torch.ones_like(kc, dtype=torch.float))
    delta_null = torch.where(Nn.unsqueeze(1) > 0, Sn / Nn.clamp_min(1).unsqueeze(1),
                             torch.zeros(1, D, device=DEV))
    hk2.remove()

    # L2 diagnostic: fraction of residual variance the partitions explain (fit-side)
    l2_real = float((delta_real * delta_real * Nr.unsqueeze(1)).sum() / max(res_sq, 1e-9))
    l2_null = float((delta_null * delta_null * Nn.unsqueeze(1)).sum() / max(res_sq, 1e-9))

    def ce_eval(mode, delta=None):
        CUR.update(mode=mode, delta=delta)
        tot = 0.0; n = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            if mode == 'tab_null':
                CUR['shuf'] = shuf_labels(idx)
            lo = fwd(idx).float()
            tot += float(F.cross_entropy(lo[:, 16:].reshape(-1, lo.shape[-1]),
                                         tg[:, 16:].reshape(-1), reduction='sum'))
            n += idx.shape[0] * (T - 16)
        return tot / n

    CUR.update(mean=gmean, tab=tab, tokcls=tokcls, ctxC=ctxC)
    ce = {'full': ce_eval(None), 'mean': ce_eval('mean'), 'tok50k': ce_eval('tab'),
          'tok50k_ctx': ce_eval('tab_ctx', delta=delta_real),
          'tok50k_null': ce_eval('tab_null', delta=delta_null)}
    hk.remove()
    stake = ce['mean'] - ce['full']
    rec = {k: round((ce['mean'] - v) / max(stake, 1e-6), 4)
           for k, v in ce.items() if k not in ('full', 'mean')}
    return {'layer': LI, 'ce': {k: round(v, 4) for k, v in ce.items()},
            'stake': round(stake, 4), 'recovery': rec,
            'add_ctx': round(rec['tok50k_ctx'] - rec['tok50k'], 4),
            'add_null': round(rec['tok50k_null'] - rec['tok50k'], 4),
            'ctx_over_null': round(rec['tok50k_ctx'] - rec['tok50k_null'], 4),
            'l2_frac_real': round(l2_real, 4), 'l2_frac_null': round(l2_null, 4)}


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ROWS = cl.fineweb_rows(NFIT + NR)
    FITR = ROWS[:NFIT, :T + 1].contiguous(); EVR = ROWS[NFIT:, :T + 1].contiguous()
    cl.assert_disjoint(FITR, EVR, label='mlp1_residual_ctx')
    gen = torch.Generator(device=DEV).manual_seed(4403)

    layers = []
    for LI in LAYERS:
        d = one_layer(LI, FITR, EVR, gen)
        layers.append(d)
        print(f"mlp{LI}: stake {d['stake']:.3f} | tok50k {d['recovery']['tok50k']:.4f} "
              f"+ctx {d['recovery']['tok50k_ctx']:.4f} +null {d['recovery']['tok50k_null']:.4f}"
              f" | add_ctx {d['add_ctx']:+.4f} add_null {d['add_null']:+.4f}"
              f" | L2 real/null {d['l2_frac_real']:.4f}/{d['l2_frac_null']:.4f}", flush=True)
        json.dump({'layers': layers, 'partial': True}, open(OUT, 'w'), indent=1)

    by = {d['layer']: d for d in layers}
    pa = by[1]['add_ctx'] >= 0.02
    pb = by[1]['ctx_over_null'] >= 0.015
    pc = by[17]['add_ctx'] >= 0.05
    out = {'n_fit': NFIT, 'n_eval': NR, 'layers': layers,
           'pred_a_adds_on_top': bool(pa), 'pred_b_beats_null': bool(pb),
           'pred_c_top_adds_more': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\npred_a adds-on-top {pa} ({by[1]['add_ctx']:+.4f}) | "
          f"pred_b beats-null {pb} ({by[1]['ctx_over_null']:+.4f}) | "
          f"pred_c top-more {pc} ({by[17]['add_ctx']:+.4f})")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
