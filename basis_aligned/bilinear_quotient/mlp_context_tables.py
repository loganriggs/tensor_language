# mlp_context_tables: WHAT IS THE UN-TABLEABLE HALF AT THE TOP OF THE MODEL?
#
# §1325/§1326 left the ladder with one sharp hole. mlp16 and mlp17 both cap out at a token-
# table ceiling of ~0.50: half of what they do is unreachable by ANY function of the current
# token. The program's independent guess for that half is coarse contextual state — document
# register (results/13), the kind of thing a 4-16-way label could carry. This prices that
# guess with the ladder's own instrument by widening the stand-in's KEY from f(token) to
# f(token-class, context-class).
#
# ARMS (per module, all scored as held-out CE recovery against the same mean-ablation stake):
#   mean        the stake denominator
#   tok50k      the §1326 ceiling (per-token mean table)
#   tok16       token classes only, K=16 k-means on the token table
#   ctx16       context classes only: C=16 k-means on the module's INPUT at each position,
#               assigned live at eval by nearest centroid
#   tok16xctx16 the product table -- 256 cells
#   tok16xrand16 THE MATCHED NULL -- 256 cells, but the second label is an information-free
#               random draw with the same marginal, fixed per (row, position) so fit and eval
#               agree. Without this arm any gain from tok16xctx16 could be pure cell count.
#
# Registered predictions:
#   pred_a CONTEXT CARRIES REAL INFORMATION AT THE TOP: for mlp17, tok16xctx16 recovery
#          exceeds tok16xrand16 by >= 0.10.
#   pred_b 256 CONTEXT-AWARE CELLS BEAT THE 50k TOKEN TABLE: mlp17 tok16xctx16 >= 0.60
#          (against the token-table ceiling of 0.497). If the un-tableable half is coarse
#          contextual state, a 16x16 key should reach past a 50,257-entry one.
#   pred_c THE FRONT CONTROL HOLDS: at mlp1 the same context increment
#          (tok16xctx16 - tok16xrand16) is < 0.05 -- a token-resolved module (ceiling 0.945)
#          has nothing contextual left to buy, so a positive result there would indict the
#          instrument rather than reveal context.
#
# ASSUMPTION REGISTERED, NOT ASKED: the context label is taken from the module's own input
# (the residual stream it reads). At layer 17 that stream also encodes the current token, so
# a positive pred_a is evidence that the LABEL is informative, not yet that the information
# is non-token. `ctx_token_purity` is reported for exactly this reason -- the fraction of
# each context class explained by its most common token. High purity would mean the context
# arm is smuggling token identity, and the result would have to be read down accordingly.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp_context_tables_results.json'
NFIT = 1920; NR = 480; V = 50257
KTOK = 16; KCTX = 16
LAYERS = (17, 16, 1)
H = m.transformer.h
CUR = {'toks': None, 'mode': None, 'mean': None, 'tab': None,
       'tokcls': None, 'ctxC': None, 'rand': None, 'joint': None}


def mlp_hook(mod, args, out):
    """Stand-in installer. CUR['mode'] selects the arm; the context label is
    computed live from the module's own input (args[0])."""
    mo = CUR['mode']
    if mo is None:
        return out
    if mo == 'mean':
        return CUR['mean'].to(out.dtype).expand_as(out)
    toks = CUR['toks']
    if mo == 'tab':                      # any pure token-keyed table
        return CUR['tab'][toks].to(out.dtype)
    x = args[0]
    B2, T2, _ = x.shape
    if mo == 'ctx':
        c = torch.cdist(x.reshape(-1, D).float(), CUR['ctxC']).argmin(1)
        return CUR['joint'][c].reshape(B2, T2, D).to(out.dtype)
    if mo == 'tokctx':
        c = torch.cdist(x.reshape(-1, D).float(), CUR['ctxC']).argmin(1)
        k = CUR['tokcls'][toks].reshape(-1)
        return CUR['joint'][k * KCTX + c].reshape(B2, T2, D).to(out.dtype)
    if mo == 'tokrand':
        c = CUR['rand'].reshape(-1)
        k = CUR['tokcls'][toks].reshape(-1)
        return CUR['joint'][k * KCTX + c].reshape(B2, T2, D).to(out.dtype)
    raise ValueError(mo)


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
def one_layer(LI, FITR, EVR, RFIT, REVAL):
    hk = H[LI].mlp.register_forward_hook(mlp_hook)
    caps = {}
    def cap_hook(mod, args, out):
        caps['in'] = args[0].detach().float(); caps['out'] = out.detach().float()
        return out
    hk2 = H[LI].mlp.register_forward_hook(cap_hook)

    # ---- pass 1: token table + a subsample of inputs for the context clustering
    sums = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
    sub = []
    CUR['mode'] = None
    for i in range(0, NFIT, 8):
        idx = FITR[i:i + 8, :-1].to(DEV).contiguous()
        fwd(idx)
        ft = idx.reshape(-1)
        sums.index_add_(0, ft, caps['out'].reshape(-1, D))
        cnts.index_add_(0, ft, torch.ones_like(ft, dtype=torch.float))
        sub.append(caps['in'].reshape(-1, D)[::16].clone())
    gmean = sums.sum(0) / cnts.sum()
    tab = torch.where(cnts.unsqueeze(1) > 0, sums / cnts.clamp_min(1).unsqueeze(1),
                      gmean.unsqueeze(0))
    seen = cnts > 0; tok_ids = torch.nonzero(seen).squeeze(1)

    a, _ = kmeans(tab[seen], KTOK, cnts[seen])
    tokcls = torch.zeros(V, dtype=torch.long, device=DEV); tokcls[tok_ids] = a
    _, ctxC = kmeans(torch.cat(sub), KCTX, seed=17)
    del sub

    # ---- pass 2: joint tables, one accumulation per key scheme
    Sctx = torch.zeros(KCTX, D, device=DEV);          Nctx = torch.zeros(KCTX, device=DEV)
    Stc = torch.zeros(KTOK * KCTX, D, device=DEV);    Ntc = torch.zeros(KTOK * KCTX, device=DEV)
    Str = torch.zeros(KTOK * KCTX, D, device=DEV);    Ntr = torch.zeros(KTOK * KCTX, device=DEV)
    # ctx/token purity: how much of each context class is explained by its top token
    pur_cnt = torch.zeros(KCTX, V, device=DEV)
    for i in range(0, NFIT, 8):
        idx = FITR[i:i + 8, :-1].to(DEV).contiguous()
        fwd(idx)
        o = caps['out'].reshape(-1, D)
        c = torch.cdist(caps['in'].reshape(-1, D), ctxC).argmin(1)
        k = tokcls[idx].reshape(-1)
        r = RFIT[i:i + 8, :T].to(DEV).reshape(-1)
        Sctx.index_add_(0, c, o); Nctx.index_add_(0, c, torch.ones_like(c, dtype=torch.float))
        kc = k * KCTX + c
        Stc.index_add_(0, kc, o); Ntc.index_add_(0, kc, torch.ones_like(kc, dtype=torch.float))
        kr = k * KCTX + r
        Str.index_add_(0, kr, o); Ntr.index_add_(0, kr, torch.ones_like(kr, dtype=torch.float))
        pur_cnt.index_put_((c, idx.reshape(-1)),
                           torch.ones_like(c, dtype=torch.float), accumulate=True)
    hk2.remove()

    def norm(S, N):
        return torch.where(N.unsqueeze(1) > 0, S / N.clamp_min(1).unsqueeze(1),
                           gmean.unsqueeze(0))
    Tctx, Ttc, Ttr = norm(Sctx, Nctx), norm(Stc, Ntc), norm(Str, Ntr)
    pur = float((pur_cnt.max(1).values.sum() / pur_cnt.sum()))
    del pur_cnt

    def ce_eval(mode, tab_=None, joint=None):
        CUR['mode'] = mode; CUR['tab'] = tab_; CUR['joint'] = joint
        tot = 0.0; n = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            CUR['rand'] = REVAL[i:i + 8, :T].to(DEV)
            lo = fwd(idx).float()
            tot += float(F.cross_entropy(lo[:, 16:].reshape(-1, lo.shape[-1]),
                                         tg[:, 16:].reshape(-1), reduction='sum'))
            n += idx.shape[0] * (T - 16)
        return tot / n

    CUR.update(mean=gmean, ctxC=ctxC, tokcls=tokcls)
    ce = {'full': ce_eval(None), 'mean': ce_eval('mean'),
          'tok50k': ce_eval('tab', tab_=tab)}
    # tok16 as a token-keyed table: every token mapped to its class's frequency-weighted mean
    S16 = torch.zeros(KTOK, D, device=DEV); N16 = torch.zeros(KTOK, device=DEV)
    S16.index_add_(0, tokcls[tok_ids], tab[tok_ids] * cnts[tok_ids].unsqueeze(1))
    N16.index_add_(0, tokcls[tok_ids], cnts[tok_ids])
    Ttok16 = norm(S16, N16)
    ce['tok16'] = ce_eval('tab', tab_=Ttok16[tokcls])
    ce['ctx16'] = ce_eval('ctx', joint=Tctx)
    ce['tok16xctx16'] = ce_eval('tokctx', joint=Ttc)
    ce['tok16xrand16'] = ce_eval('tokrand', joint=Ttr)
    hk.remove()

    stake = ce['mean'] - ce['full']
    rec = {k: round((ce['mean'] - v) / max(stake, 1e-6), 4)
           for k, v in ce.items() if k not in ('full', 'mean')}
    return {'layer': LI, 'ce': {k: round(v, 4) for k, v in ce.items()},
            'stake': round(stake, 4), 'recovery': rec,
            'ctx_increment': round(rec['tok16xctx16'] - rec['tok16xrand16'], 4),
            'ctx_token_purity': round(pur, 4)}


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ROWS = cl.fineweb_rows(NFIT + NR)
    FITR = ROWS[:NFIT, :T + 1].contiguous(); EVR = ROWS[NFIT:, :T + 1].contiguous()
    cl.assert_disjoint(FITR, EVR, label='mlp_context_tables')
    g = torch.Generator().manual_seed(2029)
    RFIT = torch.randint(0, KCTX, (NFIT, T), generator=g)
    REVAL = torch.randint(0, KCTX, (NR, T), generator=g)

    layers = []
    for LI in LAYERS:
        d = one_layer(LI, FITR, EVR, RFIT, REVAL)
        layers.append(d)
        r = d['recovery']
        print(f"mlp{LI}: stake {d['stake']:.3f} | tok50k {r['tok50k']:.3f} "
              f"tok16 {r['tok16']:.3f} ctx16 {r['ctx16']:.3f} "
              f"tok16xctx16 {r['tok16xctx16']:.3f} null {r['tok16xrand16']:.3f} "
              f"| incr {d['ctx_increment']:+.3f} | purity {d['ctx_token_purity']:.3f}",
              flush=True)
        json.dump({'layers': layers, 'partial': True}, open(OUT, 'w'), indent=1)

    by = {d['layer']: d for d in layers}
    pa = by[17]['ctx_increment'] >= 0.10
    pb = by[17]['recovery']['tok16xctx16'] >= 0.60
    pc = by[1]['ctx_increment'] < 0.05
    out = {'n_fit': NFIT, 'n_eval': NR, 'k_tok': KTOK, 'k_ctx': KCTX, 'layers': layers,
           'pred_a_context_real_at_top': bool(pa), 'pred_b_256_beats_50k': bool(pb),
           'pred_c_front_control': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\npred_a context-real {pa} (incr {by[17]['ctx_increment']:+.3f})")
    print(f"pred_b 256-beats-50k {pb} ({by[17]['recovery']['tok16xctx16']:.3f} "
          f"vs ceiling {by[17]['recovery']['tok50k']:.3f})")
    print(f"pred_c front-control {pc} (mlp1 incr {by[1]['ctx_increment']:+.3f})")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
