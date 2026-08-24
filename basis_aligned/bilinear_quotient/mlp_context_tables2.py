# mlp_context_tables2: THE WITHIN-TOKEN NULL — is the context arm measuring CONTEXT, or is it
# measuring TOKEN IDENTITY a second time?
#
# §1327 ran the context arm with an information-free random label as its null. That null did
# its job: 256 cells per se buy nothing (tok16xrand16 reproduced tok16 to within 0.0002). But
# it controls the WRONG confound. A label that correlates with the current token is not
# information-free, and the front control caught exactly that: mlp1 — a module the ladder
# showed to be 94.5% token-static, with nothing contextual left to buy — posted the LARGEST
# increment of all three (+0.260). At layer 1 the residual stream is dominated by the token
# embedding, so k-means on it is a second 16-way partition of token identity, and the "joint"
# table is just a finer token table. §1327's claims are held provisional on this run.
#
# THE CORRECT NULL (the whole point of this script). Resample each position's context label
# from the empirical P(ctx | token) estimated on the fit set, independently per position, at
# BOTH fit and eval. That holds the token->context marginal EXACTLY fixed while destroying
# position-to-position variation. Anything the real label buys over this null is context that
# varies WITHIN a fixed token — which is the only thing that deserves the name.
#
# THREE LABEL SOURCES, each scored against its own within-token null:
#   ctx_self  k-means on the module input at position t  (§1327's label, now correctly nulled)
#   ctx_prev  k-means on the module input at position t-1 (strictly prior context)
#   ctx_doc   k-means on the running mean of module inputs over [0, t) — slow document-level
#             state that CANNOT contain the current token at all. This is the register probe.
# All three share one centroid set per source, fit on the fit rows; assignment is live.
#
# Registered predictions:
#   pred_a THE V1 RESULT WAS A TOKEN ARTIFACT: mlp1's ctx_self increment over the within-token
#          null falls below 0.05 (from +0.260 against the random null). If this holds, §1327's
#          front-control failure is explained and the mlp1 number is retracted as token
#          repartitioning.
#   pred_b THE TOP SURVIVES ANYWAY: mlp17's ctx_self increment over the within-token null is
#          still >= 0.08 (down from +0.193, but real).
#   pred_c DOCUMENT STATE IS THE CARRIER AT THE TOP: mlp17's ctx_doc increment is >= 0.05 AND
#          exceeds mlp1's ctx_doc increment. ctx_doc is token-free by construction, so a
#          positive here is the cleanest evidence the un-tableable half is slow context.
#
# DIAGNOSTIC REPLACING §1327's RETIRED PURITY: normalized mutual information
# I(ctx; token) / H(ctx) on the fit set, per module per source. Purity was powerless (0.13-0.17
# everywhere, including where the confound was near-certain) because no single token can
# dominate a class of tens of thousands of positions. NMI answers the actual question: what
# share of the label is already determined by the token?
#
# ASSUMPTION REGISTERED, NOT ASKED: ctx_prev and ctx_doc use their own k-means centroids fit on
# their own feature (previous-position input; running-mean input), not reused from ctx_self —
# the three are separate label systems, compared only through their increments.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp_context_tables2_results.json'
NFIT = 1920; NR = 480; V = 50257
KTOK = 16; KCTX = 16
SOURCES = ('self', 'prev', 'doc')
LAYERS = (17, 16, 1)
H = m.transformer.h
CUR = {'toks': None, 'mode': None, 'mean': None, 'tab': None, 'tokcls': None,
       'ctxC': None, 'src': None, 'shuf': None, 'joint': None}


def feat(x, src):
    """The context feature at every position, from the module input x (B,T,D)."""
    if src == 'self':
        return x
    if src == 'prev':
        return torch.cat([x[:, :1], x[:, :-1]], 1)
    if src == 'doc':                      # running mean over strictly-prior positions
        cs = x.cumsum(1)
        prior = torch.cat([torch.zeros_like(cs[:, :1]), cs[:, :-1]], 1)
        denom = torch.arange(1, x.shape[1] + 1, device=x.device).clamp_min(1) - 1
        return prior / denom.clamp_min(1).view(1, -1, 1)
    raise ValueError(src)


def mlp_hook(mod, args, out):
    """Stand-in installer. 'shuf' selects the within-token null (labels are then
    supplied precomputed in CUR['shuf'] rather than derived from the input)."""
    mo = CUR['mode']
    if mo is None:
        return out
    if mo == 'mean':
        return CUR['mean'].to(out.dtype).expand_as(out)
    toks = CUR['toks']
    if mo == 'tab':
        return CUR['tab'][toks].to(out.dtype)
    B2, T2 = toks.shape
    k = CUR['tokcls'][toks].reshape(-1)
    if mo == 'tokshuf':
        c = CUR['shuf'].reshape(-1)
    else:                                  # 'tokctx'
        f = feat(args[0].float(), CUR['src']).reshape(-1, D)
        c = torch.cdist(f, CUR['ctxC']).argmin(1)
    return CUR['joint'][k * KCTX + c].reshape(B2, T2, D).to(out.dtype)


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

    # ---- pass 1: token table + feature subsamples for each context source
    sums = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
    sub = {s: [] for s in SOURCES}
    CUR['mode'] = None
    for i in range(0, NFIT, 8):
        idx = FITR[i:i + 8, :-1].to(DEV).contiguous()
        fwd(idx)
        ft = idx.reshape(-1)
        sums.index_add_(0, ft, caps['out'].reshape(-1, D))
        cnts.index_add_(0, ft, torch.ones_like(ft, dtype=torch.float))
        for s in SOURCES:
            sub[s].append(feat(caps['in'], s).reshape(-1, D)[::16].clone())
    gmean = sums.sum(0) / cnts.sum()
    tab = torch.where(cnts.unsqueeze(1) > 0, sums / cnts.clamp_min(1).unsqueeze(1),
                      gmean.unsqueeze(0))
    seen = cnts > 0; tok_ids = torch.nonzero(seen).squeeze(1)
    a, _ = kmeans(tab[seen], KTOK, cnts[seen])
    tokcls = torch.zeros(V, dtype=torch.long, device=DEV); tokcls[tok_ids] = a
    ctxC = {s: kmeans(torch.cat(sub[s]), KCTX, seed=17)[1] for s in SOURCES}
    del sub

    def norm(S, N):
        return torch.where(N.unsqueeze(1) > 0, S / N.clamp_min(1).unsqueeze(1),
                           gmean.unsqueeze(0))

    # tok16-only reference table
    S16 = torch.zeros(KTOK, D, device=DEV); N16 = torch.zeros(KTOK, device=DEV)
    S16.index_add_(0, tokcls[tok_ids], tab[tok_ids] * cnts[tok_ids].unsqueeze(1))
    N16.index_add_(0, tokcls[tok_ids], cnts[tok_ids])
    Ttok16 = norm(S16, N16)

    # ---- pass 2 (per source): joint table + the conditional P(ctx|token) the null needs
    joint, PC, nmi = {}, {}, {}
    for s in SOURCES:
        S = torch.zeros(KTOK * KCTX, D, device=DEV); N = torch.zeros(KTOK * KCTX, device=DEV)
        JT = torch.zeros(V, KCTX, device=DEV)          # token x ctx counts
        CUR['mode'] = None
        for i in range(0, NFIT, 8):
            idx = FITR[i:i + 8, :-1].to(DEV).contiguous()
            fwd(idx)
            o = caps['out'].reshape(-1, D)
            f = feat(caps['in'], s).reshape(-1, D)
            c = torch.cdist(f, ctxC[s]).argmin(1)
            k = tokcls[idx].reshape(-1)
            kc = k * KCTX + c
            S.index_add_(0, kc, o); N.index_add_(0, kc, torch.ones_like(kc, dtype=torch.float))
            JT.index_put_((idx.reshape(-1), c), torch.ones_like(c, dtype=torch.float),
                          accumulate=True)
        joint[s] = norm(S, N)
        tot = JT.sum()
        pt = JT.sum(1) / tot; pc = JT.sum(0) / tot; pj = JT / tot
        nz = pj > 0
        mi = float((pj[nz] * (pj[nz] / (pt.unsqueeze(1) * pc.unsqueeze(0))[nz]).log()).sum())
        hc = float(-(pc[pc > 0] * pc[pc > 0].log()).sum())
        nmi[s] = round(mi / max(hc, 1e-9), 4)
        PC[s] = torch.where(JT.sum(1, keepdim=True) > 0, JT / JT.sum(1, keepdim=True).clamp_min(1),
                            torch.full_like(JT, 1.0 / KCTX))
    hk2.remove()

    # ---- the within-token null: labels resampled from P(ctx|token), fit AND eval
    def shuf_labels(rows_tok, s):
        flat = rows_tok.reshape(-1)
        return torch.multinomial(PC[s][flat], 1, generator=gen).squeeze(1).view(rows_tok.shape)

    joint_shuf = {}
    for s in SOURCES:
        S = torch.zeros(KTOK * KCTX, D, device=DEV); N = torch.zeros(KTOK * KCTX, device=DEV)
        hk3 = H[LI].mlp.register_forward_hook(cap_hook)
        CUR['mode'] = None
        for i in range(0, NFIT, 8):
            idx = FITR[i:i + 8, :-1].to(DEV).contiguous()
            fwd(idx)
            o = caps['out'].reshape(-1, D)
            c = shuf_labels(idx, s).reshape(-1)
            kc = tokcls[idx].reshape(-1) * KCTX + c
            S.index_add_(0, kc, o); N.index_add_(0, kc, torch.ones_like(kc, dtype=torch.float))
        hk3.remove()
        joint_shuf[s] = norm(S, N)

    def ce_eval(mode, tab_=None, joint_=None, src=None):
        CUR.update(mode=mode, tab=tab_, joint=joint_, src=src)
        tot = 0.0; n = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            if mode == 'tokshuf':
                CUR['shuf'] = shuf_labels(idx, src)
            lo = fwd(idx).float()
            tot += float(F.cross_entropy(lo[:, 16:].reshape(-1, lo.shape[-1]),
                                         tg[:, 16:].reshape(-1), reduction='sum'))
            n += idx.shape[0] * (T - 16)
        return tot / n

    CUR.update(mean=gmean, tokcls=tokcls)
    ce = {'full': ce_eval(None), 'mean': ce_eval('mean'),
          'tok50k': ce_eval('tab', tab_=tab), 'tok16': ce_eval('tab', tab_=Ttok16[tokcls])}
    for s in SOURCES:
        CUR['ctxC'] = ctxC[s]
        ce[f'ctx_{s}'] = ce_eval('tokctx', joint_=joint[s], src=s)
        ce[f'null_{s}'] = ce_eval('tokshuf', joint_=joint_shuf[s], src=s)
    hk.remove()

    stake = ce['mean'] - ce['full']
    rec = {k: round((ce['mean'] - v) / max(stake, 1e-6), 4)
           for k, v in ce.items() if k not in ('full', 'mean')}
    incr = {s: round(rec[f'ctx_{s}'] - rec[f'null_{s}'], 4) for s in SOURCES}
    return {'layer': LI, 'ce': {k: round(v, 4) for k, v in ce.items()},
            'stake': round(stake, 4), 'recovery': rec, 'increment': incr,
            'nmi_ctx_token': nmi}


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ROWS = cl.fineweb_rows(NFIT + NR)
    FITR = ROWS[:NFIT, :T + 1].contiguous(); EVR = ROWS[NFIT:, :T + 1].contiguous()
    cl.assert_disjoint(FITR, EVR, label='mlp_context_tables2')
    gen = torch.Generator(device=DEV).manual_seed(3301)

    layers = []
    for LI in LAYERS:
        d = one_layer(LI, FITR, EVR, gen)
        layers.append(d)
        r, inc = d['recovery'], d['increment']
        print(f"mlp{LI}: stake {d['stake']:.3f} | tok50k {r['tok50k']:.3f} tok16 {r['tok16']:.3f}"
              f" | self {r['ctx_self']:.3f}/{r['null_self']:.3f} ({inc['self']:+.3f})"
              f" prev {r['ctx_prev']:.3f}/{r['null_prev']:.3f} ({inc['prev']:+.3f})"
              f" doc {r['ctx_doc']:.3f}/{r['null_doc']:.3f} ({inc['doc']:+.3f})"
              f" | nmi {d['nmi_ctx_token']}", flush=True)
        json.dump({'layers': layers, 'partial': True}, open(OUT, 'w'), indent=1)

    by = {d['layer']: d for d in layers}
    pa = by[1]['increment']['self'] < 0.05
    pb = by[17]['increment']['self'] >= 0.08
    pc = (by[17]['increment']['doc'] >= 0.05 and
          by[17]['increment']['doc'] > by[1]['increment']['doc'])
    out = {'n_fit': NFIT, 'n_eval': NR, 'k_tok': KTOK, 'k_ctx': KCTX, 'sources': list(SOURCES),
           'layers': layers, 'pred_a_v1_was_token_artifact': bool(pa),
           'pred_b_top_survives': bool(pb), 'pred_c_doc_state_at_top': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\npred_a v1-token-artifact {pa} (mlp1 self {by[1]['increment']['self']:+.3f})")
    print(f"pred_b top-survives {pb} (mlp17 self {by[17]['increment']['self']:+.3f})")
    print(f"pred_c doc-state-at-top {pc} (mlp17 doc {by[17]['increment']['doc']:+.3f} "
          f"vs mlp1 doc {by[1]['increment']['doc']:+.3f})")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
