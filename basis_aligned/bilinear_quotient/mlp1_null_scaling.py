# mlp1_null_scaling: THE DIRECT TEST OF §1328's CONVICTION. mlp1's context increment
# survived the within-token null at +0.144 — 2.6x the module's entire un-tableable headroom
# (1 - 0.945 = 0.055, §1323), which is impossible for genuine context. The named suspect:
# P(ctx | token) cannot be estimated for rare tokens (single fit occurrence -> degenerate
# conditional; unseen -> uniform), so the null under-performs exactly where the real label
# generalizes by nearest-centroid, and the difference masquerades as context.
#
# The two accounts make opposite scaling predictions, so scale NFIT and watch:
#   ARTIFACT account: the increment SHRINKS as fit data grows (conditionals firm up).
#   GENUINE-CONTEXT account: the increment is flat in NFIT.
#
# Protocol: mlp1 only, self-label only, NFIT in {960, 1920, 3840, 7680}, fixed eval set
# (480 rows, disjoint from the largest fit set), same centroids re-fit per NFIT (the label
# system scales with its data, as it would in use). Incremental save per NFIT.
#
# Registered predictions:
#   pred_a MONOTONE DECREASE: the self increment decreases at every NFIT step.
#   pred_b ARTIFACT SIZE: increment < 0.10 by NFIT=7680 (down from 0.144 at 1920).
#   pred_c THE BOUND'S OTHER LEG IS FIRM: mlp1's tok50k recovery stays within +-0.02 of
#          0.945 at every NFIT — else the ceiling that convicts the increment is itself
#          data-soft and §1328's gate needs re-derivation.
import json, os, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp1_null_scaling_results.json'
NFITS = (960, 1920, 3840, 7680); NR = 480; V = 50257
KTOK = 16; KCTX = 16; LI = 1
H = m.transformer.h
CUR = {'toks': None, 'mode': None, 'mean': None, 'tab': None, 'tokcls': None,
       'ctxC': None, 'shuf': None, 'joint': None}


def mlp_hook(mod, args, out):
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
        f = args[0].float().reshape(-1, D)
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
def one_nfit(NFIT, FITR, EVR, gen):
    hk = H[LI].mlp.register_forward_hook(mlp_hook)
    caps = {}
    def cap_hook(mod, args, out):
        caps['in'] = args[0].detach().float(); caps['out'] = out.detach().float()
        return out
    hk2 = H[LI].mlp.register_forward_hook(cap_hook)

    sums = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
    sub = []
    CUR['mode'] = None
    stride = max(1, (NFIT * T) // 60000)   # cap the clustering sample ~60k
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

    def norm(S, N):
        return torch.where(N.unsqueeze(1) > 0, S / N.clamp_min(1).unsqueeze(1),
                           gmean.unsqueeze(0))

    # joint table + P(ctx|token)
    S = torch.zeros(KTOK * KCTX, D, device=DEV); N = torch.zeros(KTOK * KCTX, device=DEV)
    JT = torch.zeros(V, KCTX, device=DEV)
    for i in range(0, NFIT, 8):
        idx = FITR[i:i + 8, :-1].to(DEV).contiguous()
        fwd(idx)
        o = caps['out'].reshape(-1, D)
        c = torch.cdist(caps['in'].reshape(-1, D), ctxC).argmin(1)
        kc = tokcls[idx].reshape(-1) * KCTX + c
        S.index_add_(0, kc, o); N.index_add_(0, kc, torch.ones_like(kc, dtype=torch.float))
        JT.index_put_((idx.reshape(-1), c), torch.ones_like(c, dtype=torch.float),
                      accumulate=True)
    joint = norm(S, N)
    PC = torch.where(JT.sum(1, keepdim=True) > 0, JT / JT.sum(1, keepdim=True).clamp_min(1),
                     torch.full_like(JT, 1.0 / KCTX))
    frac_sparse = float((JT.sum(1) > 0).float().sum() / max(int(seen.sum()), 1))
    tok_le2 = float(((JT.sum(1) <= 2) & seen).float().sum() / max(int(seen.sum()), 1))

    def shuf_labels(rows_tok):
        flat = rows_tok.reshape(-1)
        return torch.multinomial(PC[flat], 1, generator=gen).squeeze(1).view(rows_tok.shape)

    Sh = torch.zeros(KTOK * KCTX, D, device=DEV); Nh = torch.zeros(KTOK * KCTX, device=DEV)
    for i in range(0, NFIT, 8):
        idx = FITR[i:i + 8, :-1].to(DEV).contiguous()
        fwd(idx)
        o = caps['out'].reshape(-1, D)
        c = shuf_labels(idx).reshape(-1)
        kc = tokcls[idx].reshape(-1) * KCTX + c
        Sh.index_add_(0, kc, o); Nh.index_add_(0, kc, torch.ones_like(kc, dtype=torch.float))
    joint_shuf = norm(Sh, Nh)
    hk2.remove()

    def ce_eval(mode, tab_=None, joint_=None):
        CUR.update(mode=mode, tab=tab_, joint=joint_)
        tot = 0.0; n = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            if mode == 'tokshuf':
                CUR['shuf'] = shuf_labels(idx)
            lo = fwd(idx).float()
            tot += float(F.cross_entropy(lo[:, 16:].reshape(-1, lo.shape[-1]),
                                         tg[:, 16:].reshape(-1), reduction='sum'))
            n += idx.shape[0] * (T - 16)
        return tot / n

    CUR.update(mean=gmean, tokcls=tokcls, ctxC=ctxC)
    ce = {'full': ce_eval(None), 'mean': ce_eval('mean'), 'tok50k': ce_eval('tab', tab_=tab),
          'ctx': ce_eval('tokctx', joint_=joint),
          'null': ce_eval('tokshuf', joint_=joint_shuf)}
    hk.remove()
    stake = ce['mean'] - ce['full']
    rec = {k: round((ce['mean'] - v) / max(stake, 1e-6), 4)
           for k, v in ce.items() if k not in ('full', 'mean')}
    return {'n_fit': NFIT, 'ce': {k: round(v, 4) for k, v in ce.items()},
            'stake': round(stake, 4), 'recovery': rec,
            'increment': round(rec['ctx'] - rec['null'], 4),
            'tok_coverage': round(frac_sparse, 4),
            'frac_tokens_le2_occurrences': round(tok_le2, 4)}


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ROWS = cl.fineweb_rows(max(NFITS) + NR)
    EVR = ROWS[max(NFITS):, :T + 1].contiguous()
    gen = torch.Generator(device=DEV).manual_seed(3301)

    steps = []
    if os.path.exists(OUT):
        try:
            steps = json.load(open(OUT)).get('steps', [])
        except Exception:
            steps = []
    done = {s['n_fit'] for s in steps}
    for NFIT in NFITS:
        if NFIT in done:
            continue
        FITR = ROWS[:NFIT, :T + 1].contiguous()
        cl.assert_disjoint(FITR, EVR, label=f'nfit{NFIT}')
        d = one_nfit(NFIT, FITR, EVR, gen)
        steps.append(d); steps.sort(key=lambda z: z['n_fit'])
        print(f"NFIT {NFIT}: incr {d['increment']:+.4f} | tok50k {d['recovery']['tok50k']:.4f}"
              f" | ctx {d['recovery']['ctx']:.4f} null {d['recovery']['null']:.4f}"
              f" | frac_le2 {d['frac_tokens_le2_occurrences']:.3f}", flush=True)
        json.dump({'steps': steps, 'partial': True}, open(OUT, 'w'), indent=1)

    incs = [s['increment'] for s in steps]
    ceils = [s['recovery']['tok50k'] for s in steps]
    pa = all(incs[i + 1] < incs[i] for i in range(len(incs) - 1))
    pb = incs[-1] < 0.10
    pc = all(abs(c - 0.945) <= 0.02 for c in ceils)
    out = {'layer': LI, 'n_eval': NR, 'steps': steps,
           'pred_a_monotone_decrease': bool(pa), 'pred_b_below_010': bool(pb),
           'pred_c_ceiling_stable': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nincrements {incs} | ceilings {ceils}")
    print(f"pred_a monotone {pa} | pred_b <0.10 {pb} | pred_c ceiling-stable {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
