"""IS THE MIDDLE'S CLASS+POSITION AMPLIFICATION (same directions as the front) OR NEW
STRUCTURE? (§814/817 refinement). Puzzle: the residual stream is additive, so zeroing the
middle's outputs leaves the front's class+position contributions intact — yet removing the
middle costs 1.9 nats. So the middle ADDS class+position content. Question: is that content
in the SAME directions the front already wrote (amplification / genuine 'maintenance' —
re-adding the same variables, e.g. to survive rms-norm renormalization) or in DIFFERENT
directions (the middle computes NEW class+position structure)?

Test: keep-only the middle band's output (simultaneous, mean-preserving) projected onto (a)
the FRONT's aggregate class+position subspace, vs (b) the MIDDLE's OWN class+position
subspace, and measure recovery of the middle's collective ~1.9-nat benefit. Also report the
principal-angle overlap between the front and middle class+position subspaces.

REGISTERED PREDICTIONS:
  (0) SANITY: middle own-subspace keep reproduces ~0.65 (§814); random null far worse;
  (a) AMPLIFICATION: if keep-FRONT-subspace ≈ keep-OWN-subspace (>=0.8 of it) and subspace
      overlap is high, the middle re-writes the SAME class+position directions -> true
      maintenance/amplification of the front's variables;
  (b) NEW STRUCTURE: if keep-OWN >> keep-FRONT (front-subspace recovers much less), the
      middle computes class+position in NEW directions -> 'maintenance' is a misnomer, it
      augments with fresh class+position structure;
  NULL: random same-rank subspace recovers far less than either."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'middle_vs_front_subspace_results.json'
NEVAL = 160; MINCOUNT = 5; RTOK = 64; RPOS = 32
FRONT = list(range(0, 6)); MID = list(range(6, 12))
SUBS = {}; MEANS = {}; MODE = {'op': None, 'Uover': None, 'rand': None, 'active': set()}


def comp(w, L): return m.transformer.h[L].mlp if w == 'mlp' else m.transformer.h[L].attn


def mk_hook(w, L):
    name = (w, L)
    def hook(mo, i_, o_):
        if MODE['op'] is None or name not in MODE['active']: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; sh = y.shape; v = y.reshape(-1, D).float()
        if MODE['op'] == 'ablate': v2 = torch.zeros_like(v)
        else:
            mu = MEANS[name]
            U = {'own': SUBS[name], 'front': MODE['Uover'], 'rand': MODE['rand']}[MODE['op']]
            v2 = mu + ((v - mu) @ U) @ U.T
        yn = v2.reshape(sh).to(y.dtype)
        return (yn,) + tuple(o_[1:]) if isinstance(o_, tuple) else yn
    return hook


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def ce_on(rows, n):
    s = 0.0; nn = 0
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1)
        s += float(F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1)))*idx.shape[0]; nn += idx.shape[0]
    return s/nn


@torch.no_grad()
def capture(rows, n, w, L):
    cap = []; toks = []; pos = []
    def h(mo, i_, o_): cap.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hh = comp(w, L).register_forward_hook(h)
    for i in range(0, n, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx)
        c = idx.cpu().numpy(); toks.append(c.reshape(-1)); pos.append(np.broadcast_to(np.arange(c.shape[1]), c.shape).reshape(-1))
    hh.remove(); return torch.cat(cap, 0), np.concatenate(toks), np.concatenate(pos)


def mean_subspace(O, labels, r, gmean):
    rows = []; wt = []
    for t in np.unique(labels):
        mk = labels == t
        if mk.sum() < MINCOUNT: continue
        rows.append(O[mk].mean(0) - gmean[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0) * torch.tensor(wt, device=O.device, dtype=O.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:r].T.contiguous()


def agg_cp_subspace(dirs_list, rank):
    C = torch.cat(dirs_list, 1)
    return torch.linalg.svd(C, full_matrices=False)[0][:, :rank].contiguous()


def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    MODE['op'] = None
    front_dirs = []; mid_dirs = []
    for L in FRONT + MID:
        for w in ('attn', 'mlp'):
            O, toks, pos = capture(rows, NEVAL, w, L)
            MEANS[(w, L)] = O.mean(0, keepdim=True)
            Utok = mean_subspace(O, toks, RTOK, MEANS[(w, L)]); Upos = mean_subspace(O, pos, RPOS, MEANS[(w, L)])
            cp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
            SUBS[(w, L)] = cp
            (front_dirs if L in FRONT else mid_dirs).append(cp)
    U_front = agg_cp_subspace(front_dirs, RTOK+RPOS)      # aggregate front class+position subspace
    U_mid = agg_cp_subspace(mid_dirs, RTOK+RPOS)          # aggregate middle class+position subspace
    # principal-angle overlap: mean cos^2 of principal angles between the two subspaces
    s = torch.linalg.svdvals(U_front.T @ U_mid)           # singular values = cos(principal angles)
    overlap = float((s**2).mean())
    MODE['Uover'] = U_front
    g = torch.Generator(device=DEV).manual_seed(0); MODE['rand'] = torch.linalg.qr(torch.randn(D, RTOK+RPOS, generator=g, device=DEV))[0]

    hooks = [comp(w, L).register_forward_hook(mk_hook(w, L)) for L in MID for w in ('attn', 'mlp')]
    MODE['active'] = {(w, L) for L in MID for w in ('attn', 'mlp')}
    MODE['op'] = None; ce_full = ce_on(rows, NEVAL)
    MODE['op'] = 'ablate'; ce_abl = ce_on(rows, NEVAL); ben = ce_abl - ce_full
    def rec(opname): MODE['op'] = opname; c = ce_on(rows, NEVAL); MODE['op'] = None; return float((ce_abl-c)/max(ben, 1e-6))
    keep_own = round(rec('own'), 4); keep_front = round(rec('front'), 4); keep_rand = round(rec('rand'), 4)
    MODE['active'] = set()
    for h in hooks: h.remove()
    frac = keep_front / max(keep_own, 1e-6)
    verdict = ('amplification (same directions as front)' if frac >= 0.8 and overlap > 0.3 else
               'new class+position structure (different directions)' if frac < 0.6 else 'mixed')
    out = {'middle_collective_benefit': round(ben, 4), 'subspace_overlap_front_middle': round(overlap, 4),
           'keep_own': keep_own, 'keep_front_subspace': keep_front, 'keep_random': keep_rand,
           'front_over_own': round(frac, 4), 'verdict': verdict, 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'middle benefit {ben:.3f} | overlap(front,middle cp) {overlap:.3f}', flush=True)
    print(f'keep: own {keep_own} | front-subspace {keep_front} | random {keep_rand} | front/own {frac:.2f}', flush=True)
    print(f'VERDICT: {verdict}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
