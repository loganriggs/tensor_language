"""WHAT IS mlp1's 26% REMAINDER? (bottom-up, §809 follow-up). mlp1 is the one early
bilin18 component with substantial loss-benefit (1.07 nats) that is NOT fully class+
position: mean-preserving keep is 0.738, leaving 26% unexplained. Decompose that residual.
mlp1 reads attn0/attn1 outputs, which the census showed build the copy-source (name /
induction-target circuit), so the leading candidate for the remainder is a PREVIOUS-TOKEN
(bigram/context) variable; a second candidate is a class x position INTERACTION that the
additive class+position subspace misses. Test both causally, with a random null, all
mean-preserving.

For mlp1: benefit, then keep-only CE-recovery (mean-preserving) for
  - class+position (baseline, expect ~0.74);
  - class+position + PREVIOUS-TOKEN class subspace;
  - class+position + JOINT (token x coarse-position-bin) mean subspace;
  - class+position + RANDOM subspace of the same added rank (NULL).
Also report the residual-after-(class+position) effective rank (is the remainder low-rank
at all?).

REGISTERED PREDICTIONS:
  (0) SANITY: class+position keep reproduces ~0.74; ablate benefit ~1.07;
  (a) if PREV-TOKEN adds recovery >> random (raises keep toward >=0.85), mlp1's remainder
      includes a previous-token/bigram variable -> a THIRD named early variable;
  (b) if JOINT adds recovery >> both class+position AND prev-token, the remainder is a
      class x position interaction (additive subspace was the limitation);
  (c) NULL: if neither beats random by much, mlp1's remainder is genuinely diffuse
      (consistent with the whole-model diffuse-remainder verdict, now confirmed at the
      single-component level);
  report residual eff-rank (low -> a handle exists; high -> diffuse)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp1_remainder_results.json'
NEVAL = 160; MINCOUNT = 5; RTOK = 64; RPOS = 32; RPREV = 64; RJOINT = 64; NBIN = 4
LAYER = 1
SUB = {'U': None, 'mean': None, 'op': None}


def comp():
    return m.transformer.h[LAYER].mlp


def hook(mo, i_, o_):
    if SUB['op'] is None: return o_
    y = o_[0] if isinstance(o_, tuple) else o_; sh = y.shape; v = y.reshape(-1, D).float()
    if SUB['op'] == 'ablate': v2 = torch.zeros_like(v)
    else: U = SUB['U']; mu = SUB['mean']; v2 = mu + ((v - mu) @ U) @ U.T   # mean-preserving keep
    yn = v2.reshape(sh).to(y.dtype)
    return (yn,) + tuple(o_[1:]) if isinstance(o_, tuple) else yn


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
def capture(rows, n):
    cap = []; toks = []; prev = []; pos = []
    def h(mo, i_, o_): cap.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hh = comp().register_forward_hook(h)
    for i in range(0, n, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx)
        c = idx.cpu().numpy(); T = c.shape[1]
        toks.append(c.reshape(-1))
        pv = np.concatenate([np.full((c.shape[0], 1), -1), c[:, :-1]], 1)   # previous token (-1 at pos0)
        prev.append(pv.reshape(-1))
        pos.append(np.broadcast_to(np.arange(T), c.shape).reshape(-1))
    hh.remove(); return torch.cat(cap, 0), np.concatenate(toks), np.concatenate(prev), np.concatenate(pos)


def mean_subspace(O, labels, r, gmean):
    rows = []; wt = []
    for t in np.unique(labels):
        if t < 0: continue
        mk = labels == t
        if mk.sum() < MINCOUNT: continue
        rows.append(O[mk].mean(0) - gmean[0]); wt.append(np.sqrt(mk.sum()))
    if not rows: return None
    M = torch.stack(rows, 0) * torch.tensor(wt, device=O.device, dtype=O.dtype)[:, None]
    k = min(r, M.shape[0])
    return torch.linalg.svd(M, full_matrices=False)[2][:k].T.contiguous()


def orth(*mats):
    mats = [x for x in mats if x is not None]
    C = torch.cat(mats, 1)
    return torch.linalg.svd(C, full_matrices=False)[0][:, :C.shape[1]].contiguous()


def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    O, toks, prev, pos = capture(rows, NEVAL)
    gmean = O.mean(0, keepdim=True)
    posbin = (pos.astype(np.int64) * NBIN // (pos.max() + 1))
    Utok = mean_subspace(O, toks, RTOK, gmean); Upos = mean_subspace(O, pos, RPOS, gmean)
    Uprev = mean_subspace(O, prev, RPREV, gmean)
    joint_lab = toks.astype(np.int64) * NBIN + posbin
    Ujoint = mean_subspace(O, joint_lab, RJOINT, gmean)
    Ucp = orth(Utok, Upos)
    # residual-after-(class+position) eff-rank
    P = O - gmean; resid = P - (P @ Ucp) @ Ucp.T
    s2 = torch.linalg.svdvals(resid)**2; eff = float((s2.sum()**2)/(s2**2).sum())
    g = torch.Generator(device=DEV).manual_seed(0)
    added = Ucp.shape[1]  # dims class+position uses; give each candidate a matched random add
    Urand_add = torch.linalg.qr(torch.randn(D, RPREV, generator=g, device=DEV))[0]

    h = comp().register_forward_hook(hook)
    SUB['mean'] = gmean; SUB['op'] = None; ce_full = ce_on(rows, NEVAL)
    SUB['op'] = 'ablate'; ce_abl = ce_on(rows, NEVAL); ben = ce_abl - ce_full
    def keeprec(U): SUB['op'] = 'keep'; SUB['U'] = U; c = ce_on(rows, NEVAL); SUB['op'] = None; return float((ce_abl-c)/max(ben, 1e-6))
    res = {}
    res['classpos'] = round(keeprec(Ucp), 4)
    res['classpos_prev'] = round(keeprec(orth(Ucp, Uprev)), 4)
    res['classpos_joint'] = round(keeprec(orth(Ucp, Ujoint)), 4)
    res['classpos_rand'] = round(keeprec(orth(Ucp, Urand_add)), 4)
    h.remove()
    prev_gain = res['classpos_prev'] - res['classpos']; rand_gain = res['classpos_rand'] - res['classpos']
    joint_gain = res['classpos_joint'] - res['classpos']
    verdict = ('prev-token variable' if prev_gain > 2*max(rand_gain, 0.02) and prev_gain > 0.06 else
               'class x position interaction' if joint_gain > 2*max(rand_gain, 0.02) and joint_gain > 0.06 else
               'diffuse (no low-rank handle beyond class+position)')
    out = {'component': f'mlp{LAYER}', 'benefit': round(ben, 4), 'residual_eff_rank': round(eff, 2),
           'keep': res, 'prev_gain': round(prev_gain, 4), 'joint_gain': round(joint_gain, 4),
           'rand_gain': round(rand_gain, 4), 'verdict': verdict, 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'mlp1 benefit {ben:.3f} | resid eff-rank {eff:.1f}', flush=True)
    print(f'keep: class+pos {res["classpos"]} | +prev {res["classpos_prev"]} (gain {prev_gain:+.3f}) | +joint {res["classpos_joint"]} (gain {joint_gain:+.3f}) | +rand {res["classpos_rand"]} (gain {rand_gain:+.3f})', flush=True)
    print(f'VERDICT: mlp1 remainder = {verdict}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
