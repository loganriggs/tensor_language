"""IS mlp1's prev-token + class×position remainder a GENERAL early-MLP recipe? (§810
follow-up). §810 found mlp1's 26% remainder is mostly diffuse (eff-rank 462) but contains
modest, causal previous-token (net +0.075 over random) and class×position-interaction (net
+0.089) slices. Test whether this holds for the other early content-carrying MLPs (mlp2,
mlp3) or is mlp1-specific, and whether COMBINING prev+joint closes more of the gap than
either alone. All mean-preserving, matched-rank random null.

For each of mlp1, mlp2, mlp3: benefit, residual eff-rank, and keep-only recovery for
class+position baseline; +prev; +joint; +BOTH(prev+joint); +random(matched to the combined
add). Net gains over the random null are the honest signal.

REGISTERED PREDICTIONS:
  (0) SANITY: class+position keep reproduces §808 values (mlp1 ~0.74, mlp2 ~0.38, mlp3 ~0.45);
  (a) GENERAL: if mlp2/mlp3 also show prev and/or joint gains >> random, the prev-token +
      interaction slice is a general early-MLP remainder recipe;
  (b) SPECIFIC: if only mlp1 shows it, it is mlp1-specific (mlp1 uniquely reads the
      attn0/attn1 copy-source);
  (c) COMBINED: report whether prev+joint together exceed the better single candidate
      (do they capture DIFFERENT slices, or the same one?);
  NULL: matched-rank random gain is the floor every candidate must clear."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp_remainder_sweep_results.json'
NEVAL = 160; MINCOUNT = 5; RTOK = 64; RPOS = 32; RPREV = 64; RJOINT = 64; NBIN = 4
LAYERS = [1, 2, 3]
SUB = {'U': None, 'mean': None, 'op': None, 'L': None}


def comp(L): return m.transformer.h[L].mlp


def mk_hook(L):
    def hook(mo, i_, o_):
        if SUB['op'] is None or SUB['L'] != L: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; sh = y.shape; v = y.reshape(-1, D).float()
        if SUB['op'] == 'ablate': v2 = torch.zeros_like(v)
        else: U = SUB['U']; mu = SUB['mean']; v2 = mu + ((v - mu) @ U) @ U.T
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
def capture(rows, n, L):
    cap = []; toks = []; prev = []; pos = []
    def h(mo, i_, o_): cap.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hh = comp(L).register_forward_hook(h)
    for i in range(0, n, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx)
        c = idx.cpu().numpy(); T = c.shape[1]
        toks.append(c.reshape(-1))
        prev.append(np.concatenate([np.full((c.shape[0], 1), -1), c[:, :-1]], 1).reshape(-1))
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


def run_layer(rows, L):
    O, toks, prev, pos = capture(rows, NEVAL, L)
    gmean = O.mean(0, keepdim=True)
    posbin = (pos.astype(np.int64) * NBIN // (pos.max() + 1))
    Utok = mean_subspace(O, toks, RTOK, gmean); Upos = mean_subspace(O, pos, RPOS, gmean)
    Uprev = mean_subspace(O, prev, RPREV, gmean)
    Ujoint = mean_subspace(O, toks.astype(np.int64) * NBIN + posbin, RJOINT, gmean)
    Ucp = orth(Utok, Upos)
    P = O - gmean; resid = P - (P @ Ucp) @ Ucp.T
    s2 = torch.linalg.svdvals(resid)**2; eff = float((s2.sum()**2)/(s2**2).sum())
    g = torch.Generator(device=DEV).manual_seed(L)
    Urand = torch.linalg.qr(torch.randn(D, RPREV + RJOINT, generator=g, device=DEV))[0]

    h = comp(L).register_forward_hook(mk_hook(L))
    SUB['mean'] = gmean; SUB['L'] = L; SUB['op'] = None; ce_full = ce_on(rows, NEVAL)
    SUB['op'] = 'ablate'; ce_abl = ce_on(rows, NEVAL); ben = ce_abl - ce_full
    def kr(U): SUB['op'] = 'keep'; SUB['U'] = U; c = ce_on(rows, NEVAL); SUB['op'] = None; return float((ce_abl-c)/max(ben, 1e-6))
    keep = {'classpos': round(kr(Ucp), 4), 'prev': round(kr(orth(Ucp, Uprev)), 4),
            'joint': round(kr(orth(Ucp, Ujoint)), 4), 'both': round(kr(orth(Ucp, Uprev, Ujoint)), 4),
            'rand': round(kr(orth(Ucp, Urand)), 4)}
    h.remove()
    base = keep['classpos']; rg = keep['rand'] - base
    return {'benefit': round(ben, 4), 'residual_eff_rank': round(eff, 1), 'keep': keep,
            'net_prev': round(keep['prev'] - base - max(rg, 0), 4),
            'net_joint': round(keep['joint'] - base - max(rg, 0), 4),
            'net_both': round(keep['both'] - base - max(rg, 0), 4), 'rand_gain': round(rg, 4)}


def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL); out = {}
    for L in LAYERS:
        r = run_layer(rows, L); out[f'mlp{L}'] = r
        print(f'mlp{L}: ben {r["benefit"]} eff-rank {r["residual_eff_rank"]} | keep cp {r["keep"]["classpos"]} prev {r["keep"]["prev"]} joint {r["keep"]["joint"]} both {r["keep"]["both"]} rand {r["keep"]["rand"]} | NET prev {r["net_prev"]:+} joint {r["net_joint"]:+} both {r["net_both"]:+}', flush=True)
    general = sum(1 for L in LAYERS if out[f'mlp{L}']['net_both'] > 0.04) >= 2
    out['general_recipe'] = bool(general); out['runtime_s'] = time.time()-t0
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) prev+interaction is a GENERAL early-MLP remainder recipe (>=2 of 3 net_both>0.04): {general}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
