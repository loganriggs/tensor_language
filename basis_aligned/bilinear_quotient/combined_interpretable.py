"""COMBINED INTERPRETABLE fraction of MLP L1 (capstone of 773-776). mlp1 =
TOKEN-CLASS (lexical, ~56% 773) + POSITION (causal, ~26% 776), entangled (overlap
0.62). How much of mlp1 is the UNION of these two interpretable low-rank subspaces,
and how small is the genuinely irreducible / distributed remainder? Build the
combined subspace span(token-class(64) UNION position(32)) (effective dim < 96 due
to overlap), and measure keep-only-combined CE-recovery + the causal weight of what
remains after removing it.

REGISTERED PREDICTIONS:
  (0) SANITY: combined subspace effective dim between 64 and 96 (overlap -> < 96);
  (a) MOSTLY INTERPRETABLE: keep-only the combined token-class+position subspace
      recovers >= 0.7 of mlp1's CE benefit (>> a random subspace of the same dim),
      so mlp1 is largely two nameable low-rank structures + a small remainder;
  (b) SMALL IRREDUCIBLE CORE: removing the combined subspace leaves a residual whose
      causal weight (dCE of removing it, i.e. mlp1 benefit minus combined) is the
      genuinely distributed part -- report its size;
  NULL: random same-dim subspace recovers far less."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; LAYER = 1
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'combined_interpretable_results.json'
NEVAL = 64; MINCOUNT = 5; RTOK = 64; RPOS = 32
MODE = {'U': None, 'op': None}


def mlp_hook(mo, i_, o_):
    if MODE['op'] is None: return o_
    sh = o_.shape; v = o_.reshape(-1, D).float()
    if MODE['op'] == 'ablate': v2 = torch.zeros_like(v)
    else:
        U = MODE['U']; v2 = (v @ U) @ U.T if MODE['op'] == 'keep' else v - (v @ U) @ U.T
    return v2.reshape(sh).to(o_.dtype)


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
    cap = []; toks = []; pos = []
    h = m.transformer.h[LAYER].mlp.register_forward_hook(lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D)))
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); forward_logits(idx)
        c = idx.cpu().numpy(); toks.append(c.reshape(-1))
        pos.append(np.broadcast_to(np.arange(c.shape[1]), c.shape).reshape(-1))
    h.remove(); return torch.cat(cap, 0), np.concatenate(toks), np.concatenate(pos)


def mean_subspace(O, labels, r):
    g = O.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(labels):
        if t < 0: continue
        mk = labels == t
        if mk.sum() < MINCOUNT: continue
        rows.append(O[mk].mean(0) - g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0) * torch.tensor(wt, device=O.device, dtype=O.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:r].T.contiguous()


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    h0 = m.transformer.h[LAYER].mlp.register_forward_hook(mlp_hook)
    O, toks, pos = capture(rows, NEVAL)
    Utok = mean_subspace(O, toks, RTOK); Upos = mean_subspace(O, pos, RPOS)
    # combined union subspace: SVD of [Utok | Upos], keep sing-vecs above threshold
    Uc, Sc, _ = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)
    kdim = int((Sc > 1e-3).sum().item()); Ucomb = Uc[:, :kdim].contiguous()
    print(f'combined union dim {kdim} (of token 64 + pos 32 = 96; overlap compresses it)', flush=True)

    MODE['op'] = None; ce_full = ce_on(rows, NEVAL)
    MODE['op'] = 'ablate'; ce_abl = ce_on(rows, NEVAL); ben = ce_abl - ce_full
    def keeprec(U):
        MODE['op'] = 'keep'; MODE['U'] = U; c = ce_on(rows, NEVAL); MODE['op'] = None; MODE['U'] = None
        return float((ce_abl - c)/max(ben, 1e-6))
    def removedce(U):
        MODE['op'] = 'remove'; MODE['U'] = U; c = ce_on(rows, NEVAL); MODE['op'] = None; MODE['U'] = None
        return float(c - ce_full)
    rec_comb = keeprec(Ucomb); rec_tok = keeprec(Utok); rec_pos = keeprec(Upos)
    g = torch.Generator(device=DEV).manual_seed(0); Ur = torch.linalg.qr(torch.randn(D, kdim, generator=g, device=DEV))[0]
    rec_rand = keeprec(Ur)
    dce_remainder = removedce(Ucomb)     # causal weight of what's OUTSIDE token+position
    print(f'keep-only: combined {rec_comb:.3f} | token {rec_tok:.3f} | position {rec_pos:.3f} | random-{kdim}d {rec_rand:.3f}', flush=True)
    print(f'remove combined -> irreducible remainder dCE {dce_remainder:.3f} (of benefit {ben:.3f})', flush=True)
    h0.remove()

    p0 = 64 <= kdim <= 96
    pa = rec_comb >= 0.7 and rec_comb > 1.5*rec_rand
    out = {'benefit': round(ben, 4), 'combined_dim': kdim, 'keep_combined': round(rec_comb, 4),
           'keep_token': round(rec_tok, 4), 'keep_position': round(rec_pos, 4), 'keep_random': round(rec_rand, 4),
           'remove_combined_dce': round(dce_remainder, 4), 'irreducible_frac': round(dce_remainder/max(ben, 1e-6), 4),
           'pred_0': bool(p0), 'pred_a_mostly_interpretable': bool(pa), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) mlp1 mostly token-class+position (keep-combined >=0.7): {pa}; irreducible remainder {out["irreducible_frac"]:.2f} of benefit', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
