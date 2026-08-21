"""SEMANTIC SUFFICIENCY (capstone to 767/768). 767 showed the token-semantic
subspace of mlp0 is NECESSARY (removing it costs 1.34 nats). This asks the
complement: is it SUFFICIENT? Keep ONLY the top-r semantic subspace of mlp0's
output (project onto it, discard the rest) and measure how much of mlp0's CE
benefit survives, vs keeping only a random r-subspace. If a small semantic r
preserves most of the benefit, the nameable part-of-speech structure (768) is not
just necessary but a near-complete low-rank summary of what mlp0 does for the loss.

REGISTERED PREDICTIONS:
  (0) SANITY: keeping the full output = full CE; keeping nothing = ablation;
  (a) SUFFICIENT: keeping only the top-64 semantic subspace recovers >= 0.6 of
      mlp0's CE benefit (and >> keeping a random 64-subspace), so the nameable
      token-class structure is a near-complete low-rank summary of mlp0's loss role;
  (b) report CE-recovery vs r (16/64/128/256) for semantic-only vs random-only;
  NULL: random r-subspace-only recovers far less at matched r."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'semantic_sufficiency_results.json'
NEVAL = 48; MINCOUNT = 5; RS = [16, 64, 128, 256]
KEEP = {'U': None, 'ablate': False}


def mlp0_hook(mo, i_, o_):
    if KEEP['ablate']: return torch.zeros_like(o_)
    if KEEP['U'] is None: return o_
    U = KEEP['U']; sh = o_.shape; o = o_.reshape(-1, D).float()
    return ((o @ U) @ U.T).reshape(sh).to(o_.dtype)          # KEEP only span(U)


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
def capture_out(rows, n):
    cap = []; toks = []
    h = m.transformer.h[0].mlp.register_forward_hook(lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D)))
    for i in range(0, n, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); toks.append(idx.reshape(-1).cpu())
        forward_logits(idx)
    h.remove(); return torch.cat(cap, 0), torch.cat(toks).numpy()


def semantic_dirs(O, toks):
    g = O.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(toks):
        mk = toks == t
        if mk.sum() < MINCOUNT: continue
        rows.append(O[mk].mean(0) - g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0) * torch.tensor(wt, device=O.device, dtype=O.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2]


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    h0 = m.transformer.h[0].mlp.register_forward_hook(mlp0_hook)
    O, toks = capture_out(rows, NEVAL)
    Vh = semantic_dirs(O, toks)

    KEEP['U'] = None; KEEP['ablate'] = False; ce_full = ce_on(rows, NEVAL)
    KEEP['ablate'] = True; ce_abl = ce_on(rows, NEVAL); KEEP['ablate'] = False
    ben = ce_abl - ce_full
    print(f'CE_full {ce_full:.3f}  ablate {ce_abl:.3f}  benefit {ben:.3f}', flush=True)
    g = torch.Generator(device=DEV).manual_seed(0)
    res = {}
    for r in RS:
        KEEP['U'] = Vh[:r].T.contiguous(); ce_sem = ce_on(rows, NEVAL)
        Ur = torch.linalg.qr(torch.randn(D, r, generator=g, device=DEV))[0]
        KEEP['U'] = Ur; ce_rand = ce_on(rows, NEVAL); KEEP['U'] = None
        rec_sem = (ce_abl - ce_sem)/max(ben, 1e-6); rec_rand = (ce_abl - ce_rand)/max(ben, 1e-6)
        res[str(r)] = {'sem_recovery': round(float(rec_sem), 4), 'rand_recovery': round(float(rec_rand), 4)}
        print(f'r={r:3d}: keep-semantic CE-recovery {rec_sem:.3f}  keep-random {rec_rand:.3f}', flush=True)
    h0.remove()

    pa = res['64']['sem_recovery'] >= 0.6 and res['64']['sem_recovery'] > 2*max(res['64']['rand_recovery'], 1e-6)
    null_ok = res['64']['rand_recovery'] < res['64']['sem_recovery']
    out = {'ce_full': round(ce_full, 4), 'benefit': round(ben, 4), 'rs': RS, 'results': res,
           'pred_a_sufficient': bool(pa), 'null_ok': bool(null_ok), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) semantic subspace SUFFICIENT (keep-only top-64 >=0.6 & >>random): {pa}; NULL: {null_ok}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
