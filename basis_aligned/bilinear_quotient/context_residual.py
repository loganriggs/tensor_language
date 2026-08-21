"""CONTEXT RESIDUAL of MLP L1 (773 located L1-L3 as where NON-token-class,
context-dependent computation enters -- what IS it?). Take mlp1's output, remove
its current-token-class subspace (the part 767-773 explains), and ask what the
REMAINING variance encodes. First hypothesis: the PREVIOUS token (bigram/context).
Compute the previous-token-conditional-mean subspace of the residual and test how
much of the residual it explains, whether it is causal, and name a few directions.

REGISTERED PREDICTIONS:
  (0) SANITY: removing the current-token subspace leaves nonzero residual at L1;
  (a) BIGRAM CONTEXT: the previous-token-conditional-mean subspace explains a
      substantial share of the L1 context residual (top-64 captures >= 40% of
      residual variance, >> a shuffled-previous-token null), i.e. much of L1's
      non-token-class computation is PREVIOUS-TOKEN (bigram) driven;
  (b) report residual share explained by prev-token, causal dCE of removing the
      prev-token subspace from mlp1, and a few named prev-token directions;
  NULL: shuffling the previous-token labels destroys the explained share."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; LAYER = 1
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'context_residual_results.json'
NEVAL = 48; MINCOUNT = 5; RSEM = 64
PROJ = {'U': None}


def mlp_proj_hook(mo, i_, o_):
    if PROJ['U'] is None: return o_
    U = PROJ['U']; sh = o_.shape; v = o_.reshape(-1, D).float()
    return (v - (v @ U) @ U.T).reshape(sh).to(o_.dtype)


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
    cap = []; cur = []; prev = []
    h = m.transformer.h[LAYER].mlp.register_forward_hook(lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D)))
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous()   # (b, T)
        forward_logits(idx)
        c = idx.cpu().numpy()                                            # (b, T)
        p = np.full_like(c, -1); p[:, 1:] = c[:, :-1]                    # previous token per position
        cur.append(c.reshape(-1)); prev.append(p.reshape(-1))
    h.remove(); return torch.cat(cap, 0), np.concatenate(cur), np.concatenate(prev)


def mean_subspace(O, labels, r=RSEM):
    g = O.mean(0, keepdim=True); rows = []; wt = []; ids = []
    for t in np.unique(labels):
        if t < 0: continue
        mk = labels == t
        if mk.sum() < MINCOUNT: continue
        rows.append(O[mk].mean(0) - g[0]); wt.append(np.sqrt(mk.sum())); ids.append(int(t))
    M = torch.stack(rows, 0) * torch.tensor(wt, device=O.device, dtype=O.dtype)[:, None]
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    return Vh[:r].T.contiguous(), torch.stack(rows, 0), np.array(ids)


def var_in_subspace(R, U):
    # fraction of R's total variance captured by projecting onto U
    Rc = R - R.mean(0, keepdim=True)
    proj = (Rc @ U) @ U.T
    return float((proj**2).sum()/(Rc**2).sum().clamp_min(1e-9))


def d1(t):
    try: return repr(cl.d1(int(t)))
    except Exception: return f'<{t}>'


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    h0 = m.transformer.h[LAYER].mlp.register_forward_hook(mlp_proj_hook)
    O, cur, prev = capture(rows, NEVAL)

    Ucur, _, _ = mean_subspace(O, cur)                    # current-token-class subspace
    Rc = O - O.mean(0, keepdim=True)
    R = Rc - (Rc @ Ucur) @ Ucur.T                         # context residual (current-token removed)
    resid_frac = float((R**2).sum()/(Rc**2).sum())
    print(f'residual after removing current-token subspace: {resid_frac:.3f} of variance', flush=True)

    Uprev, Mprev, ids = mean_subspace(R, prev)            # prev-token subspace OF the residual
    share = var_in_subspace(R, Uprev)
    # shuffled-prev null
    g = np.random.RandomState(0); prev_sh = prev.copy(); m_ = prev_sh >= 0; prev_sh[m_] = g.permutation(prev_sh[m_])
    Uprev_n, _, _ = mean_subspace(R, prev_sh); share_null = var_in_subspace(R, Uprev_n)
    print(f'(a) prev-token subspace explains {share:.3f} of the L1 context residual (shuffled null {share_null:.3f})', flush=True)

    # causal: remove prev-token subspace (in full mlp1 output space) -> dCE
    PROJ['U'] = None; ce_full = ce_on(rows, NEVAL)
    PROJ['U'] = Uprev; ce_rem = ce_on(rows, NEVAL)
    gg = torch.Generator(device=DEV).manual_seed(0); Ur = torch.linalg.qr(torch.randn(D, RSEM, generator=gg, device=DEV))[0]
    PROJ['U'] = Ur; ce_rand = ce_on(rows, NEVAL); PROJ['U'] = None
    d_prev = ce_rem - ce_full; d_rand = ce_rand - ce_full
    print(f'(b) causal: remove prev-token subspace dCE {d_prev:.3f} vs random {d_rand:.3f}  ratio {d_prev/max(d_rand,1e-6):.1f}', flush=True)

    proj = (Mprev @ Uprev[:, :6]).cpu().numpy()
    named = []
    for dctr in range(6):
        col = proj[:, dctr]; hi = ids[np.argsort(-col)[:6]]
        named.append([d1(t) for t in hi]); print(f'   prev-dir {dctr} HIGH prev-tokens {[d1(t) for t in hi]}', flush=True)
    h0.remove()

    p0 = resid_frac > 0.05
    pa = share >= 0.4 and share > 2*share_null
    null_ok = share_null < 0.5*share
    out = {'residual_frac': round(resid_frac, 4), 'prev_share': round(share, 4), 'prev_share_null': round(share_null, 4),
           'dce_prev': round(d_prev, 4), 'dce_random': round(d_rand, 4), 'named_prev_dirs': named,
           'pred_0': bool(p0), 'pred_a_bigram': bool(pa), 'null_ok': bool(null_ok), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) L1 context residual is prev-token (bigram) driven (>=0.4 & >>null): {pa}; NULL: {null_ok}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
