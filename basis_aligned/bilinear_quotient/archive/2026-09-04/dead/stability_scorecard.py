"""COMPONENT SCORECARD -- convergent validity WITHOUT ground truth (answers: how do
we know a component is a real/good circuit in a real model?). The idea: no single
property certifies a circuit, so TRIANGULATE -- measure independent properties and
test whether they CONVERGE on the same atoms. If the atoms that RECUR across
random re-fits (stability = the ML analog of replication) are ALSO the causally
important ones AND the monosemantic ones, that convergence is strong ground-truth-
free evidence they are real circuits (the measures don't share a failure mode).

Measure, per Down_0 weight-action SAE atom:
  STABILITY  -- train the SAE with 4 seeds; stability[a] = mean over other seeds of
                the best decoder-cosine match (does the atom recur?).
  CAUSAL     -- dCE when the atom is knocked out of the reconstruction (sample of
                active atoms; full-model forward).
  MONOSEM    -- KL of the atom's top-activating tokens from the base token
                distribution (753 metric; higher = more selective/nameable).
Then the CONVERGENCE test: correlations among (stability, causal, monosem) on the
sampled active atoms, and stable-vs-unstable group means. Shuffled-atom null.

REGISTERED PREDICTIONS:
  (0) SANITY: some atoms are stable (best-match cos > 0.8 in >=3/4 seeds);
  (a) CONVERGENCE: STABLE atoms are MORE causally important AND MORE monosemantic
      than unstable ones (both group-mean gaps positive; Spearman rho(stability,
      causal) >= 0.25 and rho(stability, monosem) >= 0.25) -- convergent validity,
      so stability is a usable ground-truth-free filter for good circuits;
  (b) report the three pairwise correlations + stable/unstable group means;
  NULL: shuffling atom identities destroys the correlations (rho ~ 0)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; HID = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'stability_scorecard_results.json'
NFIT = 48; NEVAL = 24; P = 512; K = 32; NSEED = 4; DCE_SAMPLE = 64; TOPN = 150
SUB = {'d0': None}; KNOCK = {'mask': None}


def topk(pre, k):
    val, idx = pre.topk(k, dim=1); z = torch.zeros_like(pre); z.scatter_(1, idx, F.relu(val)); return z


def hook_d0(mo, i_, o_):
    return o_ if SUB['d0'] is None else SUB['d0'](i_[0].reshape(-1, HID)).reshape(o_.shape).to(o_.dtype)


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
def capture(rows, n, want_tokens=False):
    cap = []; toks = []
    h = m.transformer.h[0].mlp.Down.register_forward_hook(lambda mo, i_, o_: cap.append(i_[0].detach().float().reshape(-1, HID)))
    for i in range(0, n, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous()
        if want_tokens: toks.append(idx.reshape(-1).cpu())
        forward_logits(idx)
    h.remove()
    g = torch.cat(cap, 0)
    return (g, torch.cat(toks).numpy()) if want_tokens else g


def sub_d0(Dm, Em, b):
    def fn(g):
        z = topk(g @ Em.T, K)
        if KNOCK['mask'] is not None: z = z * KNOCK['mask']
        return z @ Dm.T + b
    return fn


def train_sae(Xin, Ytrue, seed):
    torch.manual_seed(seed)
    Dm = (torch.randn(D, P, device=DEV)/np.sqrt(D)).requires_grad_(True)
    Em = (torch.randn(P, HID, device=DEV)/np.sqrt(HID)).requires_grad_(True)
    b = Ytrue.mean(0).clone().requires_grad_(True); opt = torch.optim.Adam([Dm, Em, b], lr=3e-3)
    for s in range(700):
        z = topk(Xin @ Em.T, K); loss = F.mse_loss(z @ Dm.T + b, Ytrue)
        opt.zero_grad(); loss.backward(); opt.step()
    return Dm.detach(), Em.detach(), b.detach()


def kl_selectivity(code_col, toks, base, topn=TOPN):
    idx = np.argsort(-code_col)[:topn]; idx = idx[code_col[idx] > 0]
    if len(idx) < 10: return 0.0
    c = Counter(toks[idx].tolist()); tot = len(idx); kl = 0.0
    for t, cnt in c.items():
        pt = cnt/tot; kl += pt*np.log(pt/max(base.get(t, 1e-9), 1e-9))
    return float(kl)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NFIT + NEVAL); fit, ev = rows[:NFIT], rows[NFIT:NFIT+NEVAL]
    W0 = m.transformer.h[0].mlp.Down.weight.data.float().to(DEV)
    h0 = m.transformer.h[0].mlp.Down.register_forward_hook(hook_d0)
    g0 = capture(fit, NFIT); Y0 = g0 @ W0.T
    g0_ev, toks = capture(ev, NEVAL, want_tokens=True)
    base = Counter(toks.tolist()); N = len(toks); base = {t: c/N for t, c in base.items()}

    # train NSEED SAEs
    saes = []
    for s in range(NSEED):
        with torch.enable_grad(): saes.append(train_sae(g0, Y0, seed=s))
    Dref, Eref, bref = saes[0]

    # STABILITY: best cross-seed decoder-cosine match per ref atom
    Dref_n = F.normalize(Dref, dim=0)
    match = np.zeros((P, NSEED-1))
    for si, (Ds, _, _) in enumerate(saes[1:]):
        cos = (Dref_n.T @ F.normalize(Ds, dim=0)).abs()      # (P, P)
        match[:, si] = cos.max(1).values.cpu().numpy()
    stability = match.mean(1)                                 # (P,)
    n_stable = int((( match > 0.8).sum(1) >= 3).sum())
    print(f'trained {NSEED} seeds. mean stability {stability.mean():.3f}  atoms stable(>0.8 in>=3) {n_stable}', flush=True)

    # MONOSEM (KL) per atom, and usage, on eval
    SUB['d0'] = sub_d0(Dref, Eref, bref); KNOCK['mask'] = None
    codes = topk(g0_ev @ Eref.T, K).cpu().numpy()            # (Nev, P)
    usage = (codes > 1e-6).mean(0)
    monosem = np.array([kl_selectivity(codes[:, a], toks, base) for a in range(P)])
    active = np.where(usage > 0)[0]

    # CAUSAL dCE on a sample of active atoms (spread across stability range)
    samp = active[np.argsort(-stability[active])]            # order by stability
    samp = np.concatenate([samp[:DCE_SAMPLE//2], samp[-DCE_SAMPLE//2:]])   # top + bottom stability
    samp = np.unique(samp)
    ce0 = ce_on(ev, NEVAL); dce = {}
    for a in samp.tolist():
        mask = torch.ones(P, device=DEV); mask[a] = 0.0; KNOCK['mask'] = mask
        dce[a] = ce_on(ev, NEVAL) - ce0
    KNOCK['mask'] = None; h0.remove()
    sa = samp; st = stability[sa]; ca = np.array([dce[a] for a in sa.tolist()]); mo = monosem[sa]

    rho_sc = spearman(st, ca); rho_sm = spearman(st, mo); rho_cm = spearman(ca, mo)
    med = np.median(st); stable = st >= med
    grp = {'causal_stable': float(ca[stable].mean()), 'causal_unstable': float(ca[~stable].mean()),
           'monosem_stable': float(mo[stable].mean()), 'monosem_unstable': float(mo[~stable].mean())}
    # shuffled null
    g = np.random.RandomState(0); rho_null = spearman(st, ca[g.permutation(len(ca))])
    print(f'(A) convergence: rho(stability,causal) {rho_sc:.3f}  rho(stability,monosem) {rho_sm:.3f}  '
          f'rho(causal,monosem) {rho_cm:.3f}  null {rho_null:.3f}', flush=True)
    print(f'    stable vs unstable: causal {grp["causal_stable"]:.4f}/{grp["causal_unstable"]:.4f}  '
          f'monosem {grp["monosem_stable"]:.3f}/{grp["monosem_unstable"]:.3f}', flush=True)

    p0 = n_stable > 0
    pa = (grp['causal_stable'] > grp['causal_unstable'] and grp['monosem_stable'] > grp['monosem_unstable']
          and rho_sc >= 0.25 and rho_sm >= 0.25)
    null_ok = abs(rho_null) < 0.15
    out = {'n_seed': NSEED, 'mean_stability': round(float(stability.mean()), 4), 'n_stable': n_stable,
           'n_active': int(len(active)), 'dce_sample': int(len(sa)), 'ce0': round(ce0, 4),
           'rho_stability_causal': round(rho_sc, 4), 'rho_stability_monosem': round(rho_sm, 4),
           'rho_causal_monosem': round(rho_cm, 4), 'rho_null': round(rho_null, 4), 'group_means': {k: round(v, 4) for k, v in grp.items()},
           'pred_0': bool(p0), 'pred_a': bool(pa), 'null_ok': bool(null_ok), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) CONVERGENCE (stable atoms more causal AND more monosemantic): {pa}; NULL shuffled~0: {null_ok}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
