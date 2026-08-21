"""SEMANTIC SUBSPACE -- is there a SEED-FREE, canonical, interpretable, causal
structure in mlp0's output, independent of the unstable SAE (766 pivot)? The
token-conditional-mean directions of mlp0's OUTPUT are a property of the MODEL+DATA,
not of any SAE fit -- canonical by construction. Test whether this token-semantic
subspace is (a) LOW-RANK, (b) CAUSALLY important (removing it hurts CE more than a
random same-rank subspace), (c) DATA-STABLE (recurs across data splits), and (d)
whether the SAE's fitted subspace ALIGNS with it (so the SAE's instability is just
arbitrary rotation within a canonical semantic subspace).

REGISTERED PREDICTIONS:
  (0) SANITY: token-conditional means separate (nonzero semantic variance);
  (a) LOW-RANK + CAUSAL: <= 64 semantic directions capture >=90% of token-mean
      variance, AND projecting them OUT of mlp0's output hurts CE >= 2x more than
      removing a random same-rank subspace (the token-semantic part is causal);
  (b) DATA-STABLE: semantic subspace from two data halves overlaps (top-32
      principal-angle cos >= 0.8) -- canonical, unlike the seed-unstable SAE atoms
      (763: 0.40);
  (c) SAE ALIGNS: the SAE's top-r decoder subspace overlaps the semantic subspace
      well above a random subspace -> the SAE fits (a rotation of) the semantic
      structure;
  NULL: a random same-rank subspace is neither causal-heavy nor data-stable."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; HID = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'semantic_subspace_results.json'
NEVAL = 48; P = 512; K = 32; MINCOUNT = 5; RSEM = 64
PROJ = {'U': None}


def topk(pre, k):
    val, idx = pre.topk(k, dim=1); z = torch.zeros_like(pre); z.scatter_(1, idx, F.relu(val)); return z


def mlp0_proj_hook(mo, i_, o_):
    # remove span(PROJ['U']) from mlp0 output (o_ is (B,T,D))
    if PROJ['U'] is None: return o_
    U = PROJ['U']; sh = o_.shape; o = o_.reshape(-1, D).float()
    o = o - (o @ U) @ U.T
    return o.reshape(sh).to(o_.dtype)


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


@torch.no_grad()
def capture_gate(rows, n):
    cap = []
    h = m.transformer.h[0].mlp.Down.register_forward_hook(lambda mo, i_, o_: cap.append(i_[0].detach().float().reshape(-1, HID)))
    for i in range(0, n, 4): forward_logits(rows[i:i+4, :257].to(DEV)[:, :-1].contiguous())
    h.remove(); return torch.cat(cap, 0)


def token_mean_dirs(O, toks):
    """centered token-conditional means -> right singular vectors (semantic directions in D)."""
    gmean = O.mean(0, keepdim=True); rows_m = []; wts = []
    c = Counter(toks.tolist())
    for t, cnt in c.items():
        if cnt < MINCOUNT: continue
        mask = torch.from_numpy(toks == t).to(O.device)
        rows_m.append(O[mask].mean(0) - gmean.squeeze(0)); wts.append(np.sqrt(cnt))
    M = torch.stack(rows_m, 0) * torch.tensor(wts, device=O.device, dtype=O.dtype)[:, None]
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    return Vh, S                                          # Vh (k, D) directions; S singular values


def sub_overlap(A, B):
    return float(torch.linalg.svdvals(A.T @ B).mean())


def train_sae(Xin, Ytrue, seed=0):
    torch.manual_seed(seed)
    Dm = (torch.randn(D, P, device=DEV)/np.sqrt(D)).requires_grad_(True)
    Em = (torch.randn(P, HID, device=DEV)/np.sqrt(HID)).requires_grad_(True)
    b = Ytrue.mean(0).clone().requires_grad_(True); opt = torch.optim.Adam([Dm, Em, b], lr=3e-3)
    for s in range(700):
        z = topk(Xin @ Em.T, K); loss = F.mse_loss(z @ Dm.T + b, Ytrue)
        opt.zero_grad(); loss.backward(); opt.step()
    return Dm.detach()


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    h0 = m.transformer.h[0].mlp.register_forward_hook(mlp0_proj_hook)
    O, toks = capture_out(rows, NEVAL)
    Vh, S = token_mean_dirs(O, toks)
    ev = (S**2).cumsum(0)/(S**2).sum(); rank90 = int((ev < 0.9).sum().item()) + 1
    Usem = Vh[:RSEM].T.contiguous()                       # (D, RSEM) semantic subspace
    print(f'semantic variance rank90 {rank90}  (using top-{RSEM})', flush=True)

    # (a) causal: project semantic OUT of mlp0 vs random same-rank
    PROJ['U'] = None; ce_full = ce_on(rows, NEVAL)
    PROJ['U'] = Usem; ce_sem = ce_on(rows, NEVAL)
    g = torch.Generator(device=DEV).manual_seed(0)
    Ur = torch.linalg.qr(torch.randn(D, RSEM, generator=g, device=DEV))[0]
    PROJ['U'] = Ur; ce_rand = ce_on(rows, NEVAL); PROJ['U'] = None
    d_sem = ce_sem - ce_full; d_rand = ce_rand - ce_full
    print(f'(a) CE: full {ce_full:.3f}  -semantic {ce_sem:.3f} (dCE {d_sem:.3f})  -random {ce_rand:.3f} (dCE {d_rand:.3f})  ratio {d_sem/max(d_rand,1e-6):.2f}', flush=True)

    # (b) data-stability: two halves
    half = O.shape[0]//2
    Va, _ = token_mean_dirs(O[:half], toks[:half]); Vb, _ = token_mean_dirs(O[half:], toks[half:])
    r = min(32, Va.shape[0], Vb.shape[0]); data_ov = sub_overlap(Va[:r].T, Vb[:r].T)
    Rr = torch.linalg.qr(torch.randn(D, r, generator=g, device=DEV))[0]
    rand_data_ov = sub_overlap(Va[:r].T, Rr)
    print(f'(b) data-split semantic overlap (top-{r}) {data_ov:.3f} (random {rand_data_ov:.3f}; SAE atoms 763: 0.40)', flush=True)

    # (c) SAE alignment
    W0 = m.transformer.h[0].mlp.Down.weight.data.float().to(DEV)
    gate = capture_gate(rows, NEVAL)
    with torch.enable_grad(): Dsae = train_sae(gate, gate @ W0.T, 0)
    Qsae = torch.linalg.svd(Dsae, full_matrices=False)[0][:, :RSEM]
    sae_ov = sub_overlap(Qsae, Usem)
    Rr2 = torch.linalg.qr(torch.randn(D, RSEM, generator=g, device=DEV))[0]
    sae_rand = sub_overlap(Qsae, Rr2)
    print(f'(c) SAE subspace vs semantic overlap {sae_ov:.3f} (random {sae_rand:.3f})', flush=True)
    h0.remove()

    p0 = float(S[0]) > 0
    pa = rank90 <= 64 and d_sem >= 2*max(d_rand, 1e-6)
    pb = data_ov >= 0.8
    pc = sae_ov > sae_rand + 0.15
    null_ok = rand_data_ov < 0.5
    out = {'rank90': rank90, 'rsem': RSEM, 'ce_full': round(ce_full, 4), 'dce_semantic': round(d_sem, 4),
           'dce_random': round(d_rand, 4), 'causal_ratio': round(d_sem/max(d_rand, 1e-6), 3),
           'data_overlap': round(data_ov, 4), 'random_data_overlap': round(rand_data_ov, 4),
           'sae_semantic_overlap': round(sae_ov, 4), 'sae_random_overlap': round(sae_rand, 4),
           'pred_0': bool(p0), 'pred_a_lowrank_causal': bool(pa), 'pred_b_data_stable': bool(pb),
           'pred_c_sae_aligns': bool(pc), 'null_ok': bool(null_ok), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) low-rank+causal: {pa}; (b) data-stable: {pb}; (c) SAE aligns: {pc}; NULL: {null_ok}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
