"""SEMANTIC ALIGNMENT explains MONOSEMANTICITY (links 763 + 767). 763: an SAE
atom's monosemanticity is orthogonal to its causal importance and stability. 767:
there is a canonical, causal, stable token-SEMANTIC subspace, and the SAE only
half-covers it. Linking question: is an SAE atom monosemantic EXACTLY WHEN its
decoder direction lies in the semantic subspace? If yes, interpretability = semantic
alignment (a distinct axis from the fitted basis) -- which is why 763's convergence
failed (monosem tracks semantic-alignment, not the SAE's arbitrary rotation).

Per Down_0 SAE atom: (i) SEMANTIC ALIGNMENT = ||proj_Usem(decoder_a)|| / ||decoder_a||
(fraction of the atom's write direction inside the seed-free semantic subspace);
(ii) MONOSEM = KL of the atom's top-activating tokens from base. Correlate.

REGISTERED PREDICTIONS:
  (0) SANITY: alignment spread across atoms (not all identical);
  (a) LINK: Spearman rho(semantic_alignment, monosemanticity) >= 0.3 and >> a
      shuffled null -- monosemantic atoms are the semantically-aligned ones, so
      interpretability is explained by semantic-subspace alignment, not by the SAE
      basis (resolving why 763 saw monosem orthogonal to causal/stability);
  (b) report high-alignment vs low-alignment atom mean KL;
  NULL: shuffling atom identities destroys the correlation."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; HID = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'semantic_align_monosem_results.json'
NEVAL = 48; P = 512; K = 32; MINCOUNT = 5; RSEM = 64; TOPN = 150


def topk(pre, k):
    val, idx = pre.topk(k, dim=1); z = torch.zeros_like(pre); z.scatter_(1, idx, F.relu(val)); return z


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def capture(rows, n, module, dim, want_tokens=False):
    cap = []; toks = []
    h = module.register_forward_hook(lambda mo, i_, o_: cap.append(i_[0].detach().float().reshape(-1, dim)))
    for i in range(0, n, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous()
        if want_tokens: toks.append(idx.reshape(-1).cpu())
        forward_logits(idx)
    h.remove(); g = torch.cat(cap, 0)
    return (g, torch.cat(toks).numpy()) if want_tokens else g


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
    return torch.linalg.svd(M, full_matrices=False)[2]        # (k, D)


def train_sae(Xin, Ytrue, seed=0):
    torch.manual_seed(seed)
    Dm = (torch.randn(D, P, device=DEV)/np.sqrt(D)).requires_grad_(True)
    Em = (torch.randn(P, HID, device=DEV)/np.sqrt(HID)).requires_grad_(True)
    b = Ytrue.mean(0).clone().requires_grad_(True); opt = torch.optim.Adam([Dm, Em, b], lr=3e-3)
    for s in range(700):
        z = topk(Xin @ Em.T, K); loss = F.mse_loss(z @ Dm.T + b, Ytrue)
        opt.zero_grad(); loss.backward(); opt.step()
    return Dm.detach(), Em.detach(), b.detach()


def kl_sel(code_col, toks, base, topn=TOPN):
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
    rows = cl.fineweb_rows(NEVAL)
    O, toks = capture_out(rows, NEVAL)
    base = Counter(toks.tolist()); N = len(toks); base = {t: c/N for t, c in base.items()}
    Vh = semantic_dirs(O, toks); Usem = Vh[:RSEM].T.contiguous()      # (D, RSEM)

    W0 = m.transformer.h[0].mlp.Down.weight.data.float().to(DEV)
    gate = capture(rows, NEVAL, m.transformer.h[0].mlp.Down, HID)
    with torch.enable_grad(): Dm, Em, b = train_sae(gate, gate @ W0.T, 0)

    # semantic alignment per atom
    Dn = Dm / Dm.norm(dim=0, keepdim=True).clamp_min(1e-9)   # dim=0 not p=0 (LESSONS: norm 0 = order not axis)
    align = ((Dn.T @ Usem)**2).sum(1).sqrt().cpu().numpy()            # fraction of atom dir in Usem (0..1)
    codes = topk(gate @ Em.T, K).cpu().numpy(); usage = (codes > 1e-6).mean(0)
    active = np.where(usage > 0)[0]
    mono = np.array([kl_sel(codes[:, a], toks, base) for a in active])
    al = align[active]

    rho = spearman(al, mono)
    g = np.random.RandomState(0); rho_null = spearman(al, mono[g.permutation(len(mono))])
    med = np.median(al); hi = al >= med
    kl_hi = float(mono[hi].mean()); kl_lo = float(mono[~hi].mean())
    print(f'atoms {len(active)} active | mean semantic-alignment {al.mean():.3f}', flush=True)
    print(f'(a) rho(alignment, monosem) {rho:.3f} (shuffled null {rho_null:.3f})', flush=True)
    print(f'(b) high-alignment atom KL {kl_hi:.3f} vs low-alignment {kl_lo:.3f}', flush=True)

    p0 = al.std() > 0.02
    pa = rho >= 0.3 and rho - rho_null >= 0.2
    null_ok = abs(rho_null) < 0.15
    out = {'n_active': int(len(active)), 'mean_alignment': round(float(al.mean()), 4),
           'rho_align_monosem': round(rho, 4), 'rho_null': round(rho_null, 4),
           'kl_high_align': round(kl_hi, 4), 'kl_low_align': round(kl_lo, 4),
           'pred_0': bool(p0), 'pred_a_link': bool(pa), 'null_ok': bool(null_ok), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) monosem explained by semantic alignment: {pa}; NULL shuffled~0: {null_ok}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
