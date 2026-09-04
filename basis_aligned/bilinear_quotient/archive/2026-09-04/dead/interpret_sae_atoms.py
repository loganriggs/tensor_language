"""INTERPRET SAE ATOMS (the interpretability payoff: are the learned sparse
atoms nameable circuits?). Train a top-k SAE on mlp1's output; for the most-
used atoms, report (i) which CURRENT tokens activate it (top-|code|
datapoints -> their tokens), (ii) what it WRITES (atom direction ->
unembedding readout, rough since mlp1 is 16 blocks upstream). Test whether
SAE atoms are MORE MONOSEMANTIC than SVD directions: monosemanticity =
identity-concentration of the tokens that most activate the atom (does a few
token types dominate its firing?).

REGISTERED PREDICTIONS:
  (0) SANITY: atoms have non-trivial activation;
  (a) SAE ATOMS MORE MONOSEMANTIC than SVD: mean top-activation token-
      concentration of SAE atoms > SVD directions (> by >= 0.1), and several
      atoms have a human-readable firing token set;
  (b) report example atoms (top firing tokens + top writes) + concentration
      SAE vs SVD;
  NULL: random directions have LOW concentration (~ SVD or below)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
from collections import Counter

D = 1152; LAYER = 1
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'interpret_sae_atoms_results.json'
NFIT = 96; K = 32; P = 512; STEPS = 700; TOPN_ATOMS = 12


@torch.no_grad()
def capture(rows, n):
    cap = []; toks = []
    h = m.transformer.h[LAYER].mlp.register_forward_hook(lambda mo,i_,o_: cap.append(o_.detach().float().reshape(-1,D)))
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); toks.append(idx.reshape(-1).cpu())
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
    h.remove(); return torch.cat(cap,0), torch.cat(toks).numpy()


def topk(pre, k):
    val, idx = pre.topk(k, dim=1); z = torch.zeros_like(pre); z.scatter_(1, idx, F.relu(val)); return z


def train_sae(O, k, P, steps=STEPS, seed=0):
    torch.manual_seed(seed)
    We=(torch.randn(D,P,device=DEV)/np.sqrt(D)).requires_grad_(True)
    Wd=(torch.randn(P,D,device=DEV)/np.sqrt(P)).requires_grad_(True)
    b=O.mean(0).clone().requires_grad_(True); opt=torch.optim.Adam([We,Wd,b],lr=2e-3)
    for s in range(steps):
        z=topk((O-b)@We,k); loss=F.mse_loss(z@Wd+b,O); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        z=topk((O-b)@We,k)
    return We.detach(), Wd.detach(), b.detach(), z


def d1(t):
    try: return cl.d1(int(t))
    except Exception: return f'<{t}>'


def concentration(code_col, toks, topn=200):
    idx = np.argsort(-code_col)[:topn]
    c = Counter(toks[idx].tolist())
    return c.most_common(1)[0][1]/topn, [d1(t) for t,_ in c.most_common(6)]


@torch.no_grad()
def main():
    t0=time.time(); cl.use_state(PT+'census_state_diverse.pt')
    rows = cl.fineweb_rows(NFIT)
    O, toks = capture(rows, NFIT)
    with torch.enable_grad(): We, Wd, b, Z = train_sae(O, K, P)
    Zc = Z.cpu().numpy(); usage = (Zc>1e-6).mean(0)
    W_U = m.lm_head.weight.data.float().to(DEV)

    top_atoms = np.argsort(-usage)[:TOPN_ATOMS]
    sae_conc = []; atoms = []
    for a in top_atoms:
        conc, toptok = concentration(Zc[:,a], toks)
        writes = [d1(t) for t in torch.argsort(-(W_U @ (Wd[a]/Wd[a].norm())))[:6].cpu().numpy()]
        sae_conc.append(conc); atoms.append({'atom': int(a), 'usage': round(float(usage[a]),3),
                        'concentration': round(conc,3), 'fires_on': toptok, 'writes': writes})
        print(f'atom {a}: usage {usage[a]:.2f} conc {conc:.2f}  fires {toptok[:4]}  writes {writes[:3]}', flush=True)

    # SVD directions concentration (project outputs onto SVD dirs)
    U = torch.linalg.svd(O - O.mean(0), full_matrices=False)[2][:TOPN_ATOMS]   # (TOPN, D)
    svd_codes = ((O - O.mean(0)) @ U.T).cpu().numpy()                          # (N, TOPN)
    svd_conc = [concentration(np.abs(svd_codes[:,j]), toks)[0] for j in range(TOPN_ATOMS)]
    g = torch.Generator().manual_seed(0); Rr = torch.linalg.qr(torch.randn(D, TOPN_ATOMS, generator=g))[0].to(DEV)
    rand_codes = ((O-O.mean(0)) @ Rr).cpu().numpy()
    rand_conc = [concentration(np.abs(rand_codes[:,j]), toks)[0] for j in range(TOPN_ATOMS)]

    sm, vm, rm = float(np.mean(sae_conc)), float(np.mean(svd_conc)), float(np.mean(rand_conc))
    print(f'\nmean top-activation token-concentration: SAE {sm:.3f}  SVD {vm:.3f}  random {rm:.3f}', flush=True)
    pa = sm > vm + 0.1; null_ok = rm <= vm + 0.05
    print(f'(a) SAE atoms more monosemantic than SVD: {pa}; NULL random low: {null_ok}', flush=True)
    out = {'K':K,'P':P,'atoms':atoms,'sae_mean_conc':round(sm,3),'svd_mean_conc':round(vm,3),
           'rand_mean_conc':round(rm,3),'pred_a':bool(pa),'null_ok':bool(null_ok),'runtime_s':time.time()-t0}
    json.dump(out, open(OUT,'w'), indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
