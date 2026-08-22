"""FOLD layer 1's bilinear form onto layer-0's computed features (user's 2nd directive; gets SPECIFICS
behind §849's re-expansion). §849: layer 0 collapses tokens to class clusters (eff-dim 20), layer 1
RE-EXPANDS (47). What does layer 1 read to re-expand? Decompose mlp1's hidden-unit readouts (Left_k,
Right_k — vectors in the layer-1 INPUT space) onto known feature subspaces built from that same input:
  - CLASS: grammatical-class (8-way) conditional means (coarse, ~7 dim)
  - TOKEN-FINE: current-token-identity conditional means, ORTHOGONALIZED against class (fine token
    distinctions beyond the coarse class — the thing a re-expansion would re-read)
  - PREV: previous-token conditional means (orthogonalized against token+class)
  - POS: position conditional means
For each class-writing mlp1 unit, report the fraction of its readout energy in each subspace, vs the
CHANCE fraction (rank/D for a random vector) — ratio>1 means the readout preferentially reads that
feature.

REGISTERED PREDICTIONS:
  (0) SANITY: energy fractions sum to <=1 with a residual; random-vector fractions ~ rank/D;
  (a) if mlp1 readouts carry TOKEN-FINE energy well above chance, layer 1 re-reads finer token identity
      (beyond coarse class) — explaining the re-expansion (§849), specifics via folding;
  (b) report class / token-fine / prev / pos energy ratios (actual/chance) for Left and Right;
      whichever features are >1 are what layer 1 folds in."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp1_folding_results.json'
NEVAL = 250; MINCOUNT = 8; RTOK = 64; RPREV = 64; RPOS = 32; NCLASS_DIR = 12; NUNIT = 24
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'his', 'her', 'its', 'their', 'our', 'my', 'your', 'some', 'any', 'no', 'every', 'each'}
PREP = {'of', 'in', 'to', 'for', 'on', 'at', 'by', 'with', 'from', 'as', 'into', 'about', 'over', 'after', 'before', 'between', 'through', 'under', 'against'}
CONJ = {'and', 'or', 'but', 'nor', 'so', 'yet', 'because', 'although', 'while', 'if', 'than'}
PRON = {'he', 'she', 'it', 'they', 'we', 'you', 'i', 'him', 'her', 'them', 'us', 'me', 'who', 'which'}


def dec():
    import tiktoken; enc = tiktoken.get_encoding('gpt2'); return lambda i: enc.decode([int(i)])


def classify(s):
    t = s.strip()
    if t == '' or not t[0].isalnum(): return 'punct'
    if t[0].isdigit(): return 'number'
    low = t.lower()
    if low in DET: return 'det'
    if low in PREP: return 'prep'
    if low in CONJ: return 'conj'
    if low in PRON: return 'pron'
    if t[0].isupper(): return 'cap'
    return 'word'


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def capture(rows):
    mlp = m.transformer.h[1].mlp; Xs = []; Os = []; seqs = []
    def pre(mo, a): Xs.append(a[0].detach().float().reshape(-1, D))
    def post(mo, i_, o_): Os.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hp = mlp.register_forward_pre_hook(pre); ho = mlp.register_forward_hook(post)
    for i in range(0, NEVAL, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx); seqs.append(idx.cpu().numpy())
    hp.remove(); ho.remove(); return torch.cat(Xs, 0), torch.cat(Os, 0), np.concatenate(seqs, 0)


def mean_subspace(X, labels, r):
    g = X.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(labels):
        if t < 0: continue
        mk = labels == t
        if mk.sum() < MINCOUNT: continue
        rows.append(X[mk].mean(0) - g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0) * torch.tensor(wt, device=X.device, dtype=X.dtype)[:, None]
    k = min(r, M.shape[0])
    return torch.linalg.svd(M, full_matrices=False)[2][:k].T.contiguous()


def orth_against(U, *bases):
    """remove from U the span of bases; return orthonormal residual basis."""
    B = torch.cat([b for b in bases if b is not None and b.shape[1] > 0], 1)
    Q = torch.linalg.qr(B)[0]
    Ur = U - Q @ (Q.T @ U)
    keep = Ur.norm(dim=0) > 1e-3
    if keep.sum() == 0: return U[:, :0]
    return torch.linalg.qr(Ur[:, keep])[0]


def energy_frac(vecs, U):
    """mean over vecs of ||proj_U(v)||^2/||v||^2."""
    if U.shape[1] == 0: return 0.0
    p = (vecs @ U)
    return float((p.pow(2).sum(1) / (vecs.pow(2).sum(1) + 1e-9)).mean())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL); d = dec()
    mlp = m.transformer.h[1].mlp
    Lw = mlp.Left.weight.detach().float(); Rw = mlp.Right.weight.detach().float(); Dw = mlp.Down.weight.detach().float()
    X, O, seqs = capture(rows)
    toks = seqs.reshape(-1); prev = np.full_like(seqs, -1); prev[:, 1:] = seqs[:, :-1]; prev = prev.reshape(-1)
    pos = np.broadcast_to(np.arange(seqs.shape[1]), seqs.shape).reshape(-1)
    clslab = np.array([CLASSES.index(classify(d(int(t)))) for t in toks])
    # feature subspaces in the layer-1 INPUT space
    Uclass = mean_subspace(X, clslab, NCLASS_DIR)
    Utok_raw = mean_subspace(X, toks, RTOK)
    Utokfine = orth_against(Utok_raw, Uclass)                       # token id beyond coarse class
    Uprev_raw = mean_subspace(X, prev, RPREV)
    Uprev = orth_against(Uprev_raw, Uclass, Utok_raw)              # prev beyond current token+class
    Upos = mean_subspace(X, pos.astype(np.int64), RPOS)
    feats = {'class': Uclass, 'token_fine': Utokfine, 'prev': Uprev, 'pos': Upos}
    ranks = {k: v.shape[1] for k, v in feats.items()}
    chance = {k: round(v/D, 4) for k, v in ranks.items()}
    # class-writing units
    Ucls_out = mean_subspace(O, toks, NCLASS_DIR)                  # class dirs in OUTPUT space
    unit_mag = (Ucls_out.T @ Dw).norm(dim=0); top = torch.topk(unit_mag, NUNIT).indices.tolist()
    Lk = Lw[top]; Rk = Rw[top]                                     # (NUNIT, D)
    Lk = Lk / (Lk.norm(dim=1, keepdim=True) + 1e-9); Rk = Rk / (Rk.norm(dim=1, keepdim=True) + 1e-9)
    res = {}
    for name, U in feats.items():
        lf = energy_frac(Lk, U); rf = energy_frac(Rk, U)
        res[name] = {'rank': ranks[name], 'chance': chance[name],
                     'left_frac': round(lf, 4), 'right_frac': round(rf, 4),
                     'left_ratio': round(lf/max(chance[name], 1e-9), 2), 'right_ratio': round(rf/max(chance[name], 1e-9), 2)}
    # random-vector null for calibration
    g = torch.Generator(device=DEV).manual_seed(0); RND = torch.randn(NUNIT, D, generator=g, device=DEV); RND = RND/RND.norm(dim=1, keepdim=True)
    rnd = {name: round(energy_frac(RND, U)/max(chance[name], 1e-9), 2) for name, U in feats.items()}
    out = {'ranks': ranks, 'random_vector_ratio': rnd, 'per_feature': res, 'runtime_s': round(time.time()-t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print("mlp1 readout energy in layer-0 features (ratio actual/chance; >1 = preferentially reads):", flush=True)
    for name, r in res.items():
        print(f"  {name} (rank {r['rank']}): Left x{r['left_ratio']} Right x{r['right_ratio']} (random-vec x{rnd[name]})", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
