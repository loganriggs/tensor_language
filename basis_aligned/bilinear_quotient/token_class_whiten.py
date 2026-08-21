"""TOKEN-CLASS WHITENED (clean version of 780's Q2, removing the baseline-similarity
confound). 780: the mlp0 per-token mean table is a low-rank CLASS structure (eff-rank
22.7 vs embedding 132.4), but raw within-across COSINE was confounded because the
mean table's dominant shared direction makes all tokens cosine-similar. Re-measure
class separation with a shared-component-ROBUST metric: (i) Fisher discriminant ratio
(between-class / within-class scatter, invariant to a shared additive direction), and
(ii) within-across cosine AFTER removing the top principal direction. Compare the
mlp0 mean table vs the raw embedding for the labelled classes.

REGISTERED PREDICTIONS:
  (0) SANITY: Fisher ratio > 0 for both;
  (a) MLP SHARPENS CLASS: the mean table has a HIGHER Fisher class-separation ratio
      than the embedding (>= 1.3x), and higher after-top-PC-removed cosine
      separation -- once the shared direction is removed, mlp0 groups tokens by
      grammatical class MORE cleanly than the raw embedding (the computed class
      structure 780 predicted);
  (b) report Fisher ratio + de-shared cosine separation, mean vs embedding, per metric;
  NULL: shuffled class labels give Fisher ratio ~ chance (<< real)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'token_class_whiten_results.json'
NEVAL = 96; MINCOUNT = 8

CLASSES = {
    'determiner': {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'his', 'her', 'its', 'their', 'our', 'my', 'your'},
    'number': {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'hundred', 'thousand', 'million'},
    'punct': {'.', ',', '!', '?', ';', ':', '(', ')', '[', ']', '"', "'", '--', '-'},
    'pronoun': {'it', 'he', 'she', 'they', 'we', 'you', 'i', 'him', 'them', 'us', 'me'},
    'prep': {'in', 'on', 'at', 'of', 'to', 'for', 'with', 'by', 'from', 'into', 'over', 'under', 'about', 'after', 'before'},
}


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def capture(rows, n):
    cap = []; toks = []
    h = m.transformer.h[0].mlp.register_forward_hook(lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D)))
    for i in range(0, n, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); toks.append(idx.reshape(-1).cpu())
        forward_logits(idx)
    h.remove(); return torch.cat(cap, 0), torch.cat(toks).numpy()


def classify(tokid):
    try: s = cl.d1(int(tokid))
    except Exception: return None
    w = s.strip().lower()
    for cls, members in CLASSES.items():
        if w in members: return cls
    return None


def fisher_ratio(V, lab):
    classes = sorted(set(lab)); mu = V.mean(0)
    between = 0.0; within = 0.0
    for c in classes:
        ii = [i for i, l in enumerate(lab) if l == c]
        if len(ii) < 2: continue
        sub = V[ii]; muc = sub.mean(0)
        between += len(ii) * float(((muc - mu)**2).sum())
        within += float(((sub - muc)**2).sum())
    return between / max(within, 1e-9)


def deshared_cos_sep(V, lab):
    Vc = V - V.mean(0, keepdim=True)
    U = torch.linalg.svd(Vc, full_matrices=False)[2]           # right singular vecs
    Vd = Vc - (Vc @ U[:1].T) @ U[:1]                           # remove top PC
    Vn = F.normalize(Vd, dim=1); classes = sorted(set(lab))
    idx = {c: [i for i, l in enumerate(lab) if l == c] for c in classes}
    win = []; acr = []
    for c in classes:
        ii = idx[c]
        if len(ii) < 2: continue
        sub = Vn[ii]; win.append(float((sub @ sub.T)[np.triu_indices(len(ii), 1)].mean()))
        others = [j for cc in classes if cc != c for j in idx[cc]]
        acr.append(float((Vn[ii] @ Vn[others].T).mean()))
    return float(np.mean(win) - np.mean(acr))


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    O, toks = capture(rows, NEVAL)
    ids = []; means = []
    for t in np.unique(toks):
        mk = toks == t
        if mk.sum() < MINCOUNT: continue
        ids.append(int(t)); means.append(O[mk].mean(0))
    ids = np.array(ids); M = torch.stack(means, 0)
    E = m.transformer.wte.weight.data.float().to(DEV)[torch.from_numpy(ids).to(DEV)]
    labels = [classify(t) for t in ids]; keep = [i for i, l in enumerate(labels) if l is not None]
    Mk = M[keep]; Ek = E[keep]; lab = [labels[i] for i in keep]
    print(f'{len(lab)} labelled tokens: {dict(Counter(lab))}', flush=True)

    f_mean = fisher_ratio(Mk, lab); f_emb = fisher_ratio(Ek, lab)
    rng = np.random.RandomState(0); rlab = list(rng.permutation(lab)); f_null = fisher_ratio(Mk, rlab)
    ds_mean = deshared_cos_sep(Mk, lab); ds_emb = deshared_cos_sep(Ek, lab)
    print(f'(a) Fisher class-separation ratio: mean-table {f_mean:.3f}  embedding {f_emb:.3f}  (shuffled null {f_null:.3f})  ratio mean/emb {f_mean/max(f_emb,1e-9):.2f}', flush=True)
    print(f'(b) de-shared (top-PC removed) cosine separation: mean-table {ds_mean:.3f}  embedding {ds_emb:.3f}', flush=True)

    pa = f_mean >= 1.3*f_emb and ds_mean > ds_emb
    null_ok = f_null < 0.3*f_mean
    out = {'n_labelled': len(lab), 'fisher_mean': round(f_mean, 4), 'fisher_emb': round(f_emb, 4),
           'fisher_null': round(f_null, 4), 'fisher_ratio_mean_over_emb': round(f_mean/max(f_emb, 1e-9), 3),
           'deshared_sep_mean': round(ds_mean, 4), 'deshared_sep_emb': round(ds_emb, 4),
           'pred_a_mlp_sharpens_class': bool(pa), 'null_ok': bool(null_ok), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) mlp0 sharpens class separation vs embedding (Fisher >=1.3x & de-shared higher): {pa}; NULL: {null_ok}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
