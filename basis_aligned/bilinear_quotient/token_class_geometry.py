"""TOKEN-CLASS GEOMETRY (user insight: the token-conditional-mean subspace can't be
re-encoding the CURRENT token -- that's already in the stream -- so it must be the
COMPUTED CLASS / relative structure). Test whether mlp0's per-token output (the
"mean table") organises tokens by GRAMMATICAL CLASS more than the raw embedding.
  Q1: are some means THE SAME? -> effective rank / clustering of the mean table vs
      the embedding table; do many tokens collapse to shared class means?
  Q2: relative geometry vs the embedding -> for labelled classes (determiners,
      numbers, punctuation, pronouns, prepositions), within-class vs across-class
      similarity in the MEAN table vs the EMBEDDING table (cosine AND norm), and how
      much of the mean is a LINEAR transform of the embedding (ridge R2).

REGISTERED PREDICTIONS:
  (0) SANITY: mean table has lower effective rank than the number of tokens;
  (a) CLASS COLLAPSE: the mean table's effective rank is much lower than the
      embedding's (many tokens share a class mean) -- tokens collapse by class;
  (b) SHARPER CLASSES: within-class minus across-class cosine is LARGER in the mean
      table than in the embedding for the labelled classes (mlp0 sharpens class
      geometry the embedding lacks); the mean is NOT fully a linear transform of the
      embedding (ridge R2 < 0.9 -> nonlinear class computation);
  NULL: random token groups show ~0 within-minus-across separation in both."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'token_class_geometry_results.json'
NEVAL = 96; MINCOUNT = 8


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


def eff_rank(X):
    Xc = X - X.mean(0, keepdim=True)
    s = torch.linalg.svdvals(Xc); s2 = s**2
    return float((s2.sum()**2)/(s2**2).sum())          # participation ratio


def dec(t):
    try: return cl.d1(int(t))
    except Exception: return None


CLASSES = {
    'determiner': {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'his', 'her', 'its', 'their', 'our', 'my', 'your'},
    'number': {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'hundred', 'thousand', 'million'},
    'punct': {'.', ',', '!', '?', ';', ':', '(', ')', '[', ']', '"', "'", '--', '-'},
    'pronoun': {'it', 'he', 'she', 'they', 'we', 'you', 'i', 'him', 'them', 'us', 'me'},
    'prep': {'in', 'on', 'at', 'of', 'to', 'for', 'with', 'by', 'from', 'into', 'over', 'under', 'about', 'after', 'before'},
}


def classify(tokid):
    s = dec(tokid)
    if s is None: return None
    w = s.strip().lower()
    for cls, members in CLASSES.items():
        if w in members: return cls
    return None


def within_across(V, labels):
    Vn = F.normalize(V, dim=1)
    classes = sorted(set(labels))
    idx = {c: [i for i, l in enumerate(labels) if l == c] for c in classes}
    win = []; acr = []
    for c in classes:
        ii = idx[c]
        if len(ii) < 2: continue
        sub = Vn[ii]; cs = (sub @ sub.T)
        win.append(float(cs[np.triu_indices(len(ii), 1)].mean()))
        others = [j for cc in classes if cc != c for j in idx[cc]]
        acr.append(float((Vn[ii] @ Vn[others].T).mean()))
    return float(np.mean(win)), float(np.mean(acr))


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    O, toks = capture(rows, NEVAL)
    # per-token mean table
    ids = []; means = []
    for t in np.unique(toks):
        mk = toks == t
        if mk.sum() < MINCOUNT: continue
        ids.append(int(t)); means.append(O[mk].mean(0))
    ids = np.array(ids); M = torch.stack(means, 0)                  # (T, D) mean table
    E = m.transformer.wte.weight.data.float().to(DEV)[torch.from_numpy(ids).to(DEV)]  # embeddings, same tokens
    print(f'{len(ids)} tokens with >= {MINCOUNT} occurrences', flush=True)

    # Q1: effective rank / collapse
    er_mean = eff_rank(M); er_emb = eff_rank(E)
    print(f'(a) effective rank: mean-table {er_mean:.1f}  embedding {er_emb:.1f}  (of {len(ids)} tokens)', flush=True)

    # Q2: class geometry
    labels = [classify(t) for t in ids]
    keep = [i for i, l in enumerate(labels) if l is not None]
    Mk = M[keep]; Ek = E[keep]; lab = [labels[i] for i in keep]
    from collections import Counter
    print(f'    labelled tokens: {dict(Counter(lab))}', flush=True)
    mw, ma = within_across(Mk, lab); ew, ea = within_across(Ek, lab)
    sep_mean = mw - ma; sep_emb = ew - ea
    # random-group null
    rng = np.random.RandomState(0); rlab = list(rng.permutation(lab))
    rmw, rma = within_across(Mk, rlab); sep_null = rmw - rma
    print(f'(b) class separation (within-across cos): mean-table {sep_mean:.3f} (w {mw:.2f}/a {ma:.2f}) | '
          f'embedding {sep_emb:.3f} (w {ew:.2f}/a {ea:.2f}) | shuffled-null {sep_null:.3f}', flush=True)

    # norm structure: within-class norm coefficient of variation (do classes share norm?)
    def norm_cv(V, lab):
        nrm = V.norm(dim=1).cpu().numpy(); cvs = []
        for c in set(lab):
            ii = [i for i, l in enumerate(lab) if l == c]
            if len(ii) > 1: cvs.append(float(np.std(nrm[ii])/max(np.mean(nrm[ii]), 1e-9)))
        return float(np.mean(cvs))
    cv_mean = norm_cv(Mk, lab); cv_emb = norm_cv(Ek, lab)

    # linear predictability of mean from embedding
    def ridge_r2(X, Y, ridge=1e-1):
        n = X.shape[0]; ntr = int(n*0.7); g = torch.Generator(device=X.device).manual_seed(0)
        p = torch.randperm(n, generator=g, device=X.device); tr, te = p[:ntr], p[ntr:]
        mx = X[tr].mean(0, keepdim=True); my = Y[tr].mean(0, keepdim=True)
        A = (X[tr]-mx).T @ (X[tr]-mx); A.diagonal().add_(ridge*float(A.diagonal().mean()))
        W = torch.linalg.solve(A, (X[tr]-mx).T @ (Y[tr]-my)); Yh = (X[te]-mx)@W+my
        return float(1 - ((Y[te]-Yh)**2).sum()/((Y[te]-Y[te].mean(0))**2).sum())
    r2 = ridge_r2(E, M)
    print(f'    mean = linear(embedding)? ridge R2 {r2:.3f}  | within-class norm CV: mean {cv_mean:.3f} emb {cv_emb:.3f}', flush=True)

    p0 = er_mean < len(ids)
    pa = er_mean < 0.7*er_emb
    pb = sep_mean > sep_emb + 0.05 and r2 < 0.9
    null_ok = abs(sep_null) < 0.05
    out = {'n_tokens': len(ids), 'eff_rank_mean': round(er_mean, 2), 'eff_rank_emb': round(er_emb, 2),
           'class_sep_mean': round(sep_mean, 4), 'class_sep_emb': round(sep_emb, 4), 'class_sep_null': round(sep_null, 4),
           'within_mean': round(mw, 4), 'across_mean': round(ma, 4), 'within_emb': round(ew, 4), 'across_emb': round(ea, 4),
           'mean_from_emb_r2': round(r2, 4), 'norm_cv_mean': round(cv_mean, 4), 'norm_cv_emb': round(cv_emb, 4),
           'pred_0': bool(p0), 'pred_a_collapse': bool(pa), 'pred_b_sharper_classes': bool(pb), 'null_ok': bool(null_ok),
           'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) class collapse (mean-rank << emb-rank): {pa}; (b) sharper classes + nonlinear: {pb}; NULL: {null_ok}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
