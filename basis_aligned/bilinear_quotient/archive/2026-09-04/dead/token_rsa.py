"""TOKEN RSA (user ask: how different are the RELATIVE relationships between tokens
in the embedding vs the mlp0 mean table, as a scalar -- RSA -- on a LOT more data).
Representational Similarity Analysis: build the token x token dissimilarity matrix
(RDM, 1 - cosine) in embedding space and in mean-table space, and correlate them
(Spearman of the upper triangles). RSA close to 1 = the mean table preserves the
embedding's relative geometry; lower = it REORGANISES relative structure. Split by
whether the token pair is SAME-class or DIFFERENT-class: if the mean table COLLAPSES
within-class relative structure (same-class tokens become indistinguishable
regardless of their embedding differences) but preserves across-class, that is the
class-QUANTISATION signature of a class-computing front end (780/782).

Runs on 512 rows (~131k tokens) for robust per-token means (user: default to more data).

REGISTERED PREDICTIONS:
  (0) SANITY: enough tokens with >= MINCOUNT occurrences (> 500);
  (a) REORGANISED TOWARD CLASS: full RSA(mean, embedding) is only MODERATE (< 0.7)
      -- the mean table changes relative structure -- AND within-class RSA is LOWER
      than across-class RSA (the mean table collapses within-class relative
      distinctions while keeping class distinctions = class quantisation);
  (b) report full/within/across RSA + Fisher class-sep ratio + eff-rank on this
      larger sample (robustness of 780/782);
  NULL: shuffled token identities give within~across RSA (no class effect)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'token_rsa_results.json'
NEVAL = 512; MINCOUNT = 25; MAXTOK = 1500

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
def capture_means(rows, n):
    # memory-safe: accumulate per-token sum + count on GPU into small dicts
    ssum = {}; scnt = {}
    cur = {'buf': None}
    def hk(mo, i_, o_): cur['buf'] = o_.detach().float().reshape(-1, D)
    h = m.transformer.h[0].mlp.register_forward_hook(hk)
    for i in range(0, n, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx)
        O = cur['buf']; toks = idx.reshape(-1)
        for t in torch.unique(toks):
            tid = int(t); mask = toks == t
            s = O[mask].sum(0); c = int(mask.sum())
            if tid in ssum: ssum[tid] += s; scnt[tid] += c
            else: ssum[tid] = s.clone(); scnt[tid] = c
    h.remove()
    ids = [t for t in ssum if scnt[t] >= MINCOUNT]
    ids = sorted(ids, key=lambda t: -scnt[t])[:MAXTOK]
    M = torch.stack([ssum[t]/scnt[t] for t in ids], 0)
    return np.array(ids), M


def classify(tokid):
    try: s = cl.d1(int(tokid))
    except Exception: return None
    w = s.strip().lower()
    for cls, members in CLASSES.items():
        if w in members: return cls
    return None


def rdm(V):
    Vn = F.normalize(V, dim=1)
    return 1 - (Vn @ Vn.T)                       # (T,T) cosine dissimilarity


def spearman_upper(A, B, mask=None):
    iu = torch.triu_indices(A.shape[0], A.shape[0], 1, device=A.device)
    a = A[iu[0], iu[1]]; b = B[iu[0], iu[1]]
    if mask is not None:
        mm = mask[iu[0], iu[1]]; a = a[mm]; b = b[mm]
    if a.numel() < 10: return float('nan')
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float()
    return float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])


def fisher_ratio(V, lab):
    mu = V.mean(0); between = 0.0; within = 0.0
    for c in set(lab):
        ii = [i for i, l in enumerate(lab) if l == c]
        if len(ii) < 2: continue
        sub = V[ii]; muc = sub.mean(0)
        between += len(ii)*float(((muc-mu)**2).sum()); within += float(((sub-muc)**2).sum())
    return between/max(within, 1e-9)


def eff_rank(X):
    Xc = X - X.mean(0, keepdim=True); s2 = torch.linalg.svdvals(Xc)**2
    return float((s2.sum()**2)/(s2**2).sum())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    ids, M = capture_means(rows, NEVAL)
    E = m.transformer.wte.weight.data.float().to(DEV)[torch.from_numpy(ids).to(DEV)]
    print(f'{len(ids)} tokens with >= {MINCOUNT} occurrences (from ~{NEVAL*256} tokens)', flush=True)

    RM = rdm(M); RE = rdm(E)
    rsa_full = spearman_upper(RM, RE)
    # class masks
    labels = np.array([classify(t) for t in ids], dtype=object)
    lab_idx = np.where(labels != None)[0]
    labt = torch.tensor([hash(labels[i]) % 100000 for i in lab_idx], device=DEV)
    Msub = M[torch.from_numpy(lab_idx).to(DEV)]; Esub = E[torch.from_numpy(lab_idx).to(DEV)]
    RMs = rdm(Msub); REs = rdm(Esub)
    same = (labt[:, None] == labt[None, :])
    rsa_within = spearman_upper(RMs, REs, same)
    rsa_across = spearman_upper(RMs, REs, ~same)
    print(f'(a) RSA(mean, embedding): full {rsa_full:.3f} | within-class {rsa_within:.3f} | across-class {rsa_across:.3f}', flush=True)

    lab = [labels[i] for i in lab_idx]
    f_mean = fisher_ratio(Msub, lab); f_emb = fisher_ratio(Esub, lab)
    er_m = eff_rank(M); er_e = eff_rank(E)
    print(f'(b) Fisher class-sep: mean {f_mean:.3f} emb {f_emb:.3f} (ratio {f_mean/max(f_emb,1e-9):.2f}) | eff-rank mean {er_m:.1f} emb {er_e:.1f} (of {len(ids)})', flush=True)

    p0 = len(ids) > 500
    pa = rsa_full < 0.7 and rsa_within < rsa_across
    out = {'n_tokens': len(ids), 'n_labelled': int(len(lab_idx)), 'rsa_full': round(rsa_full, 4),
           'rsa_within_class': round(rsa_within, 4), 'rsa_across_class': round(rsa_across, 4),
           'fisher_mean': round(f_mean, 4), 'fisher_emb': round(f_emb, 4), 'fisher_ratio': round(f_mean/max(f_emb, 1e-9), 3),
           'eff_rank_mean': round(er_m, 2), 'eff_rank_emb': round(er_e, 2),
           'pred_0': bool(p0), 'pred_a_reorganised_toward_class': bool(pa), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) mean table reorganises relative structure toward class (RSA<0.7 & within<across): {pa}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
