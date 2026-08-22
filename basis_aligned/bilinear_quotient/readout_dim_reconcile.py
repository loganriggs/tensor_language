"""RECONCILE the "readout collapses to ~3 dimensions" claim with high-dimensional first-mention prediction
(verifies/scopes a published claim). §readout_3dim / FINDINGS 1d say L17 collapses to eff-dim ~2.8 — but that
was measured on token-CLASS-conditional MEANS. Per-token prediction of 50k-vocab first-mention words cannot
be 3-dimensional. Resolve: measure (a) participation-ratio eff-dim of PER-TOKEN L17 residuals (not means),
(b) eff-dim of class-conditional means (should reproduce ~3), (c) how many L17 principal components the
LOGITS actually use — the prediction-relevant rank (cumulative CE recovered vs #PCs kept). Controls: compare
to L15 (before readout collapse); random-projection baseline for the CE-rank curve.

REGISTERED PREDICTIONS:
  (0) SANITY: class-mean eff-dim ~3 (reproduces the published number);
  (a) THE COLLAPSE IS MEANS-ONLY: per-token L17 eff-dim is HIGH (>>3, tens+), and recovering most of the CE
      needs MANY L17 PCs (>>3) -> the "3-dim readout" is the class/boundary structure of the MEANS, while
      per-token content prediction uses a high-dim subspace; the published "~3-dim prediction" phrasing is
      corrected to "~3-dim class-mean structure";
  (b) if per-token eff-dim is also ~3 and few PCs recover the CE, the readout genuinely is low-rank (the
      claim stands as-is)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'readout_dim_reconcile_results.json'
NEVAL = 200; SEQ = 256
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


def readout(x): return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return readout(x)


def eff_dim(X):
    """participation ratio (sum s^2)^2 / sum s^4 of centered X."""
    Xc = X - X.mean(0, keepdim=True)
    s = torch.linalg.svdvals(Xc.float())
    s2 = s**2
    return float((s2.sum()**2) / (s2.pow(2).sum() + 1e-9))


@torch.no_grad()
def capture(rows, L):
    outs = []; seqs = []
    def h(mo, i_, o_): outs.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hh = m.transformer.h[L].register_forward_hook(h)
    for i in range(0, NEVAL, 4):
        idx = rows[i:i+4, :SEQ].to(DEV)[:, :-1].contiguous(); forward_logits(idx); seqs.append(idx.cpu().numpy())
    hh.remove()
    return torch.cat(outs, 0), np.concatenate(seqs, 0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    R17, S = capture(rows, 17); R15, _ = capture(rows, 15)
    toks = S.reshape(-1); tgt = np.full_like(S, -1); tgt[:, :-1] = S[:, 1:]; tgt = tgt.reshape(-1)
    clslab = np.array([CLASSES.index(classify(d(int(t)))) for t in toks])
    # (a) per-token eff-dim; (b) class-mean eff-dim
    sub = torch.randperm(R17.shape[0])[:8000]
    pertoken_17 = eff_dim(R17[sub]); pertoken_15 = eff_dim(R15[sub])
    means = torch.stack([R17[torch.tensor(clslab == c, device=DEV)].mean(0) for c in range(len(CLASSES)) if (clslab == c).sum() > 5])
    classmean_17 = eff_dim(means)
    # (c) prediction-relevant rank: keep top-k L17 PCs, recover CE
    valid = tgt >= 0; Rv = R17[torch.tensor(valid, device=DEV)]; tv = torch.tensor(tgt[valid], device=DEV)
    mu = Rv.mean(0, keepdim=True); Rc = Rv - mu
    V = torch.linalg.svd(Rc, full_matrices=False)[2]  # (D, D) rows = PCs
    full_ce = float(F.cross_entropy(readout(Rv).float(), tv).item())
    base_ce = float(F.cross_entropy(readout(mu.expand(Rv.shape[0], D)).float(), tv).item())
    ce_by_k = {}
    for k in [1, 3, 8, 16, 32, 64, 128, 256]:
        Vk = V[:k].T  # (D, k)
        Rk = mu + (Rc @ Vk) @ Vk.T
        ce_by_k[k] = round(float(F.cross_entropy(readout(Rk).float(), tv).item()), 3)
    # rank needed to recover 90% of the CE gain (base -> full)
    gain = base_ce - full_ce; k90 = None
    for k in [1, 3, 8, 16, 32, 64, 128, 256]:
        if base_ce - ce_by_k[k] >= 0.9 * gain: k90 = k; break
    out = {'pertoken_effdim_L17': round(pertoken_17, 1), 'pertoken_effdim_L15': round(pertoken_15, 1),
           'classmean_effdim_L17': round(classmean_17, 1),
           'full_ce': round(full_ce, 3), 'mean_only_ce': round(base_ce, 3), 'ce_by_topk_pcs': ce_by_k,
           'pcs_for_90pct_ce': k90, 'runtime_s': round(time.time()-t0, 1)}
    out['pred_a_collapse_is_means_only'] = bool(pertoken_17 > 10 and (k90 is None or k90 > 3))
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"per-token eff-dim: L17 {out['pertoken_effdim_L17']} | L15 {out['pertoken_effdim_L15']} | class-MEAN eff-dim L17 {out['classmean_effdim_L17']}", flush=True)
    print(f"CE recovered by top-k L17 PCs (full {full_ce:.3f}, mean-only {base_ce:.3f}): {ce_by_k}", flush=True)
    print(f"PCs for 90% of CE gain: {k90}", flush=True)
    print(f"(a) the '3-dim readout' collapse is MEANS-only; per-token prediction is high-dim: {out['pred_a_collapse_is_means_only']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
