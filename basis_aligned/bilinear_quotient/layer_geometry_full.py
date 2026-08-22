"""IS EACH EARLY LAYER RE-CLUSTERING THE TOKEN GEOMETRY, or redundantly re-doing the same thing?
(user reframe of §847). A token-conditional mean is a CLUSTERING; collapsing it to an 8-way POS label
(as §846/847 did) hides the fine token-to-token geometry, so "layer 1 is redundant" was premature. The
right object: the token×token similarity structure and how it CHANGES across stages (embedding → after
layer 0 → after layer 1 → …). If each layer meaningfully changes the relative-similarity geometry
(including WITHIN a grammatical class), it is re-clustering — doing real work — not redundant.

Method: capture the residual-stream state after the embedding and after each of layers 0..5. For the
top frequent tokens, take the token-conditional-mean representation at each stage, build the token×token
cosine-similarity RDM, and:
 - consecutive RSA: Pearson corr of successive-stage RDMs (low = big re-clustering at that layer);
 - cumulative drift: corr of each stage's RDM to the embedding RDM;
 - WITHIN-class RSA: same, restricted to same-POS-class token pairs (isolates fine re-clustering);
 - clustering: participation ratio (effective dim) of the token-mean matrix per stage.

REGISTERED PREDICTIONS:
  (0) SANITY: consecutive RSA is < 1 (stages differ) and cumulative drift decreases with depth;
  (a) NOT REDUNDANT: each early layer changes the geometry meaningfully (consecutive RSA well below
      ~0.95), AND the WITHIN-class geometry keeps changing too -> layers 1,2,… re-cluster, they don't
      re-do the same map; report the per-layer geometry-change and eff-dim so we can SEE the
      progressive clustering;
  (b) if consecutive RSA ~1 after layer 0 (geometry frozen), the later early layers ARE redundant."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'layer_geometry_full_results.json'
NEVAL = 300; MINCOUNT = 12; NLAYERS = 18
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


@torch.no_grad()
def capture(rows):
    """residual stream after embedding and after each of layers 0..NLAYERS-1."""
    stages = {}  # name -> list of (N,D)
    caps = {L: [] for L in range(NLAYERS)}; emb = []; toks = []
    hs = []
    for L in range(NLAYERS):
        def mk(L):
            def h(mo, i_, o_): caps[L].append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
            return h
        hs.append(m.transformer.h[L].register_forward_hook(mk(L)))
    for i in range(0, NEVAL, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        emb.append(x.detach().float().reshape(-1, D))
        for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
        toks.append(idx.cpu().numpy().reshape(-1))
    for h in hs: h.remove()
    rep = {'emb': torch.cat(emb, 0)}
    for L in range(NLAYERS): rep[f'L{L}'] = torch.cat(caps[L], 0)
    return rep, np.concatenate(toks)


def token_means(R, toks, keep):
    rows = []
    for t in keep:
        rows.append(R[toks == t].mean(0))
    return torch.stack(rows, 0)   # (ntok, D)


def rdm(M):
    Mn = M / (M.norm(dim=1, keepdim=True) + 1e-9)
    S = Mn @ Mn.T
    return (1 - S)


def rsa(A, B):
    iu = torch.triu_indices(A.shape[0], A.shape[0], offset=1)
    a = A[iu[0], iu[1]].cpu().numpy(); b = B[iu[0], iu[1]].cpu().numpy()
    return float(np.corrcoef(a, b)[0, 1])


def effdim(M):
    Mc = M - M.mean(0, keepdim=True); s = torch.linalg.svdvals(Mc)**2
    return float((s.sum()**2)/(s**2).sum())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL); d = dec()
    rep, toks = capture(rows)
    uniq, cnts = np.unique(toks, return_counts=True); keep = uniq[cnts >= MINCOUNT]
    keep = keep[np.argsort(-cnts[cnts >= MINCOUNT])][:300]
    clslab = np.array([CLASSES.index(classify(d(int(t)))) for t in keep])
    stage_names = ['emb'] + [f'L{L}' for L in range(NLAYERS)]
    means = {s: token_means(rep[s], toks, keep) for s in stage_names}
    rdms = {s: rdm(means[s]) for s in stage_names}
    # within-class mask (same-class pairs)
    same = torch.tensor(clslab[:, None] == clslab[None, :], device=DEV)
    def rsa_within(A, B):
        iu = torch.triu_indices(A.shape[0], A.shape[0], offset=1)
        mk = same[iu[0], iu[1]].cpu().numpy()
        a = A[iu[0], iu[1]].cpu().numpy()[mk]; b = B[iu[0], iu[1]].cpu().numpy()[mk]
        return float(np.corrcoef(a, b)[0, 1])
    consec = {}; within = {}; drift = {}; ed = {}
    for i in range(1, len(stage_names)):
        s0, s1 = stage_names[i-1], stage_names[i]
        consec[f'{s0}->{s1}'] = round(rsa(rdms[s0], rdms[s1]), 3)
        within[f'{s0}->{s1}'] = round(rsa_within(rdms[s0], rdms[s1]), 3)
    for s in stage_names:
        drift[s] = round(rsa(rdms['emb'], rdms[s]), 3); ed[s] = round(effdim(means[s]), 1)
    out = {'n_tokens': len(keep), 'consecutive_rsa': consec, 'within_class_consecutive_rsa': within,
           'drift_vs_embedding': drift, 'effective_dim': ed,
           'pred_a_reclustering': bool(max(consec.values()) < 0.95), 'runtime_s': round(time.time()-t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print("consecutive RSA (1=identical geometry, low=re-clustered):", flush=True)
    for k, v in consec.items(): print(f"  {k}: all {v} | within-class {within[k]}", flush=True)
    print("drift vs embedding:", drift, flush=True)
    print("effective dim per stage:", ed, flush=True)
    print(f"(a) early layers RE-CLUSTER (not redundant; max consec RSA {max(consec.values())} < 0.95): {out['pred_a_reclustering']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
