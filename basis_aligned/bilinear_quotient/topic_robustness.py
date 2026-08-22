"""IS THE TOPIC STRUCTURE REAL, or a k-means artifact of one split? (robustness/null gate for the whole
§866-872 content-machine arc). The topic finding rests on clustering the content residual once; before
treating "content = topic tracker" as established, test whether the SAME topics reappear on a DISJOINT data
split and across K, and whether a structure-destroying NULL fails to.

Method: strip grammar from the L15 residual (content = R - class/pos projection, §866). Split the sequences
into two disjoint halves A and B. Cluster each into K topics; fingerprint each topic by its top distinctive
NEXT-tokens (frequency-controlled over-representation, §866). Replication score = mean over A-topics of the
best distinctive-token Jaccard overlap with any B-topic (Hungarian-free greedy best-match). Repeat for
K=8/12/16. NULL: shuffle the content vectors across token positions (destroy the token<->content link),
recluster A/B, same metric -> a real topic structure replicates well above this null.

REGISTERED PREDICTIONS:
  (0) SANITY: A-topics have coherent distinctive tokens (print a few); K=12 matches §866;
  (a) TOPIC REPLICATES: real content clusters give A<->B distinctive-token overlap WELL ABOVE the
      shuffled-content null at every K -> the topic structure is a robust property of the model, not a
      k-means artifact;
  (b) if real overlap ~ shuffled null, the topic clusters do not replicate (retract the topic naming to a
      one-split description). Report both numbers plainly."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F
from collections import Counter

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'topic_robustness_results.json'
CONTENT_L = 15; NEVAL = 300; RTOK = 64; RPOS = 32; KS = [8, 12, 16]; NDISTINCT = 25


def dec():
    import tiktoken; enc = tiktoken.get_encoding('gpt2'); return lambda i: enc.decode([int(i)])


def forward_cap(idx):
    cap = {}
    def h(mo, i_, o_): cap['r'] = (o_[0] if isinstance(o_, tuple) else o_).detach().float()
    hh = m.transformer.h[CONTENT_L].register_forward_hook(h)
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    hh.remove()
    return cap['r']


def kmeans(X, k, iters=25, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed)
    c = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for j in range(k):
            mk = a == j
            if mk.any(): c[j] = X[mk].mean(0)
    return a.cpu().numpy()


def mean_subspace(X, labels, r):
    g = X.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(labels):
        mk = labels == t
        if mk.sum() < 5: continue
        rows.append(X[mk].mean(0) - g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0) * torch.tensor(wt, device=X.device, dtype=X.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:min(r, M.shape[0])].T.contiguous(), g


def distinctive(labels, tgt, K, base, Nn):
    """top-N distinctive next-tokens per topic (over-representation)."""
    fps = {}
    for j in range(K):
        mk = labels == j
        nj = int((tgt[mk] >= 0).sum())
        if nj < 30: continue
        nc = Counter(tgt[mk][tgt[mk] >= 0]); sc = []
        for t, c in nc.items():
            if c < 4: continue
            sc.append(((c/nj)/((base.get(t, 0)+1)/Nn), t))
        sc.sort(reverse=True); fps[j] = set(t for _, t in sc[:NDISTINCT])
    return fps


def replication(fpsA, fpsB):
    """mean over A-topics of best Jaccard overlap with any B-topic."""
    scores = []
    for a, sa in fpsA.items():
        best = 0.0
        for b, sb in fpsB.items():
            j = len(sa & sb)/max(len(sa | sb), 1)
            best = max(best, j)
        scores.append(best)
    return float(np.mean(scores)) if scores else 0.0


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    Rs = []; seqs = []
    for i in range(0, NEVAL, 4):
        idx = rows[i:i+4, :256].to(DEV).contiguous(); Rs.append(forward_cap(idx).cpu()); seqs.append(idx.cpu().numpy())
    R = torch.cat(Rs, 0); S = np.concatenate(seqs, 0); Nseq = S.shape[0]
    allR = R.reshape(-1, D).to(DEV); toks = S.reshape(-1); pos = np.broadcast_to(np.arange(256), S.shape).reshape(-1)
    tgt = np.full_like(S, -1); tgt[:, :-1] = S[:, 1:]; tgt = tgt.reshape(-1)
    Utok, g = mean_subspace(allR, toks, RTOK); Upos, _ = mean_subspace(allR, pos.astype(np.int64), RPOS)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    content = (allR-g) - ((allR-g)@Ucp)@Ucp.T
    content = content/(content.norm(dim=1, keepdim=True)+1e-9)
    # disjoint sequence split -> token masks
    rng = np.random.RandomState(0); so = rng.permutation(Nseq); A_seq = set(so[:Nseq//2].tolist())
    seq_of = np.repeat(np.arange(Nseq), 256)
    inA = np.array([s in A_seq for s in seq_of])
    base = Counter(tgt[tgt >= 0]); Nn = int((tgt >= 0).sum())
    # null: shuffle content rows across token positions
    perm = rng.permutation(content.shape[0]); content_shuf = content[perm]
    out = {'ks': KS, 'by_k': {}, 'sample_topics_k12': {}}
    for K in KS:
        cA = kmeans(content[torch.tensor(inA, device=DEV)], K, seed=1)
        cB = kmeans(content[torch.tensor(~inA, device=DEV)], K, seed=2)
        fpsA = distinctive(cA, tgt[inA], K, base, Nn); fpsB = distinctive(cB, tgt[~inA], K, base, Nn)
        rep = replication(fpsA, fpsB)
        # null
        cAs = kmeans(content_shuf[torch.tensor(inA, device=DEV)], K, seed=1)
        cBs = kmeans(content_shuf[torch.tensor(~inA, device=DEV)], K, seed=2)
        fpsAs = distinctive(cAs, tgt[inA], K, base, Nn); fpsBs = distinctive(cBs, tgt[~inA], K, base, Nn)
        repnull = replication(fpsAs, fpsBs)
        out['by_k'][f'K{K}'] = {'replication': round(rep, 3), 'shuffled_null': round(repnull, 3), 'ratio': round(rep/max(repnull, 1e-6), 2)}
        print(f"K={K:>2}: A<->B distinctive-token replication {rep:.3f} | shuffled-content null {repnull:.3f} | ratio {out['by_k'][f'K{K}']['ratio']}x", flush=True)
        if K == 12:
            for j in list(fpsA.keys())[:8]:
                out['sample_topics_k12'][str(j)] = [repr(d(int(t))) for t in list(fpsA[j])[:6]]
    reps = [out['by_k'][f'K{K}']['replication'] for K in KS]; nulls = [out['by_k'][f'K{K}']['shuffled_null'] for K in KS]
    out['mean_replication'] = round(float(np.mean(reps)), 3); out['mean_null'] = round(float(np.mean(nulls)), 3)
    out['pred_a_topic_replicates'] = bool(np.mean(reps) > 2*max(np.mean(nulls), 1e-6) and np.mean(reps) > 0.2)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nmean replication {out['mean_replication']} vs shuffled-content null {out['mean_null']}", flush=True)
    print(f"(a) topic structure replicates (real >> null): {out['pred_a_topic_replicates']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
