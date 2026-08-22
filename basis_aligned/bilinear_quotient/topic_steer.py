"""CAUSAL verification of the TOPIC finding (§866): does the content machine's topic representation
CAUSALLY drive topically-coherent prediction? Cluster the content residual by topic (as §866), take each
cluster's mean content-residual direction, ADD it (amplified) to the readout-input residual, and measure
the logit change on that topic's distinctive tokens vs OTHER topics' tokens. If steering toward topic A
raises topic-A words' logits specifically (diagonal >> off-diagonal), the topic representation is causal —
not just a correlational cluster.

REGISTERED PREDICTIONS:
  (0) SANITY: topic clusters reproduce §866 (coherent distinctive tokens);
  (a) CAUSAL TOPIC: steering toward topic A raises topic-A distinctive tokens' mean logit MORE than other
      topics' tokens (diagonal of the steer×topic logit-gain matrix dominates its row) -> the topic
      representation causally drives topically-coherent prediction;
  (b) if off-diagonal ≈ diagonal, the topic clusters are correlational, not a causal steering axis."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F
from collections import Counter
D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'topic_steer_results.json'; NEVAL = 260; RTOK = 64; RPOS = 32; K = 12; READ_L = 15; ALPHA = 8.0; NDISTINCT = 30
ST = {'on': False, 'vec': None}


def forward_to_L(idx, L):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for bi, blk in enumerate(m.transformer.h):
        x, v1 = blk(x, v1, x0)
    return x


def mk_hook():
    def hook(mo, i_, o_):
        if not ST['on']: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; sh = y.shape; v = y.reshape(-1, D)
        v2 = v + ALPHA * ST['vec'].to(v.dtype)
        return (v2.reshape(sh),) + tuple(o_[1:]) if isinstance(o_, tuple) else v2.reshape(sh)
    return hook


def forward_logits_hooked(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def mean_subspace(O, labels, r):
    g = O.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(labels):
        mk = labels == t
        if mk.sum() < 5: continue
        rows.append(O[mk].mean(0)-g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0)*torch.tensor(wt, device=O.device, dtype=O.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:min(r, M.shape[0])].T.contiguous(), g


def kmeans(X, k, iters=25, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed); c = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for j in range(k):
            mk = a == j
            if mk.any(): c[j] = X[mk].mean(0)
    return a


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    import tiktoken; enc = tiktoken.get_encoding('gpt2'); d = lambda i: enc.decode([int(i)])
    # capture L15 residual + tokens
    outs = []; seqs = []
    def h(mo, i_, o_): outs.append((o_[0] if isinstance(o_, tuple) else o_).detach().float())
    hh = m.transformer.h[READ_L].register_forward_hook(h)
    for i in range(0, NEVAL, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits_hooked(idx); seqs.append(idx.cpu().numpy())
    hh.remove()
    R = torch.cat([o.reshape(-1, D) for o in outs], 0); S = np.concatenate(seqs, 0); T = S.shape[1]
    cur = S.reshape(-1); pos = np.broadcast_to(np.arange(T), S.shape).reshape(-1)
    tgt = np.full_like(S, -1); tgt[:, :-1] = S[:, 1:]; tgt = tgt.reshape(-1)
    Utok, g = mean_subspace(R, cur, RTOK); Upos, _ = mean_subspace(R, pos.astype(np.int64), RPOS)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    content = (R-g) - ((R-g)@Ucp)@Ucp.T
    cn = content/(content.norm(dim=1, keepdim=True)+1e-9)
    a = kmeans(cn, K).cpu().numpy()
    # per-topic mean content direction (in residual space) + distinctive next-tokens
    base_nxt = Counter(tgt[tgt >= 0]); Nn = int((tgt >= 0).sum())
    topics = {}
    for j in range(K):
        mk = a == j
        if mk.sum() < 30: continue
        vec = content[torch.tensor(mk, device=DEV)].mean(0); vec = vec/ (vec.norm()+1e-9)
        nxtc = Counter(tgt[mk][tgt[mk] >= 0]); scored = []
        for t, c in nxtc.items():
            if c < 4: continue
            scored.append(((c/int((tgt[mk] >= 0).sum()))/((base_nxt.get(t, 0)+1)/Nn), t))
        scored.sort(reverse=True); dtok = [t for _, t in scored[:NDISTINCT]]
        if dtok: topics[j] = {'vec': vec, 'dtok': dtok, 'name': [repr(d(t)) for t in dtok[:5]]}
    tids = list(topics.keys())
    # steer toward each topic, measure mean logit on each topic's distinctive tokens
    hh = m.transformer.h[READ_L].register_forward_hook(mk_hook())
    idxb = rows[:8, :257].to(DEV)[:, :-1].contiguous()
    ST['on'] = False; base_logits = forward_logits_hooked(idxb).float().reshape(-1, forward_logits_hooked(idxb).shape[-1])
    def mean_logit_on(logits, toks): return float(logits[:, torch.tensor(toks, device=DEV)].mean())
    mat = {}
    for A in tids:
        ST['on'] = True; ST['vec'] = topics[A]['vec']
        lg = forward_logits_hooked(idxb).float().reshape(-1, base_logits.shape[-1]); ST['on'] = False
        row = {}
        for B in tids:
            row[B] = round(mean_logit_on(lg, topics[B]['dtok']) - mean_logit_on(base_logits, topics[B]['dtok']), 3)
        mat[A] = row
    hh.remove()
    diag = [mat[A][A] for A in tids]; off = [mat[A][B] for A in tids for B in tids if B != A]
    out = {'k': K, 'n_topics': len(tids), 'topic_names': {A: topics[A]['name'] for A in tids},
           'steer_logit_gain_matrix': {str(A): {str(B): v for B, v in row.items()} for A, row in mat.items()},
           'mean_diagonal': round(float(np.mean(diag)), 3), 'mean_offdiag': round(float(np.mean(off)), 3),
           'pred_a_causal_topic': bool(np.mean(diag) > np.mean(off) + 0.5), 'runtime_s': round(time.time()-t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    for A in tids: print(f"steer->topic{A} {topics[A]['name'][:3]}: own-topic logit gain {mat[A][A]} | mean other {round(np.mean([mat[A][B] for B in tids if B!=A]),3)}", flush=True)
    print(f"\nmean diagonal (own-topic gain) {out['mean_diagonal']} | mean off-diagonal {out['mean_offdiag']}", flush=True)
    print(f"(a) topic representation is CAUSAL (diagonal >> off-diagonal): {out['pred_a_causal_topic']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
