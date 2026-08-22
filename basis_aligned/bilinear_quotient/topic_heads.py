"""WHICH ATTENTION HEADS aggregate the topic? (head-level mechanism for the §870 aggregator — folds the
next level of detail into the bottom-up map). §870: attention builds the topic representation across depth.
bilin18 has 9 heads/layer; are a FEW heads the topic aggregators, or is it spread across all heads?

Method: attention writes to the residual as c_proj(y), where y is the concatenated per-head outputs
(B,T,nh*hd). Capture y with a pre-hook on each layer's attn.c_proj; each head h's ADDITIVE contribution to
the residual is c_proj applied to y with only head h's slice kept: contrib_h = y_h @ W_proj^T (bias
excluded, as it is head-independent). Decode the §866 topic label (from the full L15 residual) from each
head's contribution with a ridge probe. A head whose contribution decodes topic well is a topic aggregator;
a head near chance is topic-blind (local/positional). Controls: shuffled-topic-label decode per head
(chance); compare to full-attention-output topic decode (ceiling).

REGISTERED PREDICTIONS:
  (0) SANITY: full attn-output topic-decode >> chance at these layers; shuffled-label ~ chance per head;
  (a) SPARSE TOPIC HEADS: topic decodability is UNEVEN across the 9 heads — a subset of heads carry most of
      the topic (top head's decode >> median head's), rather than every head contributing equally -> the
      aggregator is localized to specific heads, nameable as topic heads;
  (b) if all heads decode topic near-equally, topic aggregation is uniformly distributed across heads
      (report honestly)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'topic_heads_results.json'
CONTENT_L = 15; NEVAL = 260; RTOK = 64; RPOS = 32; K = 12
LAYERS = [2, 4, 6, 8, 10, 12, 14]


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def kmeans(X, k, iters=25, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed)
    c = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for j in range(k):
            mk = a == j
            if mk.any(): c[j] = X[mk].mean(0)
    return a


def mean_subspace(X, labels, r):
    g = X.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(labels):
        mk = labels == t
        if mk.sum() < 5: continue
        rows.append(X[mk].mean(0) - g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0) * torch.tensor(wt, device=X.device, dtype=X.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:min(r, M.shape[0])].T.contiguous(), g


def decode_acc(F_, y, ncls, seed=0):
    n = F_.shape[0]; rng = np.random.RandomState(seed); idx = rng.permutation(n)
    ntr = int(0.7*n); tr, te = idx[:ntr], idx[ntr:]
    Ft = F_[tr]; Y = torch.zeros(len(tr), ncls, device=DEV); Y[torch.arange(len(tr)), torch.tensor(y[tr], device=DEV)] = 1.0
    A = Ft.T @ Ft + 1e2*torch.eye(Ft.shape[1], device=DEV); Wp = torch.linalg.solve(A, Ft.T @ Y)
    return float(((F_[te] @ Wp).argmax(1).cpu().numpy() == y[te]).mean())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    nh = m.transformer.h[0].attn.n_head; hd = (m.transformer.h[0].attn.head_dim
          if hasattr(m.transformer.h[0].attn, 'head_dim') else D//nh)
    print(f"n_head={nh} head_dim={hd}", flush=True)
    # capture per-layer c_proj input (y = concat heads) and L15 residual
    ybuf = {L: [] for L in LAYERS}; c15 = []; seqs = []; hs = []
    for L in LAYERS:
        def mkpre(L):
            def pre(mo, a_): ybuf[L].append(a_[0].detach().float().reshape(-1, nh*hd))
            return pre
        hs.append(m.transformer.h[L].attn.c_proj.register_forward_pre_hook(mkpre(L)))
    def hc(mo, i_, o_): c15.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hs.append(m.transformer.h[CONTENT_L].register_forward_hook(hc))
    for i in range(0, NEVAL, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx); seqs.append(idx.cpu().numpy())
    for h in hs: h.remove()
    Y = {L: torch.cat(ybuf[L], 0) for L in LAYERS}; R15 = torch.cat(c15, 0)
    S = np.concatenate(seqs, 0); toks = S.reshape(-1); pos = np.broadcast_to(np.arange(S.shape[1]), S.shape).reshape(-1)
    # topic labels from L15 content
    Utok, g = mean_subspace(R15, toks, RTOK); Upos, _ = mean_subspace(R15, pos.astype(np.int64), RPOS)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    content = (R15-g) - ((R15-g)@Ucp)@Ucp.T; cn = content/(content.norm(dim=1, keepdim=True)+1e-9)
    topic = kmeans(cn, K).cpu().numpy()
    rng = np.random.RandomState(0); tshuf = topic.copy(); rng.shuffle(tshuf)
    out = {'n_head': nh, 'layers': {}, 'chance': round(float(np.bincount(topic, minlength=K).max()/len(topic)), 3)}
    for L in LAYERS:
        Wp = m.transformer.h[L].attn.c_proj.weight.detach().float()   # (D, nh*hd)
        y = Y[L]                                                      # (N, nh*hd)
        full = decode_acc((y @ Wp.T), topic, K)                      # full attention output (no bias)
        per = []
        for hh in range(nh):
            yh = torch.zeros_like(y); yh[:, hh*hd:(hh+1)*hd] = y[:, hh*hd:(hh+1)*hd]
            per.append(round(decode_acc((yh @ Wp.T), topic, K), 3))
        shuf = decode_acc((y @ Wp.T), tshuf, K)
        top = int(np.argmax(per))
        out['layers'][f'L{L}'] = {'full_attn_out': round(full, 3), 'per_head': per,
                                  'top_head': top, 'top_head_acc': per[top], 'median_head_acc': round(float(np.median(per)), 3),
                                  'shuffled_null': round(shuf, 3)}
        print(f"L{L:>2}: full {full:.3f} | heads {per} | top head #{top} ({per[top]}) vs median {np.median(per):.3f} | shuf {shuf:.3f}", flush=True)
    # aggregate: is topic sparse across heads?
    ratios = [out['layers'][f'L{L}']['top_head_acc']/max(out['layers'][f'L{L}']['median_head_acc'], 1e-6) for L in LAYERS]
    out['mean_top_over_median'] = round(float(np.mean(ratios)), 2)
    out['pred_a_sparse_topic_heads'] = bool(np.mean(ratios) > 1.5)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nmean top-head / median-head topic-decode ratio {out['mean_top_over_median']}", flush=True)
    print(f"(a) topic aggregation is localized to specific heads: {out['pred_a_sparse_topic_heads']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
