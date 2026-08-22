"""VARIABLE-LEVEL causal abstraction: is the POSITION variable causal via INTERCHANGE INTERVENTION? (completes the named-variable causal set: class §892, topic §894, position here). Patch the POSITION subspace base<-source (different position bucket) and test whether the prediction shifts toward the SOURCE position's distinctive next-tokens (early positions favor sentence-starters/capitalized; late favor continuations). vs random-subspace null + no-patch. [derived from das_topic_interchange]
(content analog of §892's class result; the right causal test the weak mean-steering §868 could not settle).
Patch ONLY the topic-subspace coordinates of a BASE residual at L15/pos with a SOURCE passage's (a different
topic), and test whether the model's next-token prediction shifts toward the SOURCE topic's distinctive words
rather than the base topic's. If it does above a random-subspace null, the topic is a causally-realized
variable — verified by interchange (which respects read+write jointly) even though single-direction steering
was weak.

Topic subspace: cluster the content residual (R − class/pos projection, §866) into K topics; the topic
directions = the K cluster-mean deviations (rank ~K−1). Readout metric: for a (base,source) pair with topics
t_b, t_s, measure Δ = [logit gain on source-topic distinctive tokens] − [logit gain on base-topic distinctive
tokens] after the patch; a positive Δ means the prediction moved toward the source topic. IIA-analog = fraction
of pairs with Δ>0 (topic flipped in the predicted direction). Controls: random-subspace patch (null);
no-patch baseline; require the topic subspace to beat both.

REGISTERED PREDICTIONS:
  (0) SANITY: topics coherent (distinctive tokens); no-patch Δ ~ 0;
  (a) TOPIC IS A CAUSAL VARIABLE via interchange: patching the topic subspace gives mean Δ > 0 and flip-rate
      (Δ>0) well above the random-subspace null -> topic is causally realized (interchange succeeds where
      steering failed, §868), the content analog of the class result (§892);
  (b) if topic-subspace Δ ~ random null, the topic is not a causally-patchable variable at this layer (report
      plainly — content may be too distributed to interchange as one subspace)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F
from collections import Counter

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'position_interchange_results.json'
NEVAL = 400; SEQ = 256; QP = 128; PATCH_L = 15; K = 6; RTOK = 64; RPOS = 32; NDIST = 40
PATCH = {'on': False, 'U': None, 'src': None}


def dec():
    import tiktoken; enc = tiktoken.get_encoding('gpt2'); return lambda i: enc.decode([int(i)])


def readout(x): return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def patch_hook(mo, i_, o_):
    if not PATCH['on']: return o_
    y = o_[0] if isinstance(o_, tuple) else o_; U = PATCH['U']; b = y[:, QP, :]
    b_new = b - (b @ U) @ U.T + PATCH['src']
    y = y.clone(); y[:, QP, :] = b_new
    return (y,) + tuple(o_[1:]) if isinstance(o_, tuple) else y


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return readout(x)


@torch.no_grad()
def capture_L(idx):
    cap = {}
    def h(mo, i_, o_): cap['r'] = (o_[0] if isinstance(o_, tuple) else o_).detach().float()
    hh = m.transformer.h[PATCH_L].register_forward_hook(h)
    forward_logits(idx); hh.remove()
    return cap['r']


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
        rows.append(X[mk].mean(0)-g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0)*torch.tensor(wt, device=X.device, dtype=X.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:min(r, M.shape[0])].T.contiguous(), g


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    # capture L15 residual at QP for all seqs + build topic clustering from ALL positions' content
    allR = []; seqs = []
    for i in range(0, nb, 8):
        idx = blocks[i:i+8].to(DEV); r = capture_L(idx); allR.append(r.reshape(-1, D).cpu()); seqs.append(idx.cpu().numpy())
    Rall = torch.cat(allR, 0).to(DEV); Sall = np.concatenate(seqs, 0)
    toks = Sall.reshape(-1); pos = np.broadcast_to(np.arange(SEQ), Sall.shape).reshape(-1)
    Utok, g = mean_subspace(Rall, toks, RTOK); Upos, _ = mean_subspace(Rall, pos.astype(np.int64), RPOS)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    content = (Rall-g) - ((Rall-g)@Ucp)@Ucp.T
    cn = content/(content.norm(dim=1, keepdim=True)+1e-9)
    # POSITION buckets instead of topic clusters
    posbucket = np.minimum((np.arange(SEQ) // (SEQ//K)), K-1)
    topic_all = np.broadcast_to(posbucket, (nb, SEQ)).copy()
    # topic subspace = content-cluster mean deviations (rank ~K-1), orthonormalized
    Utopic_raw, gc = mean_subspace(Rall, topic_all.reshape(-1), K)
    U = torch.linalg.qr(Utopic_raw)[0][:, :K-1].contiguous()
    g_content = gc
    # distinctive next-tokens per topic
    tgt = np.full_like(Sall, -1); tgt[:, :-1] = Sall[:, 1:]; tgt = tgt.reshape(-1)
    base = Counter(tgt[tgt >= 0]); Nn = int((tgt >= 0).sum()); dtok = {}
    for j in range(K):
        mk = topic_all.reshape(-1) == j; nj = int((tgt[mk] >= 0).sum())
        if nj < 30: continue
        nc = Counter(tgt[mk][tgt[mk] >= 0]); sc = []
        for t, c in nc.items():
            if c < 4: continue
            sc.append(((c/nj)/((base.get(t, 0)+1)/Nn), t))
        sc.sort(reverse=True); dtok[j] = [t for _, t in sc[:NDIST]]
    # per-seq: topic at QP, L15 residual at QP
    RqpArr = Rall.reshape(nb, SEQ, D)[:, QP, :].contiguous()
    topic_qp = topic_all[:, QP]
    valid = [i for i in range(nb) if topic_qp[i] in dtok]
    g_dev = torch.Generator(device=DEV).manual_seed(0); U_rnd = torch.linalg.qr(torch.randn(D, K-1, generator=g_dev, device=DEV))[0]
    # pairs: source topic != base topic
    rng = np.random.RandomState(0); src_of = {}
    for bi in valid:
        cand = [j for j in valid if topic_qp[j] != topic_qp[bi]]; src_of[bi] = cand[rng.randint(len(cand))]
    def mean_logit(lg_row, toks_):
        return float(lg_row[torch.tensor(toks_, device=DEV)].mean())
    hh = m.transformer.h[PATCH_L].register_forward_hook(patch_hook)
    def run(Usub, do_patch):
        PATCH['U'] = Usub; deltas = []
        vb = valid
        for i in range(0, len(vb), 8):
            batch = vb[i:i+8]; bidx = blocks[batch].to(DEV)
            if do_patch:
                src = RqpArr[torch.tensor([src_of[b] for b in batch], device=DEV)]
                PATCH['src'] = (src @ Usub) @ Usub.T; PATCH['on'] = True
            lg = forward_logits(bidx).float()[:, QP, :]; PATCH['on'] = False
            for bj, b in enumerate(batch):
                ts = topic_qp[src_of[b]]; tb = topic_qp[b]
                gain_s = mean_logit(lg[bj], dtok[ts]); gain_b = mean_logit(lg[bj], dtok[tb])
                deltas.append(gain_s - gain_b)
        return np.array(deltas)
    # baseline (no patch): the natural source-minus-base distinctive gain
    d_base = run(U, False)
    d_topic = run(U, True)
    d_rnd = run(U_rnd, True)
    hh.remove()
    out = {'patch_layer': PATCH_L, 'query_pos': QP, 'k': K, 'rank': K-1, 'n_pairs': len(valid),
           'mean_delta_topic_patch': round(float(d_topic.mean()), 4), 'flip_rate_topic': round(float((d_topic > 0).mean()), 3),
           'mean_delta_random_patch': round(float(d_rnd.mean()), 4), 'flip_rate_random': round(float((d_rnd > 0).mean()), 3),
           'mean_delta_nopatch': round(float(d_base.mean()), 4), 'flip_rate_nopatch': round(float((d_base > 0).mean()), 3),
           'runtime_s': round(time.time()-t0, 1)}
    out['pred_a_position_causal_interchange'] = bool(out['mean_delta_topic_patch'] > out['mean_delta_random_patch'] + 0.05 and
                                                  out['flip_rate_topic'] > out['flip_rate_random'] + 0.1 and
                                                  out['mean_delta_topic_patch'] > out['mean_delta_nopatch'] + 0.05)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"INTERCHANGE on POSITION subspace (base<-source at L{PATCH_L}, pos {QP}); Δ = source-topic minus base-topic distinctive-logit gain:", flush=True)
    print(f"  topic-patch: mean Δ {out['mean_delta_topic_patch']} | flip-rate {out['flip_rate_topic']}", flush=True)
    print(f"  random-patch: mean Δ {out['mean_delta_random_patch']} | flip-rate {out['flip_rate_random']}", flush=True)
    print(f"  no-patch:     mean Δ {out['mean_delta_nopatch']} | flip-rate {out['flip_rate_nopatch']}", flush=True)
    print(f"(a) position is a causal variable via interchange: {out['pred_a_position_causal_interchange']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
