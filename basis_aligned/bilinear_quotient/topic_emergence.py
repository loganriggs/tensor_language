"""WHERE does the TOPIC representation EMERGE across depth, and does ATTENTION or MLP build it? (completes
the causal chain for the content machine). §866/§868: content = high-dim topic tracker, causal. mlp_topic_
folding asks whether MIDDLE MLPs read/write topic; this asks the complementary question — a topic-decode
EMERGENCE curve across all 18 layers, split into the per-layer ATTENTION increment vs MLP increment, to
see WHERE topic decodability appears and WHICH sublayer adds it.

Method: cluster the L15 content residual into K topics (§866) -> a topic label per token. At each block,
capture three residuals: block-input (= prev block output), after-attn (= mlp input), block-output. Decode
the K-way topic label from each with a ridge linear probe (train/test split). Emergence curve = decode acc
at block output vs depth. Attention increment = acc(after-attn) - acc(block-input); MLP increment =
acc(block-output) - acc(after-attn). Sum of increments across depth attributes topic-building to attn vs
mlp. Controls: shuffled-topic-label decode (chance floor); K-way majority baseline.

REGISTERED PREDICTIONS:
  (0) SANITY: topic decode acc >> chance (1/K) and >> shuffled-label by the back layers (topic IS present,
      it defines the L15 clusters); shuffled-label ~ chance at every layer;
  (a) TOPIC EMERGES IN THE MIDDLE via ATTENTION: topic decode acc rises across the MIDDLE layers (low at
      front, near-max by ~L10-14), and the ATTENTION increments dominate the MLP increments there (attn
      aggregates context into topic; the MLP then reads/writes it per mlp_topic_folding) -> total attn
      increment > total mlp increment across the middle band;
  (b) if MLP increments dominate, topic is MLP-built not attention-aggregated; if the curve is flat/high
      from the front, topic is present early (report honestly)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'topic_emergence_results.json'
CONTENT_L = 15; NLAYER = 18
NEVAL = 260; RTOK = 64; RPOS = 32; K = 12


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
    # capture block-input, after-attn (mlp input), block-output at every layer, and L15 content
    binp = {L: [] for L in range(NLAYER)}; aattn = {L: [] for L in range(NLAYER)}; bout = {L: [] for L in range(NLAYER)}
    c15 = []; seqs = []; hs = []
    for L in range(NLAYER):
        def mkbpre(L):
            def pre(mo, a): binp[L].append(a[0].detach().float().reshape(-1, D))
            return pre
        def mkbpost(L):
            def post(mo, i_, o_): bout[L].append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
            return post
        def mkmpre(L):
            def pre(mo, a): aattn[L].append(a[0].detach().float().reshape(-1, D))
            return pre
        hs.append(m.transformer.h[L].register_forward_pre_hook(mkbpre(L)))
        hs.append(m.transformer.h[L].register_forward_hook(mkbpost(L)))
        hs.append(m.transformer.h[L].mlp.register_forward_pre_hook(mkmpre(L)))
    for i in range(0, NEVAL, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx); seqs.append(idx.cpu().numpy())
    for h in hs: h.remove()
    BI = {L: torch.cat(binp[L], 0) for L in range(NLAYER)}; AA = {L: torch.cat(aattn[L], 0) for L in range(NLAYER)}
    BO = {L: torch.cat(bout[L], 0) for L in range(NLAYER)}
    seqs = np.concatenate(seqs, 0); toks = seqs.reshape(-1); pos = np.broadcast_to(np.arange(seqs.shape[1]), seqs.shape).reshape(-1)
    R15 = BO[CONTENT_L]
    Utok, g = mean_subspace(R15, toks, RTOK); Upos, _ = mean_subspace(R15, pos.astype(np.int64), RPOS)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    content = (R15-g) - ((R15-g)@Ucp)@Ucp.T
    cn = content/(content.norm(dim=1, keepdim=True)+1e-9)
    topic = kmeans(cn, K).cpu().numpy()
    rng = np.random.RandomState(0); tshuf = topic.copy(); rng.shuffle(tshuf)
    chance = round(float(np.bincount(topic, minlength=K).max()/len(topic)), 3)
    out = {'k': K, 'chance_majority': chance, 'layers': {}}
    attn_inc = []; mlp_inc = []
    for L in range(NLAYER):
        a_in = decode_acc(BI[L], topic, K); a_at = decode_acc(AA[L], topic, K); a_out = decode_acc(BO[L], topic, K)
        ai = a_at - a_in; mi = a_out - a_at; attn_inc.append(ai); mlp_inc.append(mi)
        out['layers'][f'L{L}'] = {'block_in': round(a_in, 3), 'after_attn': round(a_at, 3), 'block_out': round(a_out, 3),
                                  'attn_increment': round(ai, 3), 'mlp_increment': round(mi, 3)}
        print(f"L{L:>2}: topic-decode in {a_in:.3f} -> after-attn {a_at:.3f} (+{ai:+.3f}) -> out {a_out:.3f} (mlp {mi:+.3f})", flush=True)
    a_shuf = decode_acc(BO[NLAYER-1], tshuf, K)
    mid = range(6, 13)
    out['shuffled_null_decode'] = round(a_shuf, 3)
    out['total_attn_increment_middle'] = round(float(np.sum([attn_inc[L] for L in mid])), 3)
    out['total_mlp_increment_middle'] = round(float(np.sum([mlp_inc[L] for L in mid])), 3)
    out['pred_a_topic_middle_attention'] = bool(out['total_attn_increment_middle'] > out['total_mlp_increment_middle'] and
                                                out['layers']['L12']['block_out'] > out['layers']['L2']['block_out'] + 0.1)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nchance(majority) {chance} | shuffled-label decode {a_shuf:.3f} (should ~ chance)", flush=True)
    print(f"middle(6-12) total attn-increment {out['total_attn_increment_middle']} vs mlp-increment {out['total_mlp_increment_middle']}", flush=True)
    print(f"(a) topic emerges in the middle via attention: {out['pred_a_topic_middle_attention']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
