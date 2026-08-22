"""WHERE is the TOPIC machine BUILT across depth? (mechanism for the middle layers — the last gap in the
bottom-up map). §866/§868: the content machine is a high-dimensional TOPIC tracker, causal but diffuse;
geometry (§857) shows the MIDDLE layers (6-14) re-inflate eff-dim into content/unnamed directions, but we
never named a MIDDLE MLP's computation. Fold each MLP's bilinear readouts onto the TOPIC subspace (built
from the §866 content clustering) the way §849/scan folded onto class/token/pos.

Method: cluster the L15 content residual into K topics (as §866) -> a topic label per token position.
For each scan layer L, build TOPIC-in = topic-conditional means of that layer's INPUT, orthogonalized
against {class, token, pos} (topic structure in the input beyond grammar); TOPIC-out = topic-conditional
means of the OUTPUT (same orth). Select the layer's top TOPIC-writing units (output writes TOPIC-out) and
measure how much their Left/Right readouts READ TOPIC-in, as energy fraction / chance (rank/D). Compare to
the class-read energy of class-writing units. Controls: SHUFFLED-label matched-rank topic subspace (null,
§836 lesson); chance = rank/D.

REGISTERED PREDICTIONS:
  (0) SANITY: topic clusters coherent (§866); class-read energy is ABOVE chance at all layers (class is
      present throughout); shuffled-topic null ~1x (chance);
  (a) TOPIC BUILT IN THE MIDDLE: topic-read energy of topic-writing units RISES front->middle — low
      (~chance) at front MLPs (0-2, topic not present yet) and HIGH (>chance, > shuffled-null) at middle
      MLPs (6-12) where §857 re-inflates content dims -> the middle layers READ topic-organized context
      and WRITE sharper topic, naming their computation;
  (b) if topic-read energy is flat/at-chance everywhere, topic is not built by a foldable MLP read
      (report honestly — topic would then be an attention-carried aggregate the MLPs pass through)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp_topic_folding_results.json'
SCAN_LAYERS = [0, 1, 2, 4, 6, 8, 10, 12, 14, 16]
CONTENT_L = 15
NEVAL = 260; MINCOUNT = 8; RTOK = 64; RPOS = 32; NCLASS_DIR = 12; RTOPIC = 24; NUNIT = 24; K = 12
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


@torch.no_grad()
def capture(rows):
    mods = {L: m.transformer.h[L].mlp for L in SCAN_LAYERS}
    Xs = {L: [] for L in SCAN_LAYERS}; Os = {L: [] for L in SCAN_LAYERS}; c15 = []; seqs = []
    hs = []
    for L in SCAN_LAYERS:
        def mkpre(L):
            def pre(mo, a): Xs[L].append(a[0].detach().float().reshape(-1, D))
            return pre
        def mkpost(L):
            def post(mo, i_, o_): Os[L].append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
            return post
        hs.append(mods[L].register_forward_pre_hook(mkpre(L))); hs.append(mods[L].register_forward_hook(mkpost(L)))
    def hc(mo, i_, o_): c15.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hs.append(m.transformer.h[CONTENT_L].register_forward_hook(hc))
    for i in range(0, NEVAL, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx); seqs.append(idx.cpu().numpy())
    for h in hs: h.remove()
    X = {L: torch.cat(Xs[L], 0) for L in SCAN_LAYERS}; O = {L: torch.cat(Os[L], 0) for L in SCAN_LAYERS}
    return X, O, torch.cat(c15, 0), np.concatenate(seqs, 0)


def mean_subspace(X, labels, r):
    g = X.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(labels):
        if t < 0: continue
        mk = labels == t
        if mk.sum() < MINCOUNT: continue
        rows.append(X[mk].mean(0) - g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0) * torch.tensor(wt, device=X.device, dtype=X.dtype)[:, None]
    k = min(r, M.shape[0])
    return torch.linalg.svd(M, full_matrices=False)[2][:k].T.contiguous()


def orth_against(U, *bases):
    B = torch.cat([b for b in bases if b is not None and b.shape[1] > 0], 1)
    if B.shape[1] == 0: return U
    Q = torch.linalg.qr(B)[0]
    Ur = U - Q @ (Q.T @ U)
    keep = Ur.norm(dim=0) > 1e-3
    if keep.sum() == 0: return U[:, :0]
    return torch.linalg.qr(Ur[:, keep])[0]


def energy_frac(vecs, U):
    if U.shape[1] == 0: return 0.0
    p = (vecs @ U)
    return float((p.pow(2).sum(1) / (vecs.pow(2).sum(1) + 1e-9)).mean())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL); d = dec()
    X, O, R15, seqs = capture(rows)
    toks = seqs.reshape(-1); pos = np.broadcast_to(np.arange(seqs.shape[1]), seqs.shape).reshape(-1)
    clslab = np.array([CLASSES.index(classify(d(int(t)))) for t in toks])
    # topic labels from L15 content residual (§866): strip class+pos, cluster the remainder
    Utok15 = mean_subspace(R15, toks, RTOK); Upos15 = mean_subspace(R15, pos.astype(np.int64), RPOS)
    g15 = R15.mean(0, keepdim=True)
    Ucp15 = torch.linalg.svd(torch.cat([Utok15, Upos15], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    content = (R15-g15) - ((R15-g15)@Ucp15)@Ucp15.T
    cn = content/(content.norm(dim=1, keepdim=True)+1e-9)
    topic_lab = kmeans(cn, K).cpu().numpy()
    rng = np.random.RandomState(0); topic_shuf = topic_lab.copy(); rng.shuffle(topic_shuf)
    # sanity: distinctive tokens per topic
    from collections import Counter
    tgt = np.full_like(seqs, -1); tgt[:, :-1] = seqs[:, 1:]; tgt = tgt.reshape(-1)
    base = Counter(tgt[tgt >= 0]); Nn = int((tgt >= 0).sum()); tnames = {}
    for j in range(K):
        mk = topic_lab == j
        if mk.sum() < 30: continue
        nc = Counter(tgt[mk][tgt[mk] >= 0]); sc = []
        for t, c in nc.items():
            if c < 4: continue
            sc.append(((c/int((tgt[mk] >= 0).sum()))/((base.get(t, 0)+1)/Nn), t))
        sc.sort(reverse=True); tnames[j] = [repr(d(t)) for _, t in sc[:5]]
    out = {'k': K, 'topic_names': tnames, 'layers': {}}
    for L in SCAN_LAYERS:
        Xl = X[L]; Ol = O[L]
        Dw = m.transformer.h[L].mlp.Down.weight.detach().float()
        Lw = m.transformer.h[L].mlp.Left.weight.detach().float(); Rw = m.transformer.h[L].mlp.Right.weight.detach().float()
        Uclass = mean_subspace(Xl, clslab, NCLASS_DIR)
        Utok = mean_subspace(Xl, toks, RTOK); Upos = mean_subspace(Xl, pos.astype(np.int64), RPOS)
        # topic structure in INPUT, beyond grammar
        Utopic_in = orth_against(mean_subspace(Xl, topic_lab, RTOPIC), Uclass, Utok, Upos)
        Ushuf_in = orth_against(mean_subspace(Xl, topic_shuf, RTOPIC), Uclass, Utok, Upos)
        # topic / class structure WRITTEN to output
        Utopic_out = orth_against(mean_subspace(Ol, topic_lab, RTOPIC), mean_subspace(Ol, clslab, NCLASS_DIR))
        Uclass_out = mean_subspace(Ol, clslab, NCLASS_DIR)
        chance_topic = Utopic_in.shape[1]/D; chance_shuf = Ushuf_in.shape[1]/D; chance_class = Uclass.shape[1]/D
        # top topic-writing units and top class-writing units
        topic_mag = (Utopic_out.T @ Dw).norm(dim=0); class_mag = (Uclass_out.T @ Dw).norm(dim=0)
        tt = torch.topk(topic_mag, NUNIT).indices.tolist(); tc = torch.topk(class_mag, NUNIT).indices.tolist()
        def rd(units):
            Lk = Lw[units]; Rk = Rw[units]
            Lk = Lk/(Lk.norm(dim=1, keepdim=True)+1e-9); Rk = Rk/(Rk.norm(dim=1, keepdim=True)+1e-9)
            return Lk, Rk
        Lt, Rt = rd(tt); Lc, Rc = rd(tc)
        topic_read = (energy_frac(Lt, Utopic_in)+energy_frac(Rt, Utopic_in))/2/max(chance_topic, 1e-9)
        shuf_read = (energy_frac(Lt, Ushuf_in)+energy_frac(Rt, Ushuf_in))/2/max(chance_shuf, 1e-9)
        class_read = (energy_frac(Lc, Uclass)+energy_frac(Rc, Uclass))/2/max(chance_class, 1e-9)
        row = {'topic_read': round(topic_read, 2), 'shuffled_null': round(shuf_read, 2), 'class_read': round(class_read, 2)}
        out['layers'][f'mlp{L}'] = row
        print(f"mlp{L:>2}: TOPIC-read x{row['topic_read']:<5} (shuffled-null x{row['shuffled_null']}) | class-read x{row['class_read']}", flush=True)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    tr = {L: out['layers'][f'mlp{L}']['topic_read'] for L in SCAN_LAYERS}
    front = np.mean([tr[L] for L in SCAN_LAYERS if L <= 2]); mid = np.mean([tr[L] for L in SCAN_LAYERS if 6 <= L <= 12])
    out['front_topic_read'] = round(float(front), 2); out['mid_topic_read'] = round(float(mid), 2)
    out['pred_a_topic_built_middle'] = bool(mid > front + 0.5 and mid > 1.5)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nfront(0-2) topic-read x{front:.2f} | middle(6-12) topic-read x{mid:.2f}", flush=True)
    print(f"(a) topic BUILT in the middle (mid>>front, mid>chance): {out['pred_a_topic_built_middle']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
