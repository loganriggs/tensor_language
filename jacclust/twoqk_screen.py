"""tick 28 (autonomous): is the two-QK causal gain a WEIGHTS-ONLY pre-screen?

tick 27 found: on the real 500M two-QK model, clustering queries by BOTH matrices beats either alone,
and the driver is that the two per-token query covectors q1til,q2til are near-orthogonal (|cos|~0.05-0.08).
That was measured on data. The program's signature move (S11 anisotropy law) is to find a WEIGHTS-ONLY
scalar that predicts a data/causal effect. So:

  weights-only alignment per head:  A_w = ||Wq_h Wq2_h^T||_F / (||Wq_h||_F ||Wq2_h||_F)   in [0,1]
      (Wq_h, Wq2_h are the head-sliced query maps, hd x d_model; 0 => orthogonal query circuits)
  data-informed:  mean_token |cos(q1til, q2til)|
  causal gain:    swap-within(both concat) - max(swap-within(q1 only), swap-within(q2 only))

Hypothesis (S11 shape): LOW alignment (independent circuits) => LARGE causal gain from folding in q2;
weights-only A_w is a necessary/cheap proxy, data-|cos| the sufficient one.

Screen A_w (instant) + data-|cos| for all heads; measure the causal gain on a SPREAD of heads chosen
across the A_w range; correlate. 5 seeds. Same swap-within metric + controls as ticks 23-27.
"""
import sys, torch, json, numpy as np, torch.nn.functional as F
from sklearn.cluster import KMeans
sys.path.insert(0, "/workspace/tensor_language")
from huggingface_hub import hf_hub_download
import jacclust.tt_model as TT

DEV = "cuda"; torch.set_default_dtype(torch.float32)
repo = "Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd"
cfg = json.load(open(hf_hub_download(repo, "config.json"))); cfg.pop("step", None)
m = TT.GPT(TT.GPTConfig(**cfg)).to(DEV).eval()
m.load_state_dict(torch.load(hf_hub_download(repo, "pytorch_model.bin"), map_location=DEV, weights_only=True))
from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained("gpt2")
import datasets
K = 8
nh, hd = m.transformer.h[0].attn.n_head, m.transformer.h[0].attn.head_dim
nL = len(m.transformer.h)

def apply_rot(x, c, s):
    d = x.shape[-1] // 2; x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * c + x2 * s, -x1 * s + x2 * c], -1)

def forward_ce(idx, tgt, LAYER=-1, HEAD=-1, qov1=None, qov2=None):
    x = m.transformer.wte(idx); x = F.rms_norm(x, (x.size(-1),)); x0 = x; v1 = None
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0; a = blk.attn
        h = F.rms_norm(x, (x.size(-1),)); B, T, C = h.shape
        q = a.c_q(h).view(B, T, nh, hd); k = a.c_k(h).view(B, T, nh, hd)
        q2 = a.c_q2(h).view(B, T, nh, hd); k2 = a.c_k2(h).view(B, T, nh, hd)
        v = a.c_v(h).view(B, T, nh, hd)
        if v1 is None: v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        cos, sin = a.rotary(q); cos = cos.float(); sin = sin.float()
        q = F.rms_norm(q, (hd,)); k = F.rms_norm(k, (hd,)); q2 = F.rms_norm(q2, (hd,)); k2 = F.rms_norm(k2, (hd,))
        if li == LAYER:
            if qov1 is not None: q = q.clone(); q[:, :, HEAD, :] = qov1
            if qov2 is not None: q2 = q2.clone(); q2[:, :, HEAD, :] = qov2
        q = apply_rot(q, cos, sin); k = apply_rot(k, cos, sin); q2 = apply_rot(q2, cos, sin); k2 = apply_rot(k2, cos, sin)
        sc = torch.einsum("bqhd,bkhd->bhqk", q, k) / hd; sc2 = torch.einsum("bqhd,bkhd->bhqk", q2, k2) / hd
        mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        pat = (sc * sc2).masked_fill(~mask, 0.0)
        y = torch.einsum("bhqk,bkhd->bqhd", pat, v).reshape(B, T, C); y = a.c_proj(y); x = x + y
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    x = F.rms_norm(x, (x.size(-1),)); logits = 30 * torch.tanh(m.lm_head(x) / 30)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1))

# ---- data ----
ds = datasets.load_dataset("NeelNanda/pile-10k", split="train", streaming=True); docs = []
for d in ds:
    t = tok(d["text"])["input_ids"]
    if len(t) >= 66: docs.append(t[:66])
    if len(docs) >= 24: break
toks = torch.tensor(docs, device=DEV); idx = toks[:, :-1].contiguous(); tgt = toks[:, 1:].contiguous()
with torch.no_grad(): ce_ref = float(m(idx, tgt)); assert abs(ce_ref - float(forward_ce(idx, tgt))) < 1e-3
B, Tm = idx.shape
valid = [(b, q) for b in range(B) for q in range(2, Tm)]

# ---- (1) weights-only alignment A_w for every head ----
Aw = np.zeros((nL, nh))
for L in range(nL):
    a = m.transformer.h[L].attn
    Wq = a.c_q.weight.detach(); Wq2 = a.c_q2.weight.detach()   # (d_model, d_model) rows = out
    for H in range(nh):
        Wqh = Wq[H * hd:(H + 1) * hd]      # (hd, d_model)
        Wq2h = Wq2[H * hd:(H + 1) * hd]
        num = torch.linalg.matrix_norm(Wqh @ Wq2h.T)           # ||Wq_h Wq2_h^T||_F
        den = torch.linalg.matrix_norm(Wqh) * torch.linalg.matrix_norm(Wq2h)
        Aw[L, H] = float(num / den.clamp_min(1e-9))

# ---- (2) data-level readouts + |cos(q1,q2)| for every head ----
resid_by_L = {}
for L in range(nL):
    st = {}; hk = m.transformer.h[L].register_forward_hook(lambda mm, i, o: st.__setitem__("r", i[0].detach()))
    with torch.no_grad(): m(idx, tgt)
    hk.remove(); resid_by_L[L] = F.rms_norm(st["r"], (st["r"].shape[-1],)).detach()

def readouts(L, H):
    a = m.transformer.h[L].attn; Xn = resid_by_L[L]
    q1 = F.rms_norm((Xn @ a.c_q.weight.detach().T).view(B, Tm, nh, hd)[:, :, H, :], (hd,)).detach()
    q2 = F.rms_norm((Xn @ a.c_q2.weight.detach().T).view(B, Tm, nh, hd)[:, :, H, :], (hd,)).detach()
    return Xn, q1, q2

datacos = np.zeros((nL, nh)); gqdmg = np.zeros((nL, nh))
for L in range(nL):
    for H in range(nh):
        _, q1, q2 = readouts(L, H)
        q1n = F.normalize(torch.stack([q1[b, q] for (b, q) in valid]), dim=1)
        q2n = F.normalize(torch.stack([q2[b, q] for (b, q) in valid]), dim=1)
        datacos[L, H] = float((q1n * q2n).sum(1).abs().mean())

# ---- pick a spread of heads across A_w range, plus causal-importance ----
allh = [(L, H) for L in range(nL) for H in range(nh)]
order = sorted(allh, key=lambda lh: Aw[lh])
picks = [order[int(r * (len(order) - 1))] for r in np.linspace(0, 1, 12)]
picks = sorted(set(picks))
print(f"500M two-QK. baseline CE {ce_ref:.4f}. A_w range [{Aw.min():.3f},{Aw.max():.3f}] mean {Aw.mean():.3f}")
print(f"data |cos(q1,q2)| range [{datacos.min():.3f},{datacos.max():.3f}] mean {datacos.mean():.3f}")
print(f"corr(A_w, data|cos|) over all {len(allh)} heads = {np.corrcoef(Aw.ravel(), datacos.ravel())[0,1]:+.3f}\n")

# ---- (3) causal gain on the spread ----
def cmeans(feat, L2):
    return [torch.stack([feat[b, q] for (b, q) in valid if L2[b, q] == c]).mean(0) if (L2 == c).any()
            else feat.mean((0, 1)) for c in range(K)]
def override(feat, L2, mns):
    ov = feat.clone()
    for (b, q) in valid:
        c = L2[b, q]
        if c >= 0: ov[b, q] = mns[c]
    return ov
def labels(mat, seed):
    l = KMeans(K, n_init=6, random_state=seed).fit_predict(mat); L2 = np.full((B, Tm), -1)
    for i, (b, q) in enumerate(valid): L2[b, q] = l[i]
    return L2
def sw(L, H, q1, q2, L2):
    m1, m2 = cmeans(q1, L2), cmeans(q2, L2)
    cw = float(forward_ce(idx, tgt, L, H, override(q1, L2, m1), override(q2, L2, m2)))
    cs = float(forward_ce(idx, tgt, L, H, override(q1, L2, [m1[(c+1)%K] for c in range(K)]),
                          override(q2, L2, [m2[(c+1)%K] for c in range(K)])))
    return cs - cw

print(f"{'head':7s} {'A_w':>6s} {'d|cos|':>7s} {'q1only':>8s} {'q2only':>8s} {'both':>8s} {'gain':>8s}")
rows = []
for (L, H) in picks:
    Xn, q1, q2 = readouts(L, H)
    q1n = F.normalize(torch.stack([q1[b, q] for (b, q) in valid]), dim=1)
    q2n = F.normalize(torch.stack([q2[b, q] for (b, q) in valid]), dim=1)
    both = torch.cat([q1n, q2n], 1).cpu().numpy()
    e1 = np.array([sw(L, H, q1, q2, labels(q1n.cpu().numpy(), s)) for s in range(5)])
    e2 = np.array([sw(L, H, q1, q2, labels(q2n.cpu().numpy(), s)) for s in range(5)])
    eb = np.array([sw(L, H, q1, q2, labels(both, s)) for s in range(5)])
    gain = eb.mean() - max(e1.mean(), e2.mean())
    rows.append((Aw[L, H], datacos[L, H], e1.mean(), e2.mean(), eb.mean(), gain))
    print(f"L{L}H{H:<4d} {Aw[L,H]:6.3f} {datacos[L,H]:7.3f} {e1.mean():+8.4f} {e2.mean():+8.4f} {eb.mean():+8.4f} {gain:+8.4f}", flush=True)
rows = np.array(rows)
print(f"\ncorr(A_w, gain)      = {np.corrcoef(rows[:,0], rows[:,5])[0,1]:+.3f}")
print(f"corr(data|cos|, gain) = {np.corrcoef(rows[:,1], rows[:,5])[0,1]:+.3f}")
print("hypothesis: both correlations NEGATIVE (more independent circuits => bigger gain from folding in q2).")
print("A_w is the WEIGHTS-ONLY pre-screen; data|cos| the data-informed one (S11 pattern).")
print("DONE")
