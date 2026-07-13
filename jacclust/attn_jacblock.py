"""tick 36 (autonomous): WHERE is the attention-Jacobian contamination? Factor decomposition backing tick 35.

tick 23: full per-query attention Jacobian J_q=∂z_q/∂x_q ≈ causally random; query readout is causal.
tick 35: query readout = gate/selection factor of the per-query bilinear op; J_q re-adds context.
This pins down the contamination: J_q = Σ_k (∂p_qk/∂x_q) v_k — a sum of (query-space covector)⊗(context VALUE).
So J_q's clustering should align with the CONTEXT/OUTPUT (values), NOT the query. Test on 12l L6H0 (tick-23
head, single-QK squared, row-normalized). Compute J_q by autodiff; cluster by {J_q, query readout qtil,
output z_q, residual x, random}; measure causal swap-within (override query) AND ARI(J_q-clusters, each).
Predicted: qtil causal, J_q≈random causally; ARI(J_q, z_q/output) >> ARI(J_q, qtil) — contamination = values.
"""
import sys, torch, json, numpy as np, torch.nn.functional as F
from torch.func import jacrev
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
sys.path.insert(0, "/workspace/tensor_language")
from huggingface_hub import hf_hub_download
import jacclust.tt_model as TT

DEV = "cuda"; torch.set_default_dtype(torch.float32)
repo = "Elriggs/gpt2-bilinear-sqrd-attn-12l-6h-768embd"
cfg = json.load(open(hf_hub_download(repo, "config.json"))); cfg.pop("step", None)
m = TT.GPT(TT.GPTConfig(**cfg)).to(DEV).eval()
m.load_state_dict(torch.load(hf_hub_download(repo, "pytorch_model.bin"), map_location=DEV, weights_only=True))
from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained("gpt2")
import datasets
LAYER, HEAD, K = 6, 0, 8; nh, hd = m.transformer.h[0].attn.n_head, m.transformer.h[0].attn.head_dim

def apply_rot(x, c, s):
    d = x.shape[-1] // 2; x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * c + x2 * s, -x1 * s + x2 * c], -1)

def forward_ce(idx, tgt, qov=None):
    x = m.transformer.wte(idx); x = F.rms_norm(x, (x.size(-1),)); x0 = x; v1 = None
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0; a = blk.attn
        h = F.rms_norm(x, (x.size(-1),)); B, T, C = h.shape
        q = a.c_q(h).view(B, T, nh, hd); k = a.c_k(h).view(B, T, nh, hd); v = a.c_v(h).view(B, T, nh, hd)
        if v1 is None: v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        cos, sin = a.rotary(q); cos = cos.float(); sin = sin.float()
        q = F.rms_norm(q, (hd,)); k = F.rms_norm(k, (hd,))
        if li == LAYER and qov is not None: q = q.clone(); q[:, :, HEAD, :] = qov
        q = apply_rot(q, cos, sin); k = apply_rot(k, cos, sin)
        sc = torch.einsum("bqhd,bkhd->bhqk", q, k) / hd; mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        pat = sc.square().masked_fill(~mask, 0.0); pat = pat / pat.sum(-1, keepdim=True).clamp_min(1e-9)
        y = torch.einsum("bhqk,bkhd->bqhd", pat, v).reshape(B, T, C); y = a.c_proj(y); x = x + y
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    x = F.rms_norm(x, (x.size(-1),)); logits = 30 * torch.tanh(m.lm_head(x) / 30)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1))

ds = datasets.load_dataset("NeelNanda/pile-10k", split="train", streaming=True); docs = []
for d in ds:
    t = tok(d["text"])["input_ids"]
    if len(t) >= 49: docs.append(t[:49])
    if len(docs) >= 24: break
toks = torch.tensor(docs, device=DEV); idx = toks[:, :-1].contiguous(); tgt = toks[:, 1:].contiguous()
with torch.no_grad(): ce_ref = float(m(idx, tgt)); assert abs(ce_ref - float(forward_ce(idx, tgt))) < 1e-3
B, Tm = idx.shape
a = m.transformer.h[LAYER].attn; Wq, Wk, Wv = a.c_q.weight.detach(), a.c_k.weight.detach(), a.c_v.weight.detach()
st = {}; hk = m.transformer.h[LAYER].register_forward_hook(lambda mm, i, o: st.__setitem__("r", i[0].detach()))
with torch.no_grad(): m(idx, tgt)
hk.remove(); Xn = F.rms_norm(st["r"], (st["r"].shape[-1],)).detach()
cos, sin = a.rotary(torch.zeros(1, Tm, nh, hd, device=DEV)); cos, sin = cos[0, :, 0].float(), sin[0, :, 0].float()
qtil = F.rms_norm((Xn @ Wq.T).view(B, Tm, nh, hd)[:, :, HEAD, :], (hd,)).detach()
Kall = F.rms_norm((Xn @ Wk.T).view(B, Tm, nh, hd)[:, :, HEAD, :], (hd,)).detach()
Vall = (Xn @ Wv.T).view(B, Tm, nh, hd)[:, :, HEAD, :].detach()

def zfun(xq, b, q):
    qh = F.rms_norm((xq @ Wq.T).view(nh, hd)[HEAD], (hd,)); kq = F.rms_norm((xq @ Wk.T).view(nh, hd)[HEAD], (hd,)); vq = (xq @ Wv.T).view(nh, hd)[HEAD]
    Kn = torch.cat([Kall[b, :q], kq[None]], 0); Vv = torch.cat([Vall[b, :q], vq[None]], 0)
    qr = apply_rot(qh[None], cos[q], sin[q])[0]; Kr = apply_rot(Kn, cos[:q + 1], sin[:q + 1])
    s = (qr[None] * Kr).sum(-1) / hd; pat = s.square(); pat = pat / pat.sum().clamp_min(1e-9)
    return (pat[:, None] * Vv).sum(0)

# per-query Jacobian J_q = dz/dx_q, and the output z_q, for a token sample
samp = [(b, q) for b in range(B) for q in range(6, Tm)]           # skip earliest positions
rng = np.random.RandomState(0); rng.shuffle(samp); samp = samp[:320]
Jf, Zf, Qf, Xf = [], [], [], []
for (b, q) in samp:
    xq = Xn[b, q]
    Jf.append(jacrev(lambda x: zfun(x, b, q))(xq).flatten().detach())
    Zf.append(zfun(xq, b, q).detach()); Qf.append(qtil[b, q]); Xf.append(Xn[b, q])
Jf = torch.stack(Jf); Zf = torch.stack(Zf); Qf = torch.stack(Qf); Xf = torch.stack(Xf)

# also J restricted to the top principal directions vs the query -- but first the alignment story
def kml(mat, seed): return KMeans(K, n_init=6, random_state=seed).fit_predict(F.normalize(mat, 1).cpu().numpy())
print(f"12l L6H0 (single-QK squared). {len(samp)} tokens. baseline CE {ce_ref:.4f}\n")

# (1) causal swap-within: cluster by each, override query qtil by cluster mean
def swap_within(labs):
    LAB = {(b, q): labs[i] for i, (b, q) in enumerate(samp)}
    def ov(shift):
        o = qtil.clone()
        means = [Qf[[i for i in range(len(samp)) if labs[i] == c]].mean(0) if (labs == c).any() else Qf.mean(0) for c in range(K)]
        for (b, q) in samp: o[b, q] = means[(LAB[(b, q)] + shift) % K]
        return o
    with torch.no_grad():
        cw = float(forward_ce(idx, tgt, ov(0))); cs = float(forward_ce(idx, tgt, ov(1)))
    return cs - cw
print("=== causal swap-within (override query by cluster mean; higher=causally distinct), 5 seeds ===")
for name, mat in [("J_q full Jacobian", Jf), ("query readout qtil", Qf), ("output z_q", Zf), ("residual x", Xf)]:
    v = np.array([swap_within(kml(mat, s)) for s in range(5)])
    print(f"  {name:24s} {v.mean():+.4f}±{v.std():.4f}")
rr = np.array([swap_within(np.random.RandomState(s).randint(0, K, len(samp))) for s in range(5)])
print(f"  {'random':24s} {rr.mean():+.4f}±{rr.std():.4f}")

# (2) WHERE is the contamination: does J_q-clustering align with the OUTPUT/values or the query?
print("\n=== alignment of J_q-clusters with query vs output (ARI), 5 seeds ===")
aq = np.mean([adjusted_rand_score(kml(Jf, s), kml(Qf, s)) for s in range(5)])
az = np.mean([adjusted_rand_score(kml(Jf, s), kml(Zf, s)) for s in range(5)])
ax = np.mean([adjusted_rand_score(kml(Jf, s), kml(Xf, s)) for s in range(5)])
print(f"  ARI(J_q, query qtil) = {aq:+.3f}")
print(f"  ARI(J_q, output z_q) = {az:+.3f}")
print(f"  ARI(J_q, residual x) = {ax:+.3f}")
print("  contamination = whichever J_q aligns with. Predicted: output z_q (context values) >> query.")
# (3) norm decomposition: J_q = query-selectivity (x) value; check its output-space is spanned by context values
print(f"\n  ||J_q||_F mean {Jf.norm(dim=1).mean():.3f}; corr(|J_q flattened|, structure) — see ARIs above.")
print("DONE")
