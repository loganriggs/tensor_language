"""tick 35 (Logan): do the ORIGINAL method for attention — the secant operator M_q = z_q x_q^T (y@x^-1),
not activation-clustering. cos(M_q,M_q') = cos(x_q,x_q')·cos(z_q,z_q'). Compare to the readout method.

Original method clusters by the OPERATOR (Jacobian / rank-1 secant y x^T), cos = cos_x·cos_y. Ticks 27-34
clustered ACTIVATIONS (q1,q2,OV readouts). This tests the faithful analog:
  input  x_q = post-norm residual (what the head reads)
  output z_q = Σ_k s_qk v_k     (head output for this query; the full contracted read)
  secant M_q = z_q x_q^T   -> feature vec(z_q ⊗ x_q~), cosine = cos_x·cos_z   (x reduced to top-256 PCA)
Compare features {x, query readout [q1;q2], secant M=z x^T, z alone} under the SAME sensitive intervention
(override q1,q2), 5 seeds. (Full Jacobian ∂z/∂x was tick 23: causally ≈ random — context-contaminated;
the secant is the cheaper rank-1 analog, tested here for the first time on attention.)
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
K = 8; nh, hd = m.transformer.h[0].attn.n_head, m.transformer.h[0].attn.head_dim

def apply_rot(x, c, s):
    d = x.shape[-1] // 2; x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * c + x2 * s, -x1 * s + x2 * c], -1)

def forward_ce(idx, tgt, LAYER=-1, HEAD=-1, qov1=None, qov2=None):
    x = m.transformer.wte(idx); x = F.rms_norm(x, (x.size(-1),)); x0 = x; v1 = None
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0; a = blk.attn
        h = F.rms_norm(x, (x.size(-1),)); B, T, C = h.shape
        q = a.c_q(h).view(B, T, nh, hd); k = a.c_k(h).view(B, T, nh, hd)
        q2 = a.c_q2(h).view(B, T, nh, hd); k2 = a.c_k2(h).view(B, T, nh, hd); v = a.c_v(h).view(B, T, nh, hd)
        if v1 is None: v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        cos, sin = a.rotary(q); cos = cos.float(); sin = sin.float()
        q = F.rms_norm(q, (hd,)); k = F.rms_norm(k, (hd,)); q2 = F.rms_norm(q2, (hd,)); k2 = F.rms_norm(k2, (hd,))
        if li == LAYER:
            if qov1 is not None: q = q.clone(); q[:, :, HEAD, :] = qov1
            if qov2 is not None: q2 = q2.clone(); q2[:, :, HEAD, :] = qov2
        q = apply_rot(q, cos, sin); k = apply_rot(k, cos, sin); q2 = apply_rot(q2, cos, sin); k2 = apply_rot(k2, cos, sin)
        sc = torch.einsum("bqhd,bkhd->bhqk", q, k) / hd; sc2 = torch.einsum("bqhd,bkhd->bhqk", q2, k2) / hd
        mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool)); pat = (sc * sc2).masked_fill(~mask, 0.0)
        y = torch.einsum("bhqk,bkhd->bqhd", pat, v).reshape(B, T, C); y = a.c_proj(y); x = x + y
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    x = F.rms_norm(x, (x.size(-1),)); logits = 30 * torch.tanh(m.lm_head(x) / 30)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1))

ds = datasets.load_dataset("NeelNanda/pile-10k", split="train", streaming=True); docs = []
for d in ds:
    t = tok(d["text"])["input_ids"]
    if len(t) >= 66: docs.append(t[:66])
    if len(docs) >= 32: break
toks = torch.tensor(docs, device=DEV); idx = toks[:, :-1].contiguous(); tgt = toks[:, 1:].contiguous()
with torch.no_grad(): ce_ref = float(m(idx, tgt)); assert abs(ce_ref - float(forward_ce(idx, tgt))) < 1e-3
B, Tm = idx.shape; valid = [(b, q) for b in range(B) for q in range(2, Tm)]
Wv0 = m.transformer.h[0].attn.c_v.weight.detach()
st0 = {}; hk0 = m.transformer.h[0].register_forward_hook(lambda mm, i, o: st0.__setitem__("v0", F.rms_norm(i[0].detach(), (i[0].shape[-1],))))
with torch.no_grad(): m(idx, tgt)
hk0.remove(); v1_full = (st0["v0"] @ Wv0.T).view(B, Tm, nh, hd)

def readouts(L, H):
    a = m.transformer.h[L].attn
    st = {}; hk = m.transformer.h[L].register_forward_hook(lambda mm, i, o: st.__setitem__("r", i[0].detach()))
    with torch.no_grad(): m(idx, tgt)
    hk.remove(); Xn = F.rms_norm(st["r"], (st["r"].shape[-1],))
    def hq(W): return F.rms_norm((Xn @ W.T).view(B, Tm, nh, hd)[:, :, H, :], (hd,))
    q1r, q2r = hq(a.c_q.weight.detach()), hq(a.c_q2.weight.detach())
    k1r, k2r = hq(a.c_k.weight.detach()), hq(a.c_k2.weight.detach())
    cos, sin = a.rotary(torch.zeros(1, Tm, nh, hd, device=DEV)); cos = cos[0, :, 0].float(); sin = sin[0, :, 0].float()
    q1, k1 = apply_rot(q1r, cos, sin), apply_rot(k1r, cos, sin); q2, k2 = apply_rot(q2r, cos, sin), apply_rot(k2r, cos, sin)
    vh = (Xn @ a.c_v.weight.detach().T).view(B, Tm, nh, hd)[:, :, H, :]
    Vmix = (1 - float(a.lamb)) * vh + float(a.lamb) * v1_full[:, :, H, :]
    s = (torch.einsum("bqd,bkd->bqk", q1, k1) / hd) * (torch.einsum("bqd,bkd->bqk", q2, k2) / hd)
    mask = torch.tril(torch.ones(Tm, Tm, device=DEV, dtype=torch.bool)); s = s.masked_fill(~mask, 0.0)
    z = torch.einsum("bqk,bkd->bqd", s, Vmix)                       # (B,Tm,hd) head output z_q
    return Xn.detach(), q1r.detach(), q2r.detach(), Vmix.detach(), z.detach()

def col(f): return torch.stack([f[b, q] for (b, q) in valid])
def nrm(f): return F.normalize(col(f), 1)
def cmeans(f, L2):
    return [torch.stack([f[b, q] for (b, q) in valid if L2[b, q] == c]).mean(0) if (L2 == c).any() else f.mean((0, 1)) for c in range(K)]
def override(f, L2, mns):
    ov = f.clone()
    for (b, q) in valid:
        c = L2[b, q]
        if c >= 0: ov[b, q] = mns[c]
    return ov
def labels(mat, seed):
    l = KMeans(K, n_init=6, random_state=seed).fit_predict(mat); L2 = np.full((B, Tm), -1)
    for i, (b, q) in enumerate(valid): L2[b, q] = l[i]
    return L2
def swap_within(L, H, q1, q2, L2):
    m1, m2 = cmeans(q1, L2), cmeans(q2, L2)
    cw = float(forward_ce(idx, tgt, L, H, override(q1, L2, m1), override(q2, L2, m2)))
    cs = float(forward_ce(idx, tgt, L, H, override(q1, L2, [m1[(c+1)%K] for c in range(K)]),
                          override(q2, L2, [m2[(c+1)%K] for c in range(K)])))
    return cs - cw

def pca(Xc, r):
    Xc = Xc - Xc.mean(0, keepdim=True); U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vh[:r].T

print(f"500M two-QK. baseline CE {ce_ref:.4f}, {len(valid)} tokens. Intervention: override q1,q2.\n")
for (L, H) in [(6, 3), (0, 2), (11, 5)]:
    Xn, q1, q2, Vm, z = readouts(L, H)
    xc = col(Xn); zc = col(z)
    xr = F.normalize(pca(xc, 256), 1); zr = F.normalize(zc, 1)                 # reduced input, output
    secant = F.normalize(torch.einsum("ni,nj->nij", zr, xr).reshape(len(valid), -1), 1)  # cos=cos_z·cos_x
    feats = {
        "x_q (activation)": nrm(Xn),
        "[q1;q2] query readout (activation)": torch.cat([nrm(q1), nrm(q2)], 1),
        "secant M=z x^T  (y@x^-1, ORIGINAL)": secant.cpu().numpy() if False else secant,
        "z_q output alone": zr,
    }
    print(f"  --- L{L}H{H} ---")
    for name, fm in feats.items():
        fmn = fm.cpu().numpy() if torch.is_tensor(fm) else fm
        vals = np.array([swap_within(L, H, q1, q2, labels(fmn, s)) for s in range(5)])
        print(f"    {name:38s} {vals.mean():+.4f}±{vals.std():.4f}", flush=True)
    rr = []
    for s in range(5):
        rl = np.random.RandomState(s).randint(0, K, len(valid)); L2 = np.full((B, Tm), -1)
        for i, (b, q) in enumerate(valid): L2[b, q] = rl[i]
        rr.append(swap_within(L, H, q1, q2, L2))
    print(f"    {'random':38s} {np.mean(rr):+.4f}±{np.std(rr):.4f}")
print("DONE")
