"""tick 33 (Logan): the FULL attention module as one (contractable) tensor + the VK read. How good, what cost.

Squared attention has NO softmax, so the head is multilinear and IS one tensor. Per query q (head space):
  z_q = sum_k (x_q^T W_QK1 x_k)(x_q^T W_QK2 x_k) (W_V x_k)         [W_QKi = Wqi^T Wki]
      = < x_q (x) x_q (x) (sum_k x_k(x)x_k(x)x_k) , T >,  T = W_QK1 (x) W_QK2 (x) W_V   (factored).
Dense T is hd*d^5 ~ 2.6e17 (1 EB, intractable); factored = 3 d x d matrices (~16MB), contracted at forward
cost. "VK" = the value contracted through key-selectivity, i.e. the realized read z_q = sum_k s_qk v_k
(no softmax needed). This script clusters queries by progressively more of the contraction and measures
causal quality with the SAME swap-within metric as ticks 23-32, but overriding ALL of (q1,q2,v) jointly by
the cluster representative (so every feature is judged by the same intervention, non-circular: cluster on a
readout, intervene on inputs). Controls: residual x, random. Causal heads L6H3,L8H3. 5 seeds.
"""
import sys, torch, json, numpy as np, time, torch.nn.functional as F
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

def forward_ce(idx, tgt, LAYER=-1, HEAD=-1, qov1=None, qov2=None, vov=None):
    x = m.transformer.wte(idx); x = F.rms_norm(x, (x.size(-1),)); x0 = x; v1 = None
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0; a = blk.attn
        h = F.rms_norm(x, (x.size(-1),)); B, T, C = h.shape
        q = a.c_q(h).view(B, T, nh, hd); k = a.c_k(h).view(B, T, nh, hd)
        q2 = a.c_q2(h).view(B, T, nh, hd); k2 = a.c_k2(h).view(B, T, nh, hd); v = a.c_v(h).view(B, T, nh, hd)
        if v1 is None: v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        if li == LAYER and vov is not None: v = v.clone(); v[:, :, HEAD, :] = vov
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
    if len(docs) >= 24: break
toks = torch.tensor(docs, device=DEV); idx = toks[:, :-1].contiguous(); tgt = toks[:, 1:].contiguous()
with torch.no_grad(): ce_ref = float(m(idx, tgt)); assert abs(ce_ref - float(forward_ce(idx, tgt))) < 1e-3
B, Tm = idx.shape; valid = [(b, q) for b in range(B) for q in range(2, Tm)]

Wv0 = m.transformer.h[0].attn.c_v.weight.detach()
st0 = {}; hk0 = m.transformer.h[0].register_forward_hook(lambda mm, i, o: st0.__setitem__("v0", F.rms_norm(i[0].detach(), (i[0].shape[-1],))))
with torch.no_grad(): m(idx, tgt)
hk0.remove(); v1_full = (st0["v0"] @ Wv0.T).view(B, Tm, nh, hd)

def readouts(L, H):
    """q1til,q2til (WHERE); Vmix (WHAT); the FULL realized read z_q = sum_k s_qk v_k (VK / full module); x."""
    a = m.transformer.h[L].attn
    st = {}; hk = m.transformer.h[L].register_forward_hook(lambda mm, i, o: st.__setitem__("r", i[0].detach()))
    with torch.no_grad(): m(idx, tgt)
    hk.remove(); Xn = F.rms_norm(st["r"], (st["r"].shape[-1],))
    def hq(W): return F.rms_norm((Xn @ W.T).view(B, Tm, nh, hd)[:, :, H, :], (hd,))
    q1r_ = hq(a.c_q.weight.detach()); q2r_ = hq(a.c_q2.weight.detach())
    k1r_ = hq(a.c_k.weight.detach()); k2r_ = hq(a.c_k2.weight.detach())
    cos, sin = a.rotary(torch.zeros(1, Tm, nh, hd, device=DEV)); cos = cos[0, :, 0].float(); sin = sin[0, :, 0].float()
    q1 = apply_rot(q1r_, cos, sin); k1 = apply_rot(k1r_, cos, sin); q2 = apply_rot(q2r_, cos, sin); k2 = apply_rot(k2r_, cos, sin)
    vh = (Xn @ a.c_v.weight.detach().T).view(B, Tm, nh, hd)[:, :, H, :]
    Vmix = (1 - float(a.lamb)) * vh + float(a.lamb) * v1_full[:, :, H, :]
    s = (torch.einsum("bqd,bkd->bqk", q1, k1) / hd) * (torch.einsum("bqd,bkd->bqk", q2, k2) / hd)
    mask = torch.tril(torch.ones(Tm, Tm, device=DEV, dtype=torch.bool)); s = s.masked_fill(~mask, 0.0)
    z = torch.einsum("bqk,bkd->bqd", s, Vmix)                        # (B,Tm,hd) FULL contracted read
    return (Xn.detach(), q1r_.detach(), q2r_.detach(), Vmix.detach(), z.detach())

def col(feat): return torch.stack([feat[b, q] for (b, q) in valid])
def cmeans(feat, L2):
    return [torch.stack([feat[b, q] for (b, q) in valid if L2[b, q] == c]).mean(0) if (L2 == c).any() else feat.mean((0, 1)) for c in range(K)]
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
def swap_within(L, H, q1, q2, Vm, L2):
    m1, m2, mv = cmeans(q1, L2), cmeans(q2, L2), cmeans(Vm, L2)
    cw = float(forward_ce(idx, tgt, L, H, override(q1, L2, m1), override(q2, L2, m2), override(Vm, L2, mv)))
    cs = float(forward_ce(idx, tgt, L, H, override(q1, L2, [m1[(c+1)%K] for c in range(K)]),
                          override(q2, L2, [m2[(c+1)%K] for c in range(K)]), override(Vm, L2, [mv[(c+1)%K] for c in range(K)])))
    return cs - cw

print(f"500M two-QK. baseline CE {ce_ref:.4f}. Intervention: joint (q1,q2,v) override by cluster rep.\n")
print(f"{'head':7s} {'feature clustered on':30s} {'swap-within':>13s}")
for (L, H) in [(6, 3), (8, 3)]:
    Xn, q1, q2, Vm, z = readouts(L, H)
    feats = {
        "query [q1;q2] (WHERE)": torch.cat([F.normalize(col(q1), 1), F.normalize(col(q2), 1)], 1),
        "value v (WHAT / VK values)": F.normalize(col(Vm), 1),
        "FULL read z=sum s_qk v_k (VK)": F.normalize(col(z), 1),
        "residual x (control)": F.normalize(col(Xn), 1),
    }
    print(f"  --- L{L}H{H} ---")
    for name, fm in feats.items():
        vals = np.array([swap_within(L, H, q1, q2, Vm, labels(fm.cpu().numpy(), s)) for s in range(5)])
        print(f"  {'':7s}{name:30s} {vals.mean():+.4f}±{vals.std():.4f}", flush=True)
    rr = []
    for s in range(5):
        rl = np.random.RandomState(s).randint(0, K, len(valid)); L2 = np.full((B, Tm), -1)
        for i, (b, q) in enumerate(valid): L2[b, q] = rl[i]
        rr.append(swap_within(L, H, q1, q2, Vm, L2))
    print(f"  {'':7s}{'random (control)':30s} {np.mean(rr):+.4f}±{np.std(rr):.4f}")
print("DONE")
