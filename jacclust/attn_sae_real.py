"""tick 39 (autonomous): real-model attention SAE — mechanism object (query) vs activations (residual),
judged by the CAUSAL swap-within test (label-free; the only real-model metric we trust — priority 4).

No feature labels exist on a real LLM, so 'better SAE' = features that carve tokens into more causally-distinct
query operations. Train matched TopK SAEs on the query readout q=[q1;q2] (mechanism object, tick 35) and on the
residual x, group each token by its argmax SAE feature, and run the query-override swap-within (tick 27/34
metric). Compare SAE-q vs SAE-x (fair: same method, same head, same m). k-means K=8 on q and x as reference.
Model: 500M two-QK, head L6H3 (good dynamic range). 5 seeds. Honest note: SAE argmax uses more groups than
k-means K=8, so SAE-vs-kmeans is confounded by group count; the clean comparison is SAE-q vs SAE-x.
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
nh, hd = m.transformer.h[0].attn.n_head, m.transformer.h[0].attn.head_dim
LAYER, HEAD = 6, 3

def apply_rot(x, c, s):
    d = x.shape[-1] // 2; x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * c + x2 * s, -x1 * s + x2 * c], -1)

def forward_ce(idx, tgt, qov1=None, qov2=None):
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
    if len(t) >= 82: docs.append(t[:82])
    if len(docs) >= 48: break
toks = torch.tensor(docs, device=DEV); idx = toks[:, :-1].contiguous(); tgt = toks[:, 1:].contiguous()
with torch.no_grad(): ce_ref = float(m(idx, tgt)); assert abs(ce_ref - float(forward_ce(idx, tgt))) < 1e-3
B, Tm = idx.shape; valid = [(b, q) for b in range(B) for q in range(2, Tm)]
a = m.transformer.h[LAYER].attn
st = {}; hk = m.transformer.h[LAYER].register_forward_hook(lambda mm, i, o: st.__setitem__("r", i[0].detach()))
with torch.no_grad(): m(idx, tgt)
hk.remove(); Xn = F.rms_norm(st["r"], (st["r"].shape[-1],)).detach()
def hq(W): return F.rms_norm((Xn @ W.T).view(B, Tm, nh, hd)[:, :, HEAD, :], (hd,)).detach()
q1, q2 = hq(a.c_q.weight.detach()), hq(a.c_q2.weight.detach())

def col(f): return torch.stack([f[b, q] for (b, q) in valid])
def cmeans(f, lab, ng):
    return [torch.stack([f[b, q] for (b, q) in valid if lab[(b, q)] == c]).mean(0) if any(lab[(b, q)] == c for (b, q) in valid) else f.mean((0, 1)) for c in range(ng)]
def override(f, lab, mns):
    ov = f.clone()
    for (b, q) in valid: ov[b, q] = mns[lab[(b, q)]]
    return ov
def swap_within(labarr, ng):
    lab = {(b, q): int(labarr[i]) for i, (b, q) in enumerate(valid)}
    m1, m2 = cmeans(q1, lab, ng), cmeans(q2, lab, ng)
    cw = float(forward_ce(idx, tgt, override(q1, lab, m1), override(q2, lab, m2)))
    cs = float(forward_ce(idx, tgt, override(q1, lab, [m1[(c + 1) % ng] for c in range(ng)]),
                          override(q2, lab, [m2[(c + 1) % ng] for c in range(ng)])))
    return cs - cw

def train_sae(F_, seed, M=32, KSP=4, STEPS=3000):
    Fm = F_.to(DEV); n, d = Fm.shape
    mu = Fm.mean(0); Fc = (Fm - mu); sc = Fc.norm(dim=1).mean().clamp_min(1e-6); Fc = Fc / sc
    g = torch.Generator(device=DEV).manual_seed(seed)
    We = (torch.randn(d, M, generator=g, device=DEV) / d ** 0.5).requires_grad_()
    Wd = We.detach().clone().T.contiguous().requires_grad_(); b = torch.zeros(M, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([We, Wd, b], lr=2e-3)
    for _ in range(STEPS):
        bi = torch.randint(0, n, (2048,), generator=g, device=DEV)
        pre = Fc[bi] @ We + b; val, ix = pre.topk(KSP, 1)
        z = torch.zeros_like(pre).scatter_(1, ix, torch.relu(val))
        loss = ((z @ Wd - Fc[bi]) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): Wd.data = Wd.data / Wd.data.norm(dim=1, keepdim=True).clamp_min(1e-6)
    with torch.no_grad():
        pre = Fc @ We + b; return pre.argmax(1).cpu().numpy(), M

Qm = torch.cat([F.normalize(col(q1), 1), F.normalize(col(q2), 1)], 1)
Xm = F.normalize(col(Xn), 1)
print(f"500M two-QK L{LAYER}H{HEAD} attention SAE (causal swap-within; higher=causally distinct). {len(valid)} tokens.\n")
def km(mat, seed, K=8):
    return KMeans(K, n_init=6, random_state=seed).fit_predict(mat)
print(f"{'method':30s} {'swap-within':>13s} {'#groups':>8s}")
for name, fn in [
    ("k-means q  [q1;q2] (K=8)", lambda s: (km(Qm.cpu().numpy(), s), 8)),
    ("k-means x  residual (K=8)", lambda s: (km(Xm.cpu().numpy(), s), 8)),
    ("SAE q  [q1;q2] (m=32,k=4)", lambda s: train_sae(Qm, s)),
    ("SAE x  residual (m=32,k=4)", lambda s: train_sae(Xm, s)),
]:
    vals = []; ngs = []
    for s in range(5):
        lab, ng = fn(s); vals.append(swap_within(lab, ng)); ngs.append(len(set(lab.tolist())))
    vals = np.array(vals)
    print(f"  {name:28s} {vals.mean():+.4f}±{vals.std():.4f}  {np.mean(ngs):6.1f}", flush=True)
rr = np.array([swap_within(np.random.RandomState(s).randint(0, 8, len(valid)), 8) for s in range(5)])
print(f"  {'random (K=8)':28s} {rr.mean():+.4f}±{rr.std():.4f}     8.0")
print("\n  Clean comparison: SAE-q vs SAE-x (same method). k-means as reference. SAE-vs-kmeans confounded by #groups.")
print("DONE")
