"""tick 57 (autonomous): best sparse dictionary for the FULL bilinear MLP? linear vs bilinear transcoder vs secant.

Extends tick 56 (linear readout) to the full layer y=D(Lh⊙Rh) (degree-2 in h), where the operator VARIES per
token. Three h-space dictionaries, m=1024 BatchTopK k=32, held-out output-FVU on 500M L8:
  A LINEAR transcoder:   a=TopK(W_enc h);              yhat=W_dec a          (can't capture degree-2 exactly)
  B BILINEAR transcoder: g=(A h)⊙(B h); a=TopK(g);     yhat=W_dec a          (matches layer structure; Dooms)
  C SECANT-SAE (tied):   M=y h+; sparse rank-1 recon;  yhat=Mhat h (sees y)  (operator approach, ticks 42-55)
Fair predictors = A,B (h-only). C sees y (flag). Prediction: bilinear transcoder best (=sparse hidden units).
"""
import sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/workspace/tensor_language")
from huggingface_hub import hf_hub_download
import jacclust.tt_model as TT
torch.set_default_dtype(torch.float32); DEV = "cuda"

def btk(zsel, k):
    nkeep = zsel.shape[0] * k; flat = zsel.abs().reshape(-1)
    thr = flat.kthvalue(flat.numel() - nkeep).values; return zsel.abs() > thr

repo = "Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd"
cfg = json.load(open(hf_hub_download(repo, "config.json"))); cfg.pop("step", None)
m = TT.GPT(TT.GPTConfig(**cfg)).to(DEV).eval()
m.load_state_dict(torch.load(hf_hub_download(repo, "pytorch_model.bin"), map_location=DEV, weights_only=True))
from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained("gpt2")
import datasets
ds = datasets.load_dataset("NeelNanda/pile-10k", split="train", streaming=True); docs = []
for dch in ds:
    t = tok(dch["text"])["input_ids"]
    if len(t) >= 129: docs.append(t[:129])
    if len(docs) >= 500: break
idx = torch.tensor(docs, device=DEV)[:, :-1].contiguous(); d = cfg["n_embd"]
L = 8; blk = m.transformer.h[L]; buf = {"h": [], "y": []}
pre = blk.mlp.register_forward_pre_hook(lambda mm, i: buf["h"].append(i[0].detach().reshape(-1, d).cpu()))
post = blk.mlp.register_forward_hook(lambda mm, i, o: buf["y"].append(o.detach().reshape(-1, d).cpu()))
with torch.no_grad():
    for s0 in range(0, idx.shape[0], 24): m(idx[s0:s0+24], idx[s0:s0+24])
pre.remove(); post.remove()
h = torch.cat(buf["h"], 0); y = torch.cat(buf["y"], 0); n = h.shape[0]
perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)); te = perm[:15000]; tr = perm[15000:]
hte = h[te].to(DEV); yte = y[te].to(DEV)
Mvar = float(((yte - yte.mean(0)) ** 2).sum())
print(f"500M L{L} full MLP: h,y={d}. {len(tr)} train / 15000 held. m=1024 BatchTopK k=32.\n")
M, K, ST = 1024, 32, 3000

def lin_tc(seed):
    We = (torch.randn(M, d, generator=torch.Generator().manual_seed(seed)) / d ** .5).to(DEV).requires_grad_()
    Wd = (torch.randn(d, M, generator=torch.Generator().manual_seed(seed + 1)) / M ** .5).to(DEV).requires_grad_()
    b = torch.zeros(M, device=DEV, requires_grad=True); opt = torch.optim.Adam([We, Wd, b], 2e-3); g = torch.Generator().manual_seed(seed)
    for _ in range(ST):
        bi = tr[torch.randint(0, len(tr), (2048,), generator=g)]; hb = h[bi].to(DEV); yb = y[bi].to(DEV)
        a = hb @ We.T + b; a = a * btk(a, K); yh = a @ Wd.T
        (((yh - yb) ** 2).sum(1) / (yb ** 2).sum(1).clamp_min(1e-9)).mean().backward(); opt.step(); opt.zero_grad()
    with torch.no_grad():
        a = hte @ We.T + b; a = a * btk(a, K); return float(((a @ Wd.T - yte) ** 2).sum() / (yte ** 2).sum())

def bilin_tc(seed):
    A = (torch.randn(M, d, generator=torch.Generator().manual_seed(seed)) / d ** .5).to(DEV).requires_grad_()
    B = (torch.randn(M, d, generator=torch.Generator().manual_seed(seed + 1)) / d ** .5).to(DEV).requires_grad_()
    Wd = (torch.randn(d, M, generator=torch.Generator().manual_seed(seed + 2)) / M ** .5).to(DEV).requires_grad_()
    opt = torch.optim.Adam([A, B, Wd], 2e-3); g = torch.Generator().manual_seed(seed)
    for _ in range(ST):
        bi = tr[torch.randint(0, len(tr), (2048,), generator=g)]; hb = h[bi].to(DEV); yb = y[bi].to(DEV)
        gg = (hb @ A.T) * (hb @ B.T); gg = gg * btk(gg, K); yh = gg @ Wd.T
        (((yh - yb) ** 2).sum(1) / (yb ** 2).sum(1).clamp_min(1e-9)).mean().backward(); opt.step(); opt.zero_grad()
    with torch.no_grad():
        gg = (hte @ A.T) * (hte @ B.T); gg = gg * btk(gg, K); return float(((gg @ Wd.T - yte) ** 2).sum() / (yte ** 2).sum())

def secant(seed):
    q = (torch.randn(M, d, generator=torch.Generator().manual_seed(seed)) / d ** .5).to(DEV).requires_grad_()
    p = (torch.randn(M, d, generator=torch.Generator().manual_seed(seed + 1)) / d ** .5).to(DEV).requires_grad_()
    opt = torch.optim.Adam([q, p], 2e-3); g = torch.Generator().manual_seed(seed)
    def err(c, hp, yy):
        cross = (c * (yy @ p.T) * (hp @ q.T)).sum(1); quad = torch.einsum("ti,ij,tj->t", c, (p @ p.T) * (q @ q.T), c)
        mn = (yy ** 2).sum(1) * (hp ** 2).sum(1); return (mn - 2 * cross + quad).clamp_min(0), mn
    for _ in range(ST):
        bi = tr[torch.randint(0, len(tr), (2048,), generator=g)]; hb = h[bi].to(DEV); yb = y[bi].to(DEV)
        hp = hb / (hb ** 2).sum(1, keepdim=True).clamp_min(1e-9); c = (hp @ q.T) * (yb @ p.T); c = c * btk(c, K)
        e, mn = err(c, hp, yb); (e / mn.clamp_min(1e-9)).mean().backward(); opt.step(); opt.zero_grad()
    with torch.no_grad():
        hp = hte / (hte ** 2).sum(1, keepdim=True).clamp_min(1e-9); c = (hp @ q.T) * (yte @ p.T); c = c * btk(c, K)
        e, mn = err(c, hp, yte); opf = float(e.sum() / mn.sum())
        yh = torch.einsum("ti,id->td", c * (hte @ q.T), p); ofv = float(((yh - yte) ** 2).sum() / (yte ** 2).sum())
    return opf, ofv

lt = lin_tc(0); bt = bilin_tc(0); sopf, sofv = secant(0)
print("=== held-out output-FVU (reconstruct y = full bilinear MLP output) ===")
print(f"  A linear transcoder  (h->y, fair)     output-FVU {lt:.3f}")
print(f"  B bilinear transcoder(h->y, fair)     output-FVU {bt:.3f}   <- matches layer structure (Dooms)")
print(f"  C secant-SAE (tied, sees y)            output-FVU {sofv:.3f}   operator-FVU {sopf:.3f}")
print(f"\n  fair predictors A,B are h-only; C sees y. bilinear TC should win (= sparse hidden units).")
print("DONE")
