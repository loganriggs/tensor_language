"""tick 52 (Logan): bilinear-secant SAE — BATCH-TopK + sweep TOTAL FEATURES (lottery-ticket) + sweep k + hist.

Logan: drop 10x-data; the headline is the LOTTERY-TICKET test — does FVU improve as total #features m grows
(more random atoms -> more chances one is well-aligned)? Sweep m at fixed avg-k. Also sweep k. BatchTopK (top
B*k over the batch -> variable features/token, splits datapoints by complexity). Correct MLP-input hook,
held-out FVU (tick-51 fix). 500M L8. Modest data (~55k train / 20k held-out).
"""
import sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/workspace/tensor_language")
from huggingface_hub import hf_hub_download
import jacclust.tt_model as TT
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float32); DEV = "cuda"

class BSAE(torch.nn.Module):
    def __init__(s, d, m):
        super().__init__(); s.q = torch.nn.Parameter(torch.randn(m, d) / d ** 0.5); s.p = torch.nn.Parameter(torch.randn(m, d) / d ** 0.5)
    def enc(s, xp, y): return (xp @ s.q.T) * (y @ s.p.T)
    def batch_topk(s, z, k):
        nkeep = z.shape[0] * k; flat = z.abs().reshape(-1)
        thr = flat.kthvalue(flat.numel() - nkeep).values
        return z * (z.abs() > thr), float(thr)
    def spar(s, z, thr): return z * (z.abs() > thr)
    def loss_from_z(s, z, xp, y):
        cross = (z * (y @ s.p.T) * (xp @ s.q.T)).sum(1)
        quad = torch.einsum("ti,ij,tj->t", z, (s.p @ s.p.T) * (s.q @ s.q.T), z)
        mnorm = (y ** 2).sum(1) * (xp ** 2).sum(1); return (mnorm - 2 * cross + quad).clamp_min(0), mnorm

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
    if len(docs) >= 620: break
idx = torch.tensor(docs, device=DEV)[:, :-1].contiguous(); d = cfg["n_embd"]
L = 8; blk = m.transformer.h[L]; buf = {"h": [], "y": []}
pre = blk.mlp.register_forward_pre_hook(lambda mm, i: buf["h"].append(i[0].detach().reshape(-1, d).cpu()))
post = blk.mlp.register_forward_hook(lambda mm, i, o: buf["y"].append(o.detach().reshape(-1, d).cpu()))
with torch.no_grad():
    for s0 in range(0, idx.shape[0], 24): m(idx[s0:s0+24], idx[s0:s0+24])
pre.remove(); post.remove()
h = torch.cat(buf["h"], 0); y = torch.cat(buf["y"], 0)
n = h.shape[0]; perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
te = perm[:20000]; tr = perm[20000:]
hte = (h[te] / (h[te] ** 2).sum(1, keepdim=True).clamp_min(1e-9)).to(DEV); yte = y[te].to(DEV)
print(f"500M L{L}: {n} tokens, {len(tr)} train / 20000 held-out. BatchTopK, correct MLP-input hook.\n")

def eval_fvu(sae, thr):
    num = den = 0.0; acts = []
    for i in range(0, hte.shape[0], 8000):
        z = sae.spar(sae.enc(hte[i:i+8000], yte[i:i+8000]), thr)
        e, mn = sae.loss_from_z(z, hte[i:i+8000], yte[i:i+8000]); num += float(e.sum()); den += float(mn.sum())
        acts.append((z.abs() > 0).sum(1).cpu())
    return num / den, torch.cat(acts)

def train(m_dict, k, steps=3000, bs=2048):
    sae = BSAE(d, m_dict).to(DEV); opt = torch.optim.Adam(sae.parameters(), lr=2e-3)
    g = torch.Generator().manual_seed(0); thr_ema = None
    for step in range(steps):
        bi = tr[torch.randint(0, len(tr), (bs,), generator=g)]
        hb = h[bi]; xp = (hb / (hb ** 2).sum(1, keepdim=True).clamp_min(1e-9)).to(DEV); yb = y[bi].to(DEV)
        z, thr = sae.batch_topk(sae.enc(xp, yb), k)
        thr_ema = thr if thr_ema is None else 0.99 * thr_ema + 0.01 * thr
        e, mn = sae.loss_from_z(z, xp, yb); (e.sum() / mn.sum()).backward(); opt.step(); opt.zero_grad()
    return sae, thr_ema

# --- LOTTERY TICKET: sweep total #features m at fixed avg-k=32 ---
print("=== (A) LOTTERY-TICKET: FVU vs total #features (BatchTopK, avg-k=32) ===")
ms = [256, 512, 1024, 2048, 4096, 8192]; msweep = []
for md in ms:
    sae, thr = train(md, 32); fvu, acts = eval_fvu(sae, thr)
    used = int((acts > 0).float().mul(0).add((torch.bincount(torch.arange(1)))).sum()) if False else 0
    msweep.append((md, fvu, float(acts.float().mean())))
    print(f"  m={md:>5d}:  held-out FVU {fvu:.3f}   feats/token {acts.float().mean():.1f}", flush=True)

# --- k sweep at m=4096 ---
print("\n=== (B) k-sweep (m=4096, BatchTopK) ===")
ks = [8, 16, 32, 64, 128]; ksweep = []; acts32 = None
for k in ks:
    sae, thr = train(4096, k); fvu, acts = eval_fvu(sae, thr)
    ksweep.append((k, fvu, float(acts.float().mean())))
    print(f"  avg-k={k:>4d}:  held-out FVU {fvu:.3f}   feats/token {acts.float().mean():.1f} (range {int(acts.min())}-{int(acts.max())})", flush=True)
    if k == 32: acts32 = acts.numpy()

M = np.array(msweep); K = np.array(ksweep)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(M[:, 0], M[:, 1], "o-"); ax[0].set_xscale("log", base=2); ax[0].set_xlabel("total #features m"); ax[0].set_ylabel("held-out secant FVU"); ax[0].set_title("(A) lottery ticket: FVU vs #features (avg-k=32)"); ax[0].grid(alpha=.3)
ax[1].plot(K[:, 0], K[:, 1], "s-", color="darkorange"); ax[1].set_xscale("log", base=2); ax[1].set_xlabel("avg-k (features/token)"); ax[1].set_ylabel("held-out secant FVU"); ax[1].set_title("(B) FVU vs k (m=4096)"); ax[1].grid(alpha=.3)
ax[2].hist(acts32, bins=range(0, int(acts32.max()) + 2), color="steelblue", edgecolor="k", lw=.2)
ax[2].axvline(32, color="r", ls="--", label="avg=32"); ax[2].set_xlabel("features per datapoint (recon rank)"); ax[2].set_ylabel("# tokens"); ax[2].set_title("(C) BatchTopK feats/datapoint (m=4096, avg-k=32)"); ax[2].legend()
plt.tight_layout(); plt.savefig("jacclust/bsae_scaled.png", dpi=110)
print(f"\n  feats/datapoint (avg-k=32): mean {acts32.mean():.1f}, median {np.median(acts32):.0f}, range {acts32.min()}-{acts32.max()}")
print("  saved jacclust/bsae_scaled.png")
print("DONE")
