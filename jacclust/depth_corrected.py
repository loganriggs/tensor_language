"""tick 53 (autonomous): CORRECTED depth sweep — faithful held-out FVU across all 18 layers.

Re-runs tick 46 with the tick-51/52 fixes: correct MLP-input hook (blk.mlp pre-hook), HELD-OUT FVU, BatchTopK.
Supersedes tick-46's rosy train-FVU-on-12k-wrong-hook numbers. Per layer: sparse (m=1024, BatchTopK avg-k=32)
held-out FVU + functional FVU, and dense rank-32 (all active) for the sparsity margin. ~55k train / 20k held.
Single seed (SAE FVU seed-variance ±~0.003 measured ticks 44/51/52 — fine for this smooth metric, unlike ARI).
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
        if nkeep >= flat.numel(): return z, float(flat.min()) - 1.0    # keep all (dense)
        thr = flat.kthvalue(flat.numel() - nkeep).values; return z * (z.abs() > thr), float(thr)
    def spar(s, z, thr): return z * (z.abs() > thr)
    def lz(s, z, xp, y):
        cross = (z * (y @ s.p.T) * (xp @ s.q.T)).sum(1)
        quad = torch.einsum("ti,ij,tj->t", z, (s.p @ s.p.T) * (s.q @ s.q.T), z)
        mn = (y ** 2).sum(1) * (xp ** 2).sum(1); return (mn - 2 * cross + quad).clamp_min(0), mn

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
    if len(docs) >= 600: break
idx = torch.tensor(docs, device=DEV)[:, :-1].contiguous(); d = cfg["n_embd"]; nL = cfg["n_layer"]
H = {L: [] for L in range(nL)}; Y = {L: [] for L in range(nL)}; hks = []
for L in range(nL):
    b = m.transformer.h[L].mlp
    hks.append(b.register_forward_pre_hook((lambda L: lambda mm, i: H[L].append(i[0].detach().reshape(-1, d).cpu()))(L)))
    hks.append(b.register_forward_hook((lambda L: lambda mm, i, o: Y[L].append(o.detach().reshape(-1, d).cpu()))(L)))
with torch.no_grad():
    for s0 in range(0, idx.shape[0], 24): m(idx[s0:s0+24], idx[s0:s0+24])
for hk in hks: hk.remove()
n = torch.cat(H[0], 0).shape[0]; perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
te = perm[:20000]; tr = perm[20000:]
print(f"500M CORRECTED depth sweep: {n} tokens, {len(tr)} train / 20000 held-out. m=1024 BatchTopK k=32, held-out FVU.\n")

def train_eval(h, y, m_dict, k, steps=2500):
    hp_te = (h[te] / (h[te] ** 2).sum(1, keepdim=True).clamp_min(1e-9)).to(DEV); y_te = y[te].to(DEV)
    sae = BSAE(d, m_dict).to(DEV); opt = torch.optim.Adam(sae.parameters(), lr=2e-3); g = torch.Generator().manual_seed(0); thr = None
    for _ in range(steps):
        bi = tr[torch.randint(0, len(tr), (2048,), generator=g)]
        hb = h[bi]; xp = (hb / (hb ** 2).sum(1, keepdim=True).clamp_min(1e-9)).to(DEV); yb = y[bi].to(DEV)
        z, t = sae.batch_topk(sae.enc(xp, yb), k); thr = t if thr is None else 0.99 * thr + 0.01 * t
        e, mn = sae.lz(z, xp, yb); (e.sum() / mn.sum()).backward(); opt.step(); opt.zero_grad()
    with torch.no_grad():
        num = den = fnum = fden = 0.0
        for i in range(0, hp_te.shape[0], 8000):
            xpc, ypc = hp_te[i:i+8000], y_te[i:i+8000]; z = sae.spar(sae.enc(xpc, ypc), thr)
            e, mn = sae.lz(z, xpc, ypc); num += float(e.sum()); den += float(mn.sum())
            Mh = torch.einsum("ti,id->td", z * ((h[te][i:i+8000].to(DEV)) @ sae.q.T), sae.p)
            fnum += float(((ypc - Mh) ** 2).sum()); fden += float((ypc ** 2).sum())
    return num / den, fnum / fden

rows = []
for L in range(nL):
    h = torch.cat(H[L], 0); y = torch.cat(Y[L], 0)
    fs, ff = train_eval(h, y, 1024, 32)
    fd, _ = train_eval(h, y, 32, 32)          # dense rank-32
    rows.append((L, fs, ff, fd, fd - fs))
    print(f"  L{L:<2d}  sparse FVU {fs:.3f}  functional {ff:.3f}  dense-32 {fd:.3f}  margin {fd-fs:+.3f}", flush=True)
R = np.array([r[1:] for r in rows])
print(f"\n  mean sparse FVU {R[:,0].mean():.3f} (tick-46 wrong-hook train was 0.37); best L{int(np.argmin(R[:,0]))} {R[:,0].min():.3f}, worst L{int(np.argmax(R[:,0]))} {R[:,0].max():.3f}")
print(f"  mean margin {R[:,3].mean():+.3f} (sparse beats dense-32 at all layers: {(R[:,3]>0).all()})")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(range(nL), R[:, 0], "o-", label="sparse (m=1024,k=32)"); ax[0].plot(range(nL), R[:, 2], "s--", label="dense rank-32", alpha=.6)
ax[0].set_xlabel("layer"); ax[0].set_ylabel("held-out secant FVU"); ax[0].set_title("500M CORRECTED depth profile (held-out, true MLP hook)"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].bar(range(nL), R[:, 3], color="teal"); ax[1].set_xlabel("layer"); ax[1].set_ylabel("sparsity margin (dense−sparse)"); ax[1].set_title("genuine-sparsity per layer"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("jacclust/depth_corrected.png", dpi=110)
print("  saved jacclust/depth_corrected.png\nDONE")
