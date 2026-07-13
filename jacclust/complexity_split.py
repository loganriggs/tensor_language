"""tick 54 (autonomous): what drives the BatchTopK complexity split? (features-per-datapoint = operator rank)

tick 52: feats/datapoint is right-skewed (0 to 459). Characterize: does #features correlate with secant
magnitude ||M||, output norm, token frequency? What ARE the 0-feature vs many-feature tokens? 500M L8,
BatchTopK m=2048 k=32, correct MLP-input hook, held-out chars on a fresh eval set.
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
    def btk(s, z, k):
        nkeep = z.shape[0] * k; flat = z.abs().reshape(-1)
        thr = flat.kthvalue(flat.numel() - nkeep).values; return z * (z.abs() > thr), float(thr)
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
    if len(docs) >= 700: break
toks = torch.tensor(docs, device=DEV); idx = toks[:, :-1].contiguous(); d = cfg["n_embd"]; T = idx.shape[1]
L = 8; blk = m.transformer.h[L]; buf = {"h": [], "y": []}
pre = blk.mlp.register_forward_pre_hook(lambda mm, i: buf["h"].append(i[0].detach().reshape(-1, d).cpu()))
post = blk.mlp.register_forward_hook(lambda mm, i, o: buf["y"].append(o.detach().reshape(-1, d).cpu()))
with torch.no_grad():
    for s0 in range(0, idx.shape[0], 24): m(idx[s0:s0+24], idx[s0:s0+24])
pre.remove(); post.remove()
h = torch.cat(buf["h"], 0); y = torch.cat(buf["y"], 0); n = h.shape[0]
tok_ids = idx.reshape(-1).cpu().numpy()                      # token at each datapoint
perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)); te = perm[:25000]; tr = perm[25000:]

sae = BSAE(d, 2048).to(DEV); opt = torch.optim.Adam(sae.parameters(), lr=2e-3); g = torch.Generator().manual_seed(0); thr = None
for _ in range(3000):
    bi = tr[torch.randint(0, len(tr), (2048,), generator=g)]
    hb = h[bi]; xp = (hb / (hb ** 2).sum(1, keepdim=True).clamp_min(1e-9)).to(DEV); yb = y[bi].to(DEV)
    z, t = sae.btk(sae.enc(xp, yb), 32); thr = t if thr is None else 0.99 * thr + 0.01 * t
    e, mn = sae.lz(z, xp, yb); (e.sum() / mn.sum()).backward(); opt.step(); opt.zero_grad()

# eval on held-out: features per datapoint + covariates
tei = te.numpy()
hte = h[te]; yte = y[te]; hp = (hte / (hte ** 2).sum(1, keepdim=True).clamp_min(1e-9))
nfeat = []; Mnorm = []
with torch.no_grad():
    for i in range(0, len(te), 8000):
        xpc = hp[i:i+8000].to(DEV); ypc = yte[i:i+8000].to(DEV)
        z = sae.enc(xpc, ypc); z = z * (z.abs() > thr)
        nfeat.append((z.abs() > 0).sum(1).cpu().numpy())
        Mnorm.append((ypc ** 2).sum(1).cpu().numpy() / (hte[i:i+8000] ** 2).sum(1).clamp_min(1e-9).numpy())
nfeat = np.concatenate(nfeat); Mnorm = np.concatenate(Mnorm)
ynorm = (yte ** 2).sum(1).sqrt().numpy(); hnorm = (hte ** 2).sum(1).sqrt().numpy()
# token frequency (over whole corpus)
freq = np.bincount(tok_ids, minlength=50304); tokfreq = freq[tok_ids[tei]]

from scipy.stats import spearmanr
print(f"500M L8 complexity split: {len(te)} held-out tokens, BatchTopK m=2048 k=32. feats/datapoint stats:")
print(f"  mean {nfeat.mean():.1f}  median {np.median(nfeat):.0f}  range {nfeat.min()}-{nfeat.max()}  frac-zero {np.mean(nfeat==0):.3f}\n")
print("  Spearman correlations of #features-per-datapoint with:")
print(f"    secant norm ||M||^2 = ||y||^2/||h||^2 : {spearmanr(nfeat, Mnorm).correlation:+.3f}")
print(f"    output norm ||y||                     : {spearmanr(nfeat, ynorm).correlation:+.3f}")
print(f"    input  norm ||h||                     : {spearmanr(nfeat, hnorm).correlation:+.3f}")
print(f"    token corpus frequency                : {spearmanr(nfeat, tokfreq).correlation:+.3f}")

def toptoks(mask, n_top=18):
    ids = tok_ids[tei][mask]; c = np.bincount(ids, minlength=50304)
    return [repr(tok.decode([int(i)])) for i in np.argsort(-c)[:n_top] if c[np.argsort(-c)][0] > 0][:n_top]
print(f"\n  ZERO-feature tokens (n={np.sum(nfeat==0)}): {toptoks(nfeat==0)}")
print(f"  LOW (1-5 feats, n={np.sum((nfeat>=1)&(nfeat<=5))}): {toptoks((nfeat>=1)&(nfeat<=5))}")
hi = nfeat >= np.percentile(nfeat, 99)
print(f"  HIGH (top-1%, >={int(np.percentile(nfeat,99))} feats, n={np.sum(hi)}): {toptoks(hi)}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(np.log10(Mnorm + 1e-9), nfeat, s=2, alpha=.1); ax[0].set_xlabel("log10 ||M||^2 (secant norm)"); ax[0].set_ylabel("# features (operator rank)"); ax[0].set_title(f"L8: #features vs secant norm (rho={spearmanr(nfeat,Mnorm).correlation:+.2f})")
ax[1].scatter(np.log10(tokfreq + 1), nfeat, s=2, alpha=.1); ax[1].set_xlabel("log10 token corpus frequency"); ax[1].set_ylabel("# features"); ax[1].set_title(f"#features vs token frequency (rho={spearmanr(nfeat,tokfreq).correlation:+.2f})")
plt.tight_layout(); plt.savefig("jacclust/complexity_split.png", dpi=110)
print("\n  saved jacclust/complexity_split.png\nDONE")
