"""tick 55 (autonomous): is there operator complexity BEYOND magnitude? Magnitude-free BatchTopK.

tick 54: #features/datapoint ~ secant norm ||M|| (rho 0.945) -> the "complexity split" is a magnitude effect.
Fix: select atoms by the NORMALIZED alignment cos(M,atom) = z_i/||M|| (per-token), and score per-token FVU
(equal weight). Then #features = DIRECTIONAL complexity, magnitude-free. Question: does a real split remain
(corr with ||M|| -> 0 but residual structure), or does it collapse (magnitude was everything)?
500M L8, m=2048, avg-k=32. Compare RAW vs NORMALIZED BatchTopK on the same data.
"""
import sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/workspace/tensor_language")
from huggingface_hub import hf_hub_download
import jacclust.tt_model as TT
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
torch.set_default_dtype(torch.float32); DEV = "cuda"

class BSAE(torch.nn.Module):
    def __init__(s, d, m):
        super().__init__(); s.q = torch.nn.Parameter(torch.randn(m, d) / d ** 0.5); s.p = torch.nn.Parameter(torch.randn(m, d) / d ** 0.5)
    def enc(s, xp, y): return (xp @ s.q.T) * (y @ s.p.T)
    def perr(s, z, xp, y):   # per-token error and ||M||^2
        cross = (z * (y @ s.p.T) * (xp @ s.q.T)).sum(1)
        quad = torch.einsum("ti,ij,tj->t", z, (s.p @ s.p.T) * (s.q @ s.q.T), z)
        mn = (y ** 2).sum(1) * (xp ** 2).sum(1); return (mn - 2 * cross + quad).clamp_min(0), mn

def btk_mask(zsel, k):
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
    if len(docs) >= 700: break
idx = torch.tensor(docs, device=DEV)[:, :-1].contiguous(); d = cfg["n_embd"]
L = 8; blk = m.transformer.h[L]; buf = {"h": [], "y": []}
pre = blk.mlp.register_forward_pre_hook(lambda mm, i: buf["h"].append(i[0].detach().reshape(-1, d).cpu()))
post = blk.mlp.register_forward_hook(lambda mm, i, o: buf["y"].append(o.detach().reshape(-1, d).cpu()))
with torch.no_grad():
    for s0 in range(0, idx.shape[0], 24): m(idx[s0:s0+24], idx[s0:s0+24])
pre.remove(); post.remove()
h = torch.cat(buf["h"], 0); y = torch.cat(buf["y"], 0); n = h.shape[0]
perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)); te = perm[:25000]; tr = perm[25000:]

def train(normalize, steps=3000):
    sae = BSAE(d, 2048).to(DEV); opt = torch.optim.Adam(sae.parameters(), lr=2e-3); g = torch.Generator().manual_seed(0); thr = None
    for _ in range(steps):
        bi = tr[torch.randint(0, len(tr), (2048,), generator=g)]
        hb = h[bi]; xp = (hb / (hb ** 2).sum(1, keepdim=True).clamp_min(1e-9)).to(DEV); yb = y[bi].to(DEV)
        z = sae.enc(xp, yb); mn = (yb ** 2).sum(1) * (xp ** 2).sum(1)
        zsel = z / mn.sqrt().clamp_min(1e-9)[:, None] if normalize else z    # select by cos or by magnitude
        mask = btk_mask(zsel, 32); thr = _upd(thr, zsel, mask)
        zk = z * mask
        e, mnn = sae.perr(zk, xp, yb)
        loss = (e / mnn.clamp_min(1e-9)).mean() if normalize else (e.sum() / mnn.sum())
        loss.backward(); opt.step(); opt.zero_grad()
    return sae, thr

def _upd(thr, zsel, mask):
    t = float(zsel.abs()[mask].min()) if mask.any() else 0.0
    return t if thr is None else 0.99 * thr + 0.01 * t

def evalr(sae, thr, normalize):
    hte, yte = h[te], y[te]; hp = (hte / (hte ** 2).sum(1, keepdim=True).clamp_min(1e-9))
    nf = []; per = []; Mn = []
    with torch.no_grad():
        for i in range(0, len(te), 8000):
            xpc = hp[i:i+8000].to(DEV); ypc = yte[i:i+8000].to(DEV)
            z = sae.enc(xpc, ypc); mn = (ypc ** 2).sum(1) * (xpc ** 2).sum(1)
            zsel = z / mn.sqrt().clamp_min(1e-9)[:, None] if normalize else z
            mask = zsel.abs() > thr; zk = z * mask
            e, mnn = sae.perr(zk, xpc, ypc)
            nf.append(mask.sum(1).cpu().numpy()); per.append((e / mnn.clamp_min(1e-9)).cpu().numpy()); Mn.append(mn.cpu().numpy())
    return np.concatenate(nf), np.concatenate(per), np.concatenate(Mn)

print("500M L8: RAW (magnitude) vs NORMALIZED (cos) BatchTopK, m=2048 avg-k=32, 25k held-out.\n")
out = {}
for norm in (False, True):
    sae, thr = train(norm); nf, per, Mn = evalr(sae, thr, norm)
    rho = spearmanr(nf, Mn).correlation
    out[norm] = (nf, per, Mn, rho)
    tag = "NORMALIZED (cos)" if norm else "RAW (magnitude)"
    print(f"  {tag:20s}: feats/token mean {nf.mean():.1f} median {np.median(nf):.0f} range {nf.min()}-{nf.max()}  "
          f"| corr(#feat, ||M||^2) = {rho:+.3f}  | mean per-token FVU {per.mean():.3f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
for a, norm, ttl in [(ax[0], False, "RAW BatchTopK (magnitude)"), (ax[1], True, "NORMALIZED BatchTopK (cos)")]:
    nf, per, Mn, rho = out[norm]
    a.hist(nf, bins=range(0, int(nf.max()) + 2), color="steelblue" if not norm else "indianred", edgecolor="k", lw=.2)
    a.set_xlabel("features per datapoint"); a.set_ylabel("# tokens"); a.set_title(f"{ttl}\ncorr(#feat,||M||)={rho:+.2f}")
plt.tight_layout(); plt.savefig("jacclust/magfree.png", dpi=110)
print("\n  VERDICT: if NORMALIZED corr->0 AND its range collapses -> split was PURE magnitude.")
print("  saved jacclust/magfree.png\nDONE")
