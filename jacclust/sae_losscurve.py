"""tick 51 (Logan): loss curve + data-size / held-out generalization check for the bilinear-secant SAE.

Concern: 12k-20k tokens may overfit (m=512 atoms reconstructing per-token secants). Test: train on N tokens,
eval FVU on TRAIN and a disjoint HELD-OUT set; sweep N in {12k, 48k, 150k}; log the loss curve. If held-out
>> train, it's memorizing; if held-out≈train and both flat vs N, the operator dictionary is real+data-sufficient.
500M layer 8 (the sparse, needs-y layer — most room for overfitting). m=512, k=32.
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
    def tk(s, z, k):
        val, ix = z.abs().topk(k, 1); return torch.zeros_like(z).scatter_(1, ix, torch.gather(z, 1, ix))
    def fvu(s, xp, y, k):
        z = s.tk(s.enc(xp, y), k)
        cross = (z * (y @ s.p.T) * (xp @ s.q.T)).sum(1)
        quad = torch.einsum("ti,ij,tj->t", z, (s.p @ s.p.T) * (s.q @ s.q.T), z)
        mnorm = (y ** 2).sum(1) * (xp ** 2).sum(1); return (mnorm - 2 * cross + quad).clamp_min(0).sum() / mnorm.sum(), z

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
    if len(docs) >= 1500: break
idx = torch.tensor(docs, device=DEV)[:, :-1].contiguous(); d = cfg["n_embd"]
L = 8; blk = m.transformer.h[L]; buf = {"h": [], "y": []}
# hook the MLP module: its INPUT is the true post-attention normed h; its OUTPUT is y. (correct MLP input)
pre = blk.mlp.register_forward_pre_hook(lambda mm, i: buf["h"].append(i[0].detach().reshape(-1, d).cpu()))
post = blk.mlp.register_forward_hook(lambda mm, i, o: buf["y"].append(o.detach().reshape(-1, d).cpu()))
with torch.no_grad():
    for s0 in range(0, idx.shape[0], 24):   # small chunks so the vocab-50k head fits
        m(idx[s0:s0+24], idx[s0:s0+24])
pre.remove(); post.remove()
h = torch.cat(buf["h"], 0).to(DEV); y = torch.cat(buf["y"], 0).to(DEV)
hp = h / (h ** 2).sum(1, keepdim=True).clamp_min(1e-9)
n_all = h.shape[0]
perm = torch.randperm(n_all, generator=torch.Generator().manual_seed(0))
te = perm[:20000].to(DEV); trpool = perm[20000:].to(DEV)   # 20k held-out, rest train pool
print(f"L{L}: {n_all} total tokens, 20000 held-out, train pool {len(trpool)}. m=512,k=32.\n")

def train(N, steps=6000, log=False):
    tr = trpool[:N]; sae = BSAE(d, 512).to(DEV); opt = torch.optim.Adam(sae.parameters(), lr=2e-3)
    g = torch.Generator(device=DEV).manual_seed(0); curve = []
    for step in range(steps):
        bi = tr[torch.randint(0, N, (2048,), generator=g, device=DEV)]
        fv, _ = sae.fvu(hp[bi], y[bi], 32); fv.backward(); opt.step(); opt.zero_grad()
        if log and step % 100 == 0:
            with torch.no_grad():
                ftr, _ = sae.fvu(hp[tr[:20000]], y[tr[:20000]], 32); fte, _ = sae.fvu(hp[te], y[te], 32)
            curve.append((step, float(ftr), float(fte)))
    with torch.no_grad():
        ftr, ztr = sae.fvu(hp[tr[:20000]], y[tr[:20000]], 32); fte, _ = sae.fvu(hp[te], y[te], 32)
        used = int(((ztr.abs() > 1e-6).float().mean(0) > 0.001).sum())
    return float(ftr), float(fte), used, curve

# data-size sweep
Ns = [12000, 48000, 150000]; res = []
for N in Ns:
    ftr, fte, used, _ = train(N); res.append((N, ftr, fte, used))
    print(f"  N_train={N:>7d}:  train FVU {ftr:.3f}   held-out FVU {fte:.3f}   gap {fte-ftr:+.3f}   used atoms {used}")
# loss curve for the largest run
_, _, _, curve = train(150000, steps=6000, log=True)
cur = np.array(curve)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(cur[:, 0], cur[:, 1], label="train FVU"); ax[0].plot(cur[:, 0], cur[:, 2], label="held-out FVU", ls="--")
ax[0].set_xlabel("step"); ax[0].set_ylabel("secant FVU"); ax[0].set_title("500M L8 loss curve (N=150k)"); ax[0].legend(); ax[0].grid(alpha=.3)
R = np.array(res)
ax[1].plot(R[:, 0], R[:, 1], "o-", label="train FVU"); ax[1].plot(R[:, 0], R[:, 2], "s--", label="held-out FVU")
ax[1].set_xscale("log"); ax[1].set_xlabel("# train tokens"); ax[1].set_ylabel("secant FVU"); ax[1].set_title("data-size: train vs held-out"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("jacclust/sae_losscurve.png", dpi=110)
print("\n  saved jacclust/sae_losscurve.png")
print("DONE")
