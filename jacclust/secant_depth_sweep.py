"""tick 46 (autonomous): bilinear-secant SAE across ALL 18 layers of the 500M — depth profile of operator-sparsity.

tick 44: L6 strong (FVU 0.117), L12 weak (0.43). Map the whole model: per layer, sparse-SAE secant FVU +
functional FVU + the sparsity margin (dense rank-32 vs sparse m=512/k=32; from tick 45 the gap tells whether
that layer's maps are genuinely sparse-operator vs just low-rank). 2 seeds, 4000 steps (tick 44 seed-variance
was ±0.001, so 2 seeds suffices; noted per standing rule). One forward hooks all blocks.
"""
import sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/workspace/tensor_language")
from huggingface_hub import hf_hub_download
import jacclust.tt_model as TT
torch.set_default_dtype(torch.float32); DEV = "cuda"

class BSAE(torch.nn.Module):
    def __init__(self, d, m):
        super().__init__(); self.q = torch.nn.Parameter(torch.randn(m, d) / d ** 0.5); self.p = torch.nn.Parameter(torch.randn(m, d) / d ** 0.5)
    def enc(self, xp, y): return (xp @ self.q.T) * (y @ self.p.T)
    def tk(self, z, k):
        if k >= z.shape[1]: return z
        val, ix = z.abs().topk(k, 1); return torch.zeros_like(z).scatter_(1, ix, torch.gather(z, 1, ix))
    def err(self, xp, y, k):
        z = self.tk(self.enc(xp, y), k)
        cross = (z * (y @ self.p.T) * (xp @ self.q.T)).sum(1)
        quad = torch.einsum("ti,ij,tj->t", z, (self.p @ self.p.T) * (self.q @ self.q.T), z)
        mnorm = (y ** 2).sum(1) * (xp ** 2).sum(1); return (mnorm - 2 * cross + quad).clamp_min(0), mnorm, z

def fit(h, y, m, k, seed, steps=4000):
    n, d = h.shape; hp = h / (h ** 2).sum(1, keepdim=True).clamp_min(1e-9)
    sae = BSAE(d, m).to(DEV); opt = torch.optim.Adam(sae.parameters(), lr=2e-3); g = torch.Generator(device=DEV).manual_seed(seed)
    for _ in range(steps):
        bi = torch.randint(0, n, (2048,), generator=g, device=DEV)
        e, mn, _ = sae.err(hp[bi], y[bi], k); (e.sum() / mn.sum()).backward(); opt.step(); opt.zero_grad()
    with torch.no_grad():
        e, mn, z = sae.err(hp, y, k); fvu = float(e.sum() / mn.sum())
        Mh = torch.einsum("ti,id->td", z * (h @ sae.q.T), sae.p); ffvu = float(((y - Mh) ** 2).sum() / (y ** 2).sum())
    return fvu, ffvu

repo = "Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd"
cfg = json.load(open(hf_hub_download(repo, "config.json"))); cfg.pop("step", None)
m = TT.GPT(TT.GPTConfig(**cfg)).to(DEV).eval()
m.load_state_dict(torch.load(hf_hub_download(repo, "pytorch_model.bin"), map_location=DEV, weights_only=True))
from transformers import AutoTokenizer; tk = AutoTokenizer.from_pretrained("gpt2")
import datasets
ds = datasets.load_dataset("NeelNanda/pile-10k", split="train", streaming=True); docs = []
for dch in ds:
    t = tk(dch["text"])["input_ids"]
    if len(t) >= 129: docs.append(t[:129])
    if len(docs) >= 96: break
idx = torch.tensor(docs, device=DEV)[:, :-1].contiguous()
d = cfg["n_embd"]; nL = cfg["n_layer"]
store = {}
hks = [m.transformer.h[L].register_forward_hook((lambda L: lambda mm, i, o: store.__setitem__(L, i[0].detach()))(L)) for L in range(nL)]
with torch.no_grad(): m(idx, idx)
for hk in hks: hk.remove()

print(f"500M bilinear-secant SAE depth sweep. d={d}, {idx.numel()} tokens, sparse m=512/k=32, dense rank-32. 2 seeds.\n")
print(f"{'layer':6s} {'sparse FVU':>11s} {'functional':>11s} {'dense-32 FVU':>13s} {'sparsity margin':>16s}")
rows = []
for L in range(nL):
    h = F.rms_norm(store[L].reshape(-1, d), (d,)).detach()
    with torch.no_grad(): y = m.transformer.h[L].mlp(h).detach()
    fs = np.mean([fit(h, y, 512, 32, s)[0] for s in range(2)])
    ff = np.mean([fit(h, y, 512, 32, s)[1] for s in range(2)])
    fd = np.mean([fit(h, y, 32, 32, s)[0] for s in range(2)])
    margin = fd - fs
    rows.append((L, fs, ff, fd, margin))
    print(f"  L{L:<4d} {fs:11.3f} {ff:11.3f} {fd:13.3f} {margin:+16.3f}", flush=True)
R = np.array([r[1:] for r in rows])
print(f"\n  best (lowest sparse FVU) layer L{int(np.argmin(R[:,0]))}: {R[:,0].min():.3f}; worst L{int(np.argmax(R[:,0]))}: {R[:,0].max():.3f}")
print(f"  mean sparse FVU {R[:,0].mean():.3f}; mean sparsity margin (dense-sparse) {R[:,3].mean():+.3f} (>0 => sparse helps)")
print("DONE")
