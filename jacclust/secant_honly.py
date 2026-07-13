"""tick 47 (autonomous): does the secant reconstruction NEED y, or are the operators predictable from h alone?

The bilinear-secant SAE encoder z_i=(q_i·h^+)(p_i·y) SEES the output y -> it's an ANALYSIS tool (decomposes a
known map), not a predictor. Decision-relevant fork before building a predictive transcoder / running a
loss-recovered causal test: replace the encoder with an h-ONLY bilinear encoder z_i=(a_i·h)(b_i·h) (Dooms
same-input; sees only h), same rank-1 decode atoms p_i q_i^T, same secant target M=y h^+. If h-only reconstructs
nearly as well as full, the operators are h-predictable -> predictive transcoder viable. If it collapses, the
y-dependence is essential (pure analysis only). 500M L6 (low-rank), L8 (high sparsity-margin). 2 seeds.
"""
import sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/workspace/tensor_language")
from huggingface_hub import hf_hub_download
import jacclust.tt_model as TT
torch.set_default_dtype(torch.float32); DEV = "cuda"

class SecantSAE(torch.nn.Module):
    def __init__(self, d, m, honly=False):
        super().__init__(); self.honly = honly
        self.q = torch.nn.Parameter(torch.randn(m, d) / d ** 0.5)   # decode-right / (full: h^+ reader)
        self.p = torch.nn.Parameter(torch.randn(m, d) / d ** 0.5)   # decode-left  / (full: y reader)
        if honly:
            self.a = torch.nn.Parameter(torch.randn(m, d) / d ** 0.5)
            self.b = torch.nn.Parameter(torch.randn(m, d) / d ** 0.5)
    def enc(self, hp, y, h):
        return (h @ self.a.T) * (h @ self.b.T) if self.honly else (hp @ self.q.T) * (y @ self.p.T)
    def tk(self, z, k):
        val, ix = z.abs().topk(k, 1); return torch.zeros_like(z).scatter_(1, ix, torch.gather(z, 1, ix))
    def err(self, hp, y, h, k):
        z = self.tk(self.enc(hp, y, h), k)
        cross = (z * (y @ self.p.T) * (hp @ self.q.T)).sum(1)
        quad = torch.einsum("ti,ij,tj->t", z, (self.p @ self.p.T) * (self.q @ self.q.T), z)
        mnorm = (y ** 2).sum(1) * (hp ** 2).sum(1); return (mnorm - 2 * cross + quad).clamp_min(0), mnorm

def fit(h, y, honly, seed, m=512, k=32, steps=5000):
    n, d = h.shape; hp = h / (h ** 2).sum(1, keepdim=True).clamp_min(1e-9)
    sae = SecantSAE(d, m, honly).to(DEV); opt = torch.optim.Adam(sae.parameters(), lr=2e-3)
    g = torch.Generator(device=DEV).manual_seed(seed)
    for _ in range(steps):
        bi = torch.randint(0, n, (2048,), generator=g, device=DEV)
        e, mn = sae.err(hp[bi], y[bi], h[bi], k); (e.sum() / mn.sum()).backward(); opt.step(); opt.zero_grad()
    with torch.no_grad(): e, mn = sae.err(hp, y, h, k)
    return float(e.sum() / mn.sum())

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
idx = torch.tensor(docs, device=DEV)[:, :-1].contiguous(); d = cfg["n_embd"]

print(f"500M: does secant reconstruction need y? full-encoder (sees y) vs h-only encoder. FVU, 2 seeds.\n")
print(f"{'layer':7s} {'full (sees y)':>14s} {'h-only':>10s}  verdict")
for L in (6, 8, 16):
    blk = m.transformer.h[L]; st = {}
    hk = blk.register_forward_hook(lambda mm, i, o: st.__setitem__("r", i[0].detach()))
    with torch.no_grad(): m(idx, idx)
    hk.remove(); h = F.rms_norm(st["r"].reshape(-1, d), (d,)).detach()
    with torch.no_grad(): y = blk.mlp(h).detach()
    ff = np.mean([fit(h, y, False, s) for s in range(2)])
    fh = np.mean([fit(h, y, True, s) for s in range(2)])
    verd = "h-predictable (transcoder viable)" if fh < ff + 0.10 else "needs y (analysis only)"
    print(f"  L{L:<5d} {ff:14.3f} {fh:10.3f}  {verd}", flush=True)
print("\n(full sees the true output; h-only must PREDICT the operator from h. small gap => transcoder viable.)")
print("DONE")
