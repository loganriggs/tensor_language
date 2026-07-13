"""tick 45 (autonomous): the control that could kill tick 43/44 — is the secant reconstruction genuinely
SPARSE-adaptive, or just globally LOW-RANK?

The bilinear-secant SAE (m=256/512, k active) reconstructs the real-MLP secant well. But if the secant
collection {M_t = y_t h_t^+} lives in a low-dim subspace, a small FIXED basis (dense rank-R, all components
active) would do as well and the 'operator dictionary' buys nothing. Control: sweep DENSE rank-R (m=R, k=R,
all active) and compare secant-FVU to the SPARSE SAE (m=256, k=16). If sparse achieves FVU that dense needs
rank >> 16 to match, sparsity/overcompleteness is real. If dense-16 ties sparse, it's just low-rank.
block2 MLP#0/#1 (d=128) full sweep; 500M layer-6 (d=1152) key points. 3 seeds.
"""
import sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/workspace/tensor_language")
torch.set_default_dtype(torch.float32); DEV = "cuda" if torch.cuda.is_available() else "cpu"

class BSAE(torch.nn.Module):
    def __init__(self, d, m):
        super().__init__(); self.q = torch.nn.Parameter(torch.randn(m, d) / d ** 0.5); self.p = torch.nn.Parameter(torch.randn(m, d) / d ** 0.5)
    def enc(self, xp, y): return (xp @ self.q.T) * (y @ self.p.T)
    def tk(self, z, k):
        if k >= z.shape[1]: return z
        val, ix = z.abs().topk(k, 1); return torch.zeros_like(z).scatter_(1, ix, torch.gather(z, 1, ix))
    def fvu(self, xp, y, k):
        z = self.tk(self.enc(xp, y), k)
        cross = (z * (y @ self.p.T) * (xp @ self.q.T)).sum(1)
        quad = torch.einsum("ti,ij,tj->t", z, (self.p @ self.p.T) * (self.q @ self.q.T), z)
        mnorm = (y ** 2).sum(1) * (xp ** 2).sum(1); return (mnorm - 2 * cross + quad).clamp_min(0), mnorm

def fit(h, y, m, k, seed, steps=6000):
    d = h.shape[1]; n = h.shape[0]; hp = h / (h ** 2).sum(1, keepdim=True).clamp_min(1e-9)
    sae = BSAE(d, m).to(DEV); opt = torch.optim.Adam(sae.parameters(), lr=2e-3); g = torch.Generator(device=DEV).manual_seed(seed)
    for _ in range(steps):
        bi = torch.randint(0, n, (2048,), generator=g, device=DEV)
        e, mn = sae.fvu(hp[bi], y[bi], k); (e.sum() / mn.sum()).backward(); opt.step(); opt.zero_grad()
    with torch.no_grad(): e, mn = sae.fvu(hp, y, k)
    return float(e.sum() / mn.sum())

# ---- block2 ----
from deep_model import DeepModel
cfg = json.load(open("runs_owt/block2-dense-seed0/config.json"))
model = DeepModel(cfg["vocab"], cfg["d_model"], cfg["n_head"], cfg["spec"], cfg["n_ctx"], d_hidden=512,
                  scale=cfg.get("scale", 0.5), norm=cfg["norm"], residual=cfg["residual"], attention=cfg["attention"])
model.load_state_dict(torch.load("runs_owt/block2-dense-seed0/model.pt", map_location="cpu", weights_only=True)); model.eval()
rms = torch.nn.RMSNorm(cfg["d_model"], elementwise_affine=False)
toks = np.fromfile("data_text/val.bin", dtype=np.uint16).astype(np.int64); T = cfg["n_ctx"]
seqs = torch.tensor(toks[:120 * T].reshape(120, T))
with torch.no_grad(): stream = model.residuals(seqs)
def b2(li, ii):
    mlp = model.layers[li]; h = rms(stream[ii].reshape(-1, cfg["d_model"])).detach()
    with torch.no_grad(): y = (mlp.D.weight @ ((mlp.L.weight @ h.T) * (mlp.R.weight @ h.T))).T.detach()
    return h.to(DEV), y.to(DEV)

print("CONTROL: sparse operator-dict vs dense low-rank (secant FVU; lower=better). 3 seeds.\n")
for name, (li, ii) in {"block2 MLP#0": (1, 0), "block2 MLP#1": (3, 2)}.items():
    h, y = b2(li, ii)
    print(f"=== {name} (d={h.shape[1]}) ===")
    print(f"  {'dense rank-R (all active)':32s}", end="")
    for R in (4, 8, 16, 32, 64):
        v = np.mean([fit(h, y, R, R, s) for s in range(3)]); print(f" R{R}:{v:.3f}", end="")
    print()
    vs = np.mean([fit(h, y, 256, 16, s) for s in range(3)])
    print(f"  SPARSE m=256 k=16  FVU {vs:.3f}   (compare to dense curve above at k=16 and beyond)\n", flush=True)

# ---- 500M layer 6 ----
from huggingface_hub import hf_hub_download
import jacclust.tt_model as TT
repo = "Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd"
c5 = json.load(open(hf_hub_download(repo, "config.json"))); c5.pop("step", None)
m5 = TT.GPT(TT.GPTConfig(**c5)).to(DEV).eval()
m5.load_state_dict(torch.load(hf_hub_download(repo, "pytorch_model.bin"), map_location=DEV, weights_only=True))
from transformers import AutoTokenizer; tk = AutoTokenizer.from_pretrained("gpt2")
import datasets
ds = datasets.load_dataset("NeelNanda/pile-10k", split="train", streaming=True); docs = []
for dch in ds:
    t = tk(dch["text"])["input_ids"]
    if len(t) >= 129: docs.append(t[:129])
    if len(docs) >= 96: break
idx = torch.tensor(docs, device=DEV)[:, :-1].contiguous()
blk = m5.transformer.h[6]; st = {}
hk = blk.register_forward_hook(lambda mm, i, o: st.__setitem__("r", i[0].detach()))
with torch.no_grad(): m5(idx, idx)
hk.remove(); h5 = F.rms_norm(st["r"].reshape(-1, c5["n_embd"]), (c5["n_embd"],)).detach()
with torch.no_grad(): y5 = blk.mlp(h5).detach()
print(f"=== 500M layer-6 MLP (d={c5['n_embd']}) ===")
print(f"  {'dense rank-R (all active)':32s}", end="")
for R in (8, 16, 32, 64):
    v = np.mean([fit(h5, y5, R, R, s) for s in range(2)]); print(f" R{R}:{v:.3f}", end="")
print()
vs = np.mean([fit(h5, y5, 512, 32, s) for s in range(2)])
print(f"  SPARSE m=512 k=32  FVU {vs:.3f}\n")
print("VERDICT: sparsity real iff SPARSE FVU < dense at matched-or-higher active count.")
print("DONE")
