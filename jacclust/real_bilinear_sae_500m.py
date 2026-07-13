"""tick 44 (autonomous): bilinear-secant SAE on the 500M model's bilinear MLPs (d=1152) — scale + outliers.

Continues tick 43 (block2, d=128, non-null positive, no outliers) to the 500M gpt2-bilinear-sqrd-attn-18l.
Demonstrates large-d feasibility (expanded loss never forms the 1152x1152 secant) and tests for outliers on a
bigger model. x = post-norm MLP input h; y = MLP output D(Lh⊙Rh)+bias; secant M = y h^+. Metrics: secant recon
FVU (vs random-atom control), functional FVU (M̂h vs y), outlier diagnostics. Layers 6 & 12. 3 seeds.
"""
import sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/workspace/tensor_language")
from huggingface_hub import hf_hub_download
import jacclust.tt_model as TT
torch.set_default_dtype(torch.float32); DEV = "cuda"

class BilinearSAE(torch.nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.q = torch.nn.Parameter(torch.randn(m, d) / d ** 0.5)
        self.p = torch.nn.Parameter(torch.randn(m, d) / d ** 0.5)
    def encode(self, xp, y): return (xp @ self.q.T) * (y @ self.p.T)
    def topk(self, z, k):
        val, ix = z.abs().topk(k, 1); return torch.zeros_like(z).scatter_(1, ix, torch.gather(z, 1, ix))
    def loss(self, xp, y, k):
        z = self.topk(self.encode(xp, y), k)
        cross = (z * (y @ self.p.T) * (xp @ self.q.T)).sum(1)
        quad = torch.einsum("ti,ij,tj->t", z, (self.p @ self.p.T) * (self.q @ self.q.T), z)
        mnorm = (y ** 2).sum(1) * (xp ** 2).sum(1)
        return (mnorm - 2 * cross + quad).clamp_min(0).mean(), z

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
    if len(docs) >= 96: break
idx = torch.tensor(docs, device=DEV)[:, :-1].contiguous()
d = cfg["n_embd"]; M_DICT, KSP, STEPS = 512, 32, 6000

def get_hy(layer):
    blk = m.transformer.h[layer]
    st = {}; hk = blk.register_forward_hook(lambda mm, i, o: st.__setitem__("r", i[0].detach()))
    with torch.no_grad(): m(idx, idx)
    hk.remove()
    x = st["r"].reshape(-1, d)
    h = F.rms_norm(x, (d,)).detach()
    with torch.no_grad(): y = blk.mlp(h).detach()
    return h, y

def run(h, y, seed, train=True):
    n = h.shape[0]; hp = h / (h ** 2).sum(1, keepdim=True).clamp_min(1e-9)
    sae = BilinearSAE(d, M_DICT).to(DEV)
    if train:
        opt = torch.optim.Adam(sae.parameters(), lr=2e-3); g = torch.Generator(device=DEV).manual_seed(seed)
        for _ in range(STEPS):
            bi = torch.randint(0, n, (2048,), generator=g, device=DEV)
            loss, _ = sae.loss(hp[bi], y[bi], KSP); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        z = sae.topk(sae.encode(hp, y), KSP)
        cross = (z * (y @ sae.p.T) * (hp @ sae.q.T)).sum(1)
        quad = torch.einsum("ti,ij,tj->t", z, (sae.p @ sae.p.T) * (sae.q @ sae.q.T), z)
        mnorm = (y ** 2).sum(1) * (hp ** 2).sum(1); err = (mnorm - 2 * cross + quad).clamp_min(0)
        fvu_s = float(err.sum() / mnorm.sum())
        Mhat_h = torch.einsum("ti,id->td", z * (h @ sae.q.T), sae.p)
        fvu_f = float(((y - Mhat_h) ** 2).sum() / (y ** 2).sum())
        per = (err / mnorm.clamp_min(1e-12)).cpu().numpy()
    return fvu_s, fvu_f, per, mnorm.cpu().numpy()

print(f"500M bilinear-secant SAE. d={d}, dict={M_DICT}, k={KSP}, {idx.numel()} tokens, 3 seeds.\n")
for layer in (6, 12):
    h, y = get_hy(layer); n = h.shape[0]
    fs = np.array([run(h, y, s)[0] for s in range(3)])
    ff = np.array([run(h, y, s)[1] for s in range(3)])
    fr = np.array([run(h, y, s, train=False)[0] for s in range(3)])
    _, _, per, mm = run(h, y, 0)
    hn, yn = h.cpu().numpy(), y.cpu().numpy()
    kurt_y = ((yn - yn.mean(0)) ** 4).mean(0) / (((yn - yn.mean(0)) ** 2).mean(0) ** 2 + 1e-12)
    outdim_y = np.abs(yn).max(0) / (np.median(np.abs(yn), 0) + 1e-9)
    order = np.argsort(-mm); top1 = mm[order[:max(1, n // 100)]].sum() / mm.sum()
    corr = np.corrcoef(per, mm)[0, 1]
    print(f"=== layer {layer} MLP (n={n}) ===")
    print(f"  secant recon FVU  trained {fs.mean():.3f}±{fs.std():.3f}   random-atoms {fr.mean():.3f}")
    print(f"  functional FVU (M̂h vs y): {ff.mean():.3f}")
    print(f"  OUTLIERS: top-1% tokens hold {top1*100:.1f}% of secant mass | y max/median dim-ratio {outdim_y.max():.0f}"
          f" | #y-dims kurtosis>20: {(kurt_y>20).sum()}/{d} | corr(per-tok FVU, ||M||^2) {corr:+.2f}\n", flush=True)
print("DONE")
