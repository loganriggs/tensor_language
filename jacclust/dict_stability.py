"""tick 48 (autonomous): is the recovered operator dictionary STABLE (canonical) or a seed-dependent basis?

Precursor to any interpretation: if two independently-trained SAEs find the SAME operators, the dictionary is
real; if not, it's an arbitrary compression basis. Rank-1 atoms p_i q_i^T; cosine between atoms = (p·p')(q·q').
Dictionary MMCS: mean over used atoms in A of max cosine to any atom in B. Control: MMCS of two RANDOM dicts
(chance stability). Layers: L0 (highest sparsity margin), L8 (sparse, needs-y), L6 (low-rank). 3 seeds each.
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
        val, ix = z.abs().topk(k, 1); return torch.zeros_like(z).scatter_(1, ix, torch.gather(z, 1, ix))
    def err(self, xp, y, k):
        z = self.tk(self.enc(xp, y), k)
        cross = (z * (y @ self.p.T) * (xp @ self.q.T)).sum(1)
        quad = torch.einsum("ti,ij,tj->t", z, (self.p @ self.p.T) * (self.q @ self.q.T), z)
        mnorm = (y ** 2).sum(1) * (xp ** 2).sum(1); return (mnorm - 2 * cross + quad).clamp_min(0), mnorm, z

def fit(h, y, seed, m=512, k=32, steps=5000):
    n, d = h.shape; hp = h / (h ** 2).sum(1, keepdim=True).clamp_min(1e-9)
    sae = BSAE(d, m).to(DEV); opt = torch.optim.Adam(sae.parameters(), lr=2e-3); g = torch.Generator(device=DEV).manual_seed(seed)
    for _ in range(steps):
        bi = torch.randint(0, n, (2048,), generator=g, device=DEV)
        e, mn, _ = sae.err(hp[bi], y[bi], k); (e.sum() / mn.sum()).backward(); opt.step(); opt.zero_grad()
    with torch.no_grad():
        _, _, z = sae.err(hp, y, k)
        used = ((z.abs() > 1e-6).float().mean(0) > 0.001).cpu().numpy()   # atoms active on >0.1% tokens
        P = F.normalize(sae.p.detach(), dim=1); Q = F.normalize(sae.q.detach(), dim=1)
    return P.cpu(), Q.cpu(), used

def dict_mmcs(Pa, Qa, ua, Pb, Qb, ub):
    """mean over used atoms in A of max |(p·p')(q·q')| to used atoms in B (rank-1 atom cosine)."""
    sim = (Pa @ Pb.T).abs() * (Qa @ Qb.T).abs()          # (mA, mB)
    sim = sim[ua][:, ub]
    return float(sim.max(1).values.mean()) if sim.numel() else float("nan")

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

# random-dict chance MMCS
def rand_mmcs(mm=512):
    g = torch.Generator().manual_seed(0)
    Pa = F.normalize(torch.randn(mm, d, generator=g), 1); Qa = F.normalize(torch.randn(mm, d, generator=g), 1)
    Pb = F.normalize(torch.randn(mm, d, generator=g), 1); Qb = F.normalize(torch.randn(mm, d, generator=g), 1)
    return dict_mmcs(Pa, Qa, np.ones(mm, bool), Pb, Qb, np.ones(mm, bool))
chance = rand_mmcs()

print(f"500M operator-dictionary STABILITY across seeds (rank-1 atom MMCS; chance {chance:.3f}).\n")
print(f"{'layer':7s} {'#used atoms':>12s} {'dict MMCS (3-seed pairs)':>26s}")
for L, tag in [(0, "sparse hi-margin"), (8, "sparse needs-y"), (6, "low-rank")]:
    blk = m.transformer.h[L]; st = {}
    hk = blk.register_forward_hook(lambda mm, i, o: st.__setitem__("r", i[0].detach()))
    with torch.no_grad(): m(idx, idx)
    hk.remove(); h = F.rms_norm(st["r"].reshape(-1, d), (d,)).detach()
    with torch.no_grad(): y = blk.mlp(h).detach()
    dicts = [fit(h, y, s) for s in range(3)]
    nused = np.mean([dd[2].sum() for dd in dicts])
    pairs = []
    for i in range(3):
        for j in range(3):
            if i < j:
                pairs.append(dict_mmcs(*dicts[i], *dicts[j]))
    pairs = np.array(pairs)
    print(f"  L{L:<5d} {nused:12.0f}   {pairs.mean():.3f}±{pairs.std():.3f}   [{tag}]", flush=True)
print(f"\n  stable/canonical iff MMCS >> chance ({chance:.3f}). low MMCS => arbitrary basis, not real operators.")
print("DONE")
