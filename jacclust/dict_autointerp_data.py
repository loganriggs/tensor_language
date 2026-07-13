"""tick 49 (autonomous): autointerp PILOT data for L0 operators — real features vs random-feature controls, blind.

Rigor: present N real operator features + M control (random rank-1) features' top-activating tokens BLIND to a
labeler; if it can't score real > control, coherence is illusory. This script produces the token lists (JSON).
"""
import sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/workspace/tensor_language")
from huggingface_hub import hf_hub_download
import jacclust.tt_model as TT
torch.set_default_dtype(torch.float32); DEV = "cuda"

class BSAE(torch.nn.Module):
    def __init__(s, d, m):
        super().__init__(); s.q = torch.nn.Parameter(torch.randn(m, d) / d ** 0.5); s.p = torch.nn.Parameter(torch.randn(m, d) / d ** 0.5)
    def enc(s, xp, y): return (xp @ s.q.T) * (y @ s.p.T)
    def tk(s, z, k):
        val, ix = z.abs().topk(k, 1); return torch.zeros_like(z).scatter_(1, ix, torch.gather(z, 1, ix))
    def err(s, xp, y, k):
        z = s.tk(s.enc(xp, y), k); cross = (z * (y @ s.p.T) * (xp @ s.q.T)).sum(1)
        quad = torch.einsum("ti,ij,tj->t", z, (s.p @ s.p.T) * (s.q @ s.q.T), z)
        mnorm = (y ** 2).sum(1) * (xp ** 2).sum(1); return (mnorm - 2 * cross + quad).clamp_min(0), mnorm, z

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
    if len(docs) >= 160: break
toks = torch.tensor(docs, device=DEV); idx = toks[:, :-1].contiguous(); d = cfg["n_embd"]; T = idx.shape[1]
import sys as _s; L = int(_s.argv[1]) if len(_s.argv)>1 else 0
blk = m.transformer.h[L]; st = {}
hk = blk.register_forward_hook(lambda mm, i, o: st.__setitem__("r", i[0].detach()))
with torch.no_grad(): m(idx, idx)
hk.remove(); h = F.rms_norm(st["r"].reshape(-1, d), (d,)).detach()
with torch.no_grad(): y = blk.mlp(h).detach()
hp = h / (h ** 2).sum(1, keepdim=True).clamp_min(1e-9)

sae = BSAE(d, 512).to(DEV); opt = torch.optim.Adam(sae.parameters(), lr=2e-3); g = torch.Generator(device=DEV).manual_seed(0)
for _ in range(5000):
    bi = torch.randint(0, hp.shape[0], (2048,), generator=g, device=DEV)
    e, mn, _ = sae.err(hp[bi], y[bi], 32); (e.sum() / mn.sum()).backward(); opt.step(); opt.zero_grad()
with torch.no_grad(): z = sae.tk(sae.enc(hp, y), 32).abs()      # (n,512) activation strength

def top_tokens(zi, n_top=20):
    top = zi.topk(n_top).indices.cpu().numpy(); out = []
    for t in top:
        seq, pos = t // T, t % T
        ctx = [tok.decode([int(idx[seq, p])]) for p in range(max(0, pos - 4), pos + 1)]
        out.append("…" + "".join(ctx).replace("\n", "\\n"))
    return out

# real: top-8 by usage; control: 4 random rank-1 features
freq = (z > 1e-6).float().mean(0); real_ids = freq.topk(8).indices.cpu().numpy().tolist()
gg = torch.Generator(device=DEV).manual_seed(7)
qc = F.normalize(torch.randn(4, d, generator=gg, device=DEV), 1); pc = F.normalize(torch.randn(4, d, generator=gg, device=DEV), 1)
zc = ((hp @ qc.T) * (y @ pc.T)).abs()

items = []
for i in real_ids: items.append({"kind": "real", "src": int(i), "top_tokens": top_tokens(z[:, i])})
for j in range(4): items.append({"kind": "control", "src": f"rand{j}", "top_tokens": top_tokens(zc[:, j])})
rng = np.random.RandomState(0); order = rng.permutation(len(items))          # BLIND shuffle
blind = [{"blind_id": int(k), **items[order[k]]} for k in range(len(items))]
json.dump(blind, open(f"jacclust/autointerp_L{L}.json", "w"), indent=1)
print(f"L{L} SAE trained. wrote {len(blind)} features (8 real + 4 control) to autointerp_L{L}.json (blind order).")
print("KEY (kind by blind_id):", {b["blind_id"]: b["kind"] for b in blind})
for b in blind[:2]: print(f"  blind {b['blind_id']} ({b['kind']}): {b['top_tokens'][:6]}")
print("DONE")
