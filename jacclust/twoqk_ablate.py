"""tick 31 (autonomous): is SIGNED attention functionally load-bearing? Causal ablation of the negative pattern.

tick 30 showed the two-QK product pattern is ~49% subtractive on average (signed attention EXISTS). Does the
model USE it? Causal test, no labels:
  clamp_neg : pattern <- relu(pattern)  (remove anti-copy; keep positive mixing)   -> CE damage
  flip_neg  : pattern <- |pattern|      (turn anti-copy INTO copy; tests if the SIGN specifically matters)
  drop_ctrl : zero a RANDOM set of (mostly positive) entries carrying the SAME total |mass| as the negatives
              (matched-mass control: if clamp_neg >> drop_ctrl, the negative entries matter beyond their mass)
Global (all heads) for the controls; per-head clamp_neg to rank heads by reliance on signed attention.
Pattern construction is the one that matches model CE to 2e-4 (tick 27). 3 seeds for the random control.
"""
import sys, torch, json, numpy as np, torch.nn.functional as F
sys.path.insert(0, "/workspace/tensor_language")
from huggingface_hub import hf_hub_download
import jacclust.tt_model as TT

DEV = "cuda"; torch.set_default_dtype(torch.float32)
repo = "Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd"
cfg = json.load(open(hf_hub_download(repo, "config.json"))); cfg.pop("step", None)
m = TT.GPT(TT.GPTConfig(**cfg)).to(DEV).eval()
m.load_state_dict(torch.load(hf_hub_download(repo, "pytorch_model.bin"), map_location=DEV, weights_only=True))
from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained("gpt2")
import datasets
nh, hd = m.transformer.h[0].attn.n_head, m.transformer.h[0].attn.head_dim
nL = len(m.transformer.h)

def apply_rot(x, c, s):
    d = x.shape[-1] // 2; x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * c + x2 * s, -x1 * s + x2 * c], -1)

def forward_ce(idx, tgt, mode="clean", target="all", seed=0):
    x = m.transformer.wte(idx); x = F.rms_norm(x, (x.size(-1),)); x0 = x; v1 = None
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0; a = blk.attn
        h = F.rms_norm(x, (x.size(-1),)); B, T, C = h.shape
        q = a.c_q(h).view(B, T, nh, hd); k = a.c_k(h).view(B, T, nh, hd)
        q2 = a.c_q2(h).view(B, T, nh, hd); k2 = a.c_k2(h).view(B, T, nh, hd); v = a.c_v(h).view(B, T, nh, hd)
        if v1 is None: v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        cos, sin = a.rotary(q); cos = cos.float(); sin = sin.float()
        q = F.rms_norm(q, (hd,)); k = F.rms_norm(k, (hd,)); q2 = F.rms_norm(q2, (hd,)); k2 = F.rms_norm(k2, (hd,))
        q = apply_rot(q, cos, sin); k = apply_rot(k, cos, sin); q2 = apply_rot(q2, cos, sin); k2 = apply_rot(k2, cos, sin)
        sc = torch.einsum("bqhd,bkhd->bhqk", q, k) / hd; sc2 = torch.einsum("bqhd,bkhd->bhqk", q2, k2) / hd
        mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        pat = (sc * sc2).masked_fill(~mask, 0.0)          # (B, nh, T, T) signed
        if mode != "clean":
            hit = (target == "all") or (li == target[0])
            if hit:
                sl = slice(None) if target == "all" else target[1]
                p = pat[:, sl] if target == "all" else pat[:, sl:sl + 1]
                if mode == "clamp_neg":
                    p = p.clamp_min(0.0)
                elif mode == "flip_neg":
                    p = p.abs()
                elif mode == "drop_ctrl":
                    neg_mass = p[p < 0].abs().sum()
                    g = torch.Generator(device=DEV).manual_seed(seed + li * 97 + (0 if target == "all" else target[1]))
                    r = torch.rand(p.shape, generator=g, device=DEV).masked_fill(p <= 0, 2.0)  # only positive entries eligible
                    order = r.flatten().argsort()                    # random order over positive entries
                    cum = p.flatten().clamp_min(0)[order].cumsum(0)
                    ndrop = int((cum <= neg_mass).sum())
                    dropmask = torch.zeros(p.numel(), dtype=torch.bool, device=DEV); dropmask[order[:ndrop]] = True
                    p = p.flatten().masked_fill(dropmask, 0.0).view_as(p)
                if target == "all": pat = pat.clone(); pat[:, sl] = p
                else: pat = pat.clone(); pat[:, sl:sl + 1] = p
        y = torch.einsum("bhqk,bkhd->bqhd", pat, v).reshape(B, T, C); y = a.c_proj(y); x = x + y
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    x = F.rms_norm(x, (x.size(-1),)); logits = 30 * torch.tanh(m.lm_head(x) / 30)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1))

ds = datasets.load_dataset("NeelNanda/pile-10k", split="train", streaming=True); docs = []
for d in ds:
    t = tok(d["text"])["input_ids"]
    if len(t) >= 129: docs.append(t[:129])
    if len(docs) >= 16: break
toks = torch.tensor(docs, device=DEV); idx = toks[:, :-1].contiguous(); tgt = toks[:, 1:].contiguous()
with torch.no_grad():
    ce = float(forward_ce(idx, tgt, "clean"))
print(f"500M two-QK. clean CE {ce:.4f}\n=== GLOBAL (all heads) negative-pattern ablations ===")
with torch.no_grad():
    dcn = float(forward_ce(idx, tgt, "clamp_neg", "all")) - ce
    dfn = float(forward_ce(idx, tgt, "flip_neg", "all")) - ce
    dcs = np.array([float(forward_ce(idx, tgt, "drop_ctrl", "all", seed=s)) - ce for s in range(3)])
print(f"  clamp_neg (remove anti-copy)      CE damage +{dcn:.4f}")
print(f"  flip_neg  (anti-copy -> copy)      CE damage +{dfn:.4f}")
print(f"  drop_ctrl (matched pos-mass, ctrl) CE damage +{dcs.mean():.4f}±{dcs.std():.4f}")
print(f"  => signed attention load-bearing iff clamp_neg/flip_neg >> drop_ctrl\n")

print("=== per-head clamp_neg CE damage (reliance on signed attention) ===")
with torch.no_grad():
    dmg = np.array([[float(forward_ce(idx, tgt, "clamp_neg", (L, H))) - ce for H in range(nh)] for L in range(nL)])
flat = sorted([(dmg[L, H], L, H) for L in range(nL) for H in range(nh)], reverse=True)
print(f"  per-head damage: mean {dmg.mean():.4f}, max {dmg.max():.4f}, "
      f"heads with >0.01: {(dmg > 0.01).sum()}, >0.05: {(dmg > 0.05).sum()}")
print("  top-10 heads by signed-attn reliance:")
for d, L, H in flat[:10]:
    print(f"    L{L}H{H}: +{d:.4f}")
print(f"  per-LAYER mean clamp_neg damage: {np.round(dmg.mean(1), 4).tolist()}")
print("DONE")
