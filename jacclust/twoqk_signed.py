"""tick 30 (autonomous): WHY two QK matrices? The mechanistic capability = SIGNED attention.

The two-QK squared pattern is p_qk = (q1.k1/D)(q2.k2/D), used RAW (masked, no softmax, no abs, no
row-norm). Unlike the single-QK squared pattern (q.k)^2 >= 0 (pure positive mixing), the PRODUCT of two
scores can be NEGATIVE => the head can SUBTRACT a value (anti-copy / suppression). Hypothesis: the second
query matrix buys signed attention, and that is why q1,q2 are complementary (they define a signed bilinear
form whose sign varies over keys). Quantitative, no labels, no over-reading.

Measure per head, on real text:
  frac_neg_entries : fraction of valid (q,k) pattern entries that are < 0
  frac_neg_mass    : sum|neg weights| / sum|all weights|   (how much of the attention MASS is subtractive)
  a squared single-QK head would give 0 on both — so any positive value is capability the 2nd matrix adds.
Also: does negative-mass predict which heads gained from two-QK clustering (ticks 27-28)? Cross-check
against the measured gains (L8H3,L6H3,L0H2 positive-gain; L9H3,L11H5,etc from the screen).
Verify the pattern reconstruction against the model's own attention (custom fwd already matches CE 3.83).
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

ds = datasets.load_dataset("NeelNanda/pile-10k", split="train", streaming=True); docs = []
for d in ds:
    t = tok(d["text"])["input_ids"]
    if len(t) >= 129: docs.append(t[:129])
    if len(docs) >= 16: break
toks = torch.tensor(docs, device=DEV); idx = toks[:, :-1].contiguous()

# recompute residual inputs to every block (the model's own forward, hooked)
resid = {}
hooks = [m.transformer.h[L].register_forward_hook((lambda L: lambda mm, i, o: resid.__setitem__(L, i[0].detach()))(L)) for L in range(nL)]
with torch.no_grad(): m(idx, toks[:, 1:].contiguous())
for h in hooks: h.remove()

@torch.no_grad()
def head_pattern(L, H):
    """Reconstruct the realized (signed, masked, unnormalized) attention pattern for one head."""
    a = m.transformer.h[L].attn; Xn = F.rms_norm(resid[L], (resid[L].shape[-1],)); B, T, _ = Xn.shape
    def hq(W): return F.rms_norm((Xn @ W.T).view(B, T, nh, hd)[:, :, H, :], (hd,))
    cos, sin = a.rotary(torch.zeros(1, T, nh, hd, device=DEV)); cos = cos[0, :, 0].float(); sin = sin[0, :, 0].float()
    q1 = apply_rot(hq(a.c_q.weight), cos, sin); k1 = apply_rot(hq(a.c_k.weight), cos, sin)
    q2 = apply_rot(hq(a.c_q2.weight), cos, sin); k2 = apply_rot(hq(a.c_k2.weight), cos, sin)
    s1 = torch.einsum("bqd,bkd->bqk", q1, k1) / hd
    s2 = torch.einsum("bqd,bkd->bqk", q2, k2) / hd
    pat = s1 * s2
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    return pat, s1, s2, mask

# verify: reconstructed pattern @ v reproduces the head's z (sanity)
def stats(L, H):
    pat, s1, s2, mask = head_pattern(L, H)
    m2 = mask.clone(); m2.fill_diagonal_(False)          # off-diagonal only (self is trivially positive-ish)
    sel = pat[:, m2]                                       # valid off-diag entries, flattened over batch
    fne = float((sel < 0).float().mean())
    absmass = sel.abs().sum().clamp_min(1e-9)
    fnm = float(sel[sel < 0].abs().sum() / absmass)
    # conjunction: does the product concentrate on fewer keys than |s1| or |s2| alone? participation ratio
    def pr(w):  # per-query participation ratio of |w| over causal keys, averaged
        wa = w.abs().masked_fill(~mask, 0.0); num = wa.sum(-1) ** 2; den = (wa ** 2).sum(-1).clamp_min(1e-12)
        n = mask.sum(-1).clamp_min(1).float()
        return float(((num / den) / n).mean())            # normalized PR in (0,1]; lower => more peaked
    return fne, fnm, pr(pat), pr(s1.abs()), pr(s2.abs())

print(f"500M two-QK signed-attention analysis. (squared single-QK head would give frac_neg=0.)\n")
print(f"{'head':7s} {'frac_neg_entries':>16s} {'frac_neg_mass':>14s} {'PR(prod)':>9s} {'PR(s1)':>8s} {'PR(s2)':>8s}")
rows = []
allh = [(L, H) for L in range(nL) for H in range(nh)]
for (L, H) in allh:
    fne, fnm, prp, pr1, pr2 = stats(L, H)
    rows.append((L, H, fne, fnm, prp, pr1, pr2))
R = np.array([[r[2], r[3], r[4], r[5], r[6]] for r in rows])
print(f"  ALL {len(allh)} heads: frac_neg_entries {R[:,0].mean():.3f}±{R[:,0].std():.3f} "
      f"[{R[:,0].min():.3f},{R[:,0].max():.3f}]")
print(f"                frac_neg_mass    {R[:,1].mean():.3f}±{R[:,1].std():.3f} "
      f"[{R[:,1].min():.3f},{R[:,1].max():.3f}]")
print(f"                PR: product {R[:,2].mean():.3f}  vs s1 {R[:,3].mean():.3f}  s2 {R[:,4].mean():.3f} "
      f"(lower=more peaked; product<single => conjunctive sharpening)\n")
# causal heads from ticks 27-28
tick = {(0,2):"gain+", (6,3):"gain+", (8,3):"gain+", (3,3):"gain0", (9,3):"scr", (11,5):"scr", (1,1):"scr", (13,3):"scr"}
print("  causal/screen heads (ticks 27-28):")
for r in rows:
    if (r[0], r[1]) in tick:
        print(f"    L{r[0]}H{r[1]} [{tick[(r[0],r[1])]:5s}] frac_neg_mass {r[3]:.3f}  frac_neg_entries {r[2]:.3f}  PR prod {r[4]:.3f} / s1 {r[5]:.3f} / s2 {r[6]:.3f}")
# does product sharpen vs single?
sharper = np.mean(R[:,2] < np.minimum(R[:,3], R[:,4]))
print(f"\n  fraction of heads where PR(product) < min(PR(s1),PR(s2)) [conjunctive sharpening]: {sharper:.2f}")
print("DONE")
