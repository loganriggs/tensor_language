"""tick 27 (Logan): GENUINE two-QK bilinear attention on a REAL checkpoint.

Model gpt2-bilinear-sqrd-attn-18l-9h-1152embd (500M) has c_q,c_k,c_q2,c_k2 (config bilinear_attn=True,
squared_attn=True): pattern_qk = (q1.k1)(q2.k2)/D^2, UNNORMALIZED (verified: model runs CE 3.79 with the
unnormalized forward; do NOT add row-normalization here — that was only the single-QK variant's bug).

The query's key-selectivity is now a PRODUCT of two bilinear forms => characterized by the PAIR of query
covectors (q1til, q2til) and their tensor product q1til (x) q2til. Cosine of vec(q1n (x) q2n) between two
queries = cos(q1,q1')*cos(q2,q2') — the exact two-matrix product kernel. Question: does clustering by BOTH
matrices beat q1-alone or q2-alone, causally?

Same swap-within causal metric as ticks 23-26. Intervention overrides BOTH head queries (q1 and q2) with
the cluster representative; only the CLUSTERING feature varies. 5 seeds. Controls: residual x, random.
First screen heads by global-mean-query damage, run the full comparison on the top causal heads.
"""
import sys, torch, json, numpy as np, torch.nn.functional as F
from sklearn.cluster import KMeans
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

K = 8

def apply_rot(x, c, s):
    d = x.shape[-1] // 2; x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * c + x2 * s, -x1 * s + x2 * c], -1)

def forward_ce(idx, tgt, LAYER=-1, HEAD=-1, qov1=None, qov2=None):
    x = m.transformer.wte(idx); x = F.rms_norm(x, (x.size(-1),)); x0 = x; v1 = None
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0; a = blk.attn
        h = F.rms_norm(x, (x.size(-1),)); B, T, C = h.shape; nh, hd = a.n_head, a.head_dim
        q = a.c_q(h).view(B, T, nh, hd); k = a.c_k(h).view(B, T, nh, hd)
        q2 = a.c_q2(h).view(B, T, nh, hd); k2 = a.c_k2(h).view(B, T, nh, hd)
        v = a.c_v(h).view(B, T, nh, hd)
        if v1 is None: v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        cos, sin = a.rotary(q); cos = cos.float(); sin = sin.float()
        q = F.rms_norm(q, (hd,)); k = F.rms_norm(k, (hd,))
        q2 = F.rms_norm(q2, (hd,)); k2 = F.rms_norm(k2, (hd,))
        if li == LAYER:
            if qov1 is not None: q = q.clone(); q[:, :, HEAD, :] = qov1
            if qov2 is not None: q2 = q2.clone(); q2[:, :, HEAD, :] = qov2
        q = apply_rot(q, cos, sin); k = apply_rot(k, cos, sin)
        q2 = apply_rot(q2, cos, sin); k2 = apply_rot(k2, cos, sin)
        sc = torch.einsum("bqhd,bkhd->bhqk", q, k) / hd
        sc2 = torch.einsum("bqhd,bkhd->bhqk", q2, k2) / hd
        mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        pat = (sc * sc2).masked_fill(~mask, 0.0)            # UNNORMALIZED product (matches checkpoint)
        y = torch.einsum("bhqk,bkhd->bqhd", pat, v).reshape(B, T, C); y = a.c_proj(y); x = x + y
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    x = F.rms_norm(x, (x.size(-1),)); logits = 30 * torch.tanh(m.lm_head(x) / 30)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1))

# ---- data ----
ds = datasets.load_dataset("NeelNanda/pile-10k", split="train", streaming=True); docs = []
for d in ds:
    t = tok(d["text"])["input_ids"]
    if len(t) >= 66: docs.append(t[:66])
    if len(docs) >= 24: break
toks = torch.tensor(docs, device=DEV); idx = toks[:, :-1].contiguous(); tgt = toks[:, 1:].contiguous()
with torch.no_grad():
    ce_ref = float(m(idx, tgt))
    ce_chk = float(forward_ce(idx, tgt))
print(f"500M two-QK bilinear. model CE {ce_ref:.4f}  custom CE {ce_chk:.4f}  diff {abs(ce_ref-ce_chk):.2e}")
assert abs(ce_ref - ce_chk) < 1e-3, "custom forward mismatch"
B, Tm = idx.shape
nh, hd = m.transformer.h[0].attn.n_head, m.transformer.h[0].attn.head_dim
valid = [(b, q) for b in range(B) for q in range(2, Tm)]

def readouts(LAYER, HEAD):
    a = m.transformer.h[LAYER].attn; dm = a.n_embd
    st = {}; hk = m.transformer.h[LAYER].register_forward_hook(lambda mm, i, o: st.__setitem__("r", i[0].detach()))
    with torch.no_grad(): m(idx, tgt)
    hk.remove(); Xn = F.rms_norm(st["r"], (dm,))
    q1 = F.rms_norm((Xn @ a.c_q.weight.detach().T).view(B, Tm, nh, hd)[:, :, HEAD, :], (hd,)).detach()
    q2 = F.rms_norm((Xn @ a.c_q2.weight.detach().T).view(B, Tm, nh, hd)[:, :, HEAD, :], (hd,)).detach()
    return Xn.detach(), q1, q2

# ---- screen heads by global-mean-query damage ----
print("\n=== screen: global-mean-query damage (both q1,q2 -> global mean), pick top causal heads ===")
cand = [(L, H) for L in [3, 6, 8, 9, 11, 13, 15, 17] for H in range(0, 9, 3)]
scores = []
for (L, H) in cand:
    _, q1, q2 = readouts(L, H)
    g1 = q1.mean((0, 1)); g2 = q2.mean((0, 1))
    with torch.no_grad():
        dmg = float(forward_ce(idx, tgt, L, H, g1[None, None].expand(B, Tm, hd), g2[None, None].expand(B, Tm, hd))) - ce_ref
    scores.append(((L, H), dmg))
scores.sort(key=lambda z: -z[1])
for (LH, dmg) in scores[:8]:
    print(f"  L{LH[0]}H{LH[1]}: global-query damage {dmg:+.4f}")
top = [lh for lh, _ in scores[:3]]

# ---- full comparison on top heads ----
def cmeans(feat, L2):
    return [torch.stack([feat[b, q] for (b, q) in valid if L2[b, q] == c]).mean(0) if (L2 == c).any()
            else feat.mean((0, 1)) for c in range(K)]

def override(feat, L2, mns):
    ov = feat.clone()
    for (b, q) in valid:
        c = L2[b, q]
        if c >= 0: ov[b, q] = mns[c]
    return ov

def labels(mat, seed):
    l = KMeans(K, n_init=6, random_state=seed).fit_predict(mat); L2 = np.full((B, Tm), -1)
    for i, (b, q) in enumerate(valid): L2[b, q] = l[i]
    return L2

def swap_within(L, H, q1, q2, L2):
    m1, m2 = cmeans(q1, L2), cmeans(q2, L2)
    cw = float(forward_ce(idx, tgt, L, H, override(q1, L2, m1), override(q2, L2, m2)))
    s1 = [m1[(c + 1) % K] for c in range(K)]; s2 = [m2[(c + 1) % K] for c in range(K)]
    cs = float(forward_ce(idx, tgt, L, H, override(q1, L2, s1), override(q2, L2, s2)))
    return cs - cw

print("\n=== two-QK query clustering (override BOTH q1,q2; vary only the clustering feature), 5 seeds ===")
for (L, H) in top:
    Xn, q1, q2 = readouts(L, H)
    q1n = F.normalize(torch.stack([q1[b, q] for (b, q) in valid]), dim=1)
    q2n = F.normalize(torch.stack([q2[b, q] for (b, q) in valid]), dim=1)
    feats = {
        "q1 only": q1n.cpu().numpy(),
        "q2 only": q2n.cpu().numpy(),
        "both concat [q1;q2]": torch.cat([q1n, q2n], 1).cpu().numpy(),
        "both tensor q1(x)q2 (product kernel)": F.normalize(torch.einsum("ni,nj->nij", q1n, q2n).reshape(len(valid), -1), dim=1).cpu().numpy(),
        "residual x": F.normalize(torch.stack([Xn[b, q] for (b, q) in valid]), dim=1).cpu().numpy(),
    }
    print(f"\n  --- L{L}H{H} ---")
    for name, mat in feats.items():
        vals = np.array([swap_within(L, H, q1, q2, labels(mat, s)) for s in range(5)])
        print(f"    {name:38s} {vals.mean():+.4f} +- {vals.std():.4f}", flush=True)
    rvals = []
    for s in range(5):
        rl = np.random.RandomState(s).randint(0, K, len(valid)); L2 = np.full((B, Tm), -1)
        for i, (b, q) in enumerate(valid): L2[b, q] = rl[i]
        rvals.append(swap_within(L, H, q1, q2, L2))
    rvals = np.array(rvals)
    print(f"    {'random':38s} {rvals.mean():+.4f} +- {rvals.std():.4f}")
    # redundancy: are q1 and q2 selectivities correlated?
    cc = float((q1n * q2n).sum(1).abs().mean())
    print(f"    [mean |cos(q1,q2)| per token = {cc:.3f}  (high => the two matrices are redundant)]")
print("DONE")
