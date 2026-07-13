"""tick 26 (Logan Q): extend attention clustering to (A) the squared-attention TENSOR kernel
(the "two q,k matrices" structure the square already gives), (B) the OV / value pathway, and
(C) the full-head contraction's prediction that OV is redundant WITHIN a head.

Model: gpt2-bilinear-sqrd-attn-12l-6h (single-QK SQUARED attention). L6H0 = the causally-important head.

Squared attention weights key k from query q by (score_qk)^2, score = qtil_q . ktil_k (post QK-norm+rotary).
So query q's key-selectivity is a QUADRATIC form on keys: the rank-1 PSD matrix u_q u_q^T, u_q = the
query covector. Two queries do the SAME thing iff u_q u_q^T ~ u_q' u_q'^T, i.e. |cos(u_q,u_q')| ~ 1
(cos of the rank-1 forms == cos^2 of the vectors). This is the tensor / "two matrices" object for the
model we actually have. Test: does clustering the rank-1 form (== |cos|, sign-collapsed per the degree-2
homogeneity result) beat plain cos causally?

OV: the value token k contributes is v_k (head value); after the weighted sum, c_proj writes it. The
"what is copied" circuit is W_OV = W_O W_V. We have NEVER tested it. Cluster tokens by their value v_k and
causally swap the value written (within-cluster mean vs another cluster's) -> does the value pathway carry
causal cluster structure, separate from the query (WHERE) pathway?

Metric (identical to ticks 23/25): replace a token's readout by its cluster mean. within = own cluster
mean; swapped = next cluster's mean. swap-within = CE(swapped)-CE(within). Higher => clusters causally
distinct. Controls: raw residual-x clusters, random assignment. 5 kmeans seeds, mean+-sd.
"""
import sys, torch, json, numpy as np, torch.nn.functional as F
from sklearn.cluster import KMeans
sys.path.insert(0, "/workspace/tensor_language")
from huggingface_hub import hf_hub_download
import jacclust.tt_model as TT

DEV = "cuda"; torch.set_default_dtype(torch.float32)
repo = "Elriggs/gpt2-bilinear-sqrd-attn-12l-6h-768embd"
cfg = json.load(open(hf_hub_download(repo, "config.json"))); cfg.pop("step", None)
m = TT.GPT(TT.GPTConfig(**cfg)).to(DEV).eval()
m.load_state_dict(torch.load(hf_hub_download(repo, "pytorch_model.bin"), map_location=DEV, weights_only=True))
from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained("gpt2")
import datasets

LAYER, HEAD, K = 6, 0, 8

def apply_rot(x, c, s):
    d = x.shape[-1] // 2; x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * c + x2 * s, -x1 * s + x2 * c], -1)

def forward_ce(idx, tgt, qov=None, vov=None):
    """qov: (B,T,hd) override head query; vov: (B,T,hd) override head value (the OV pathway)."""
    x = m.transformer.wte(idx); x = F.rms_norm(x, (x.size(-1),)); x0 = x; v1 = None
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0; a = blk.attn
        h = F.rms_norm(x, (x.size(-1),)); B, T, C = h.shape; nh, hd = a.n_head, a.head_dim
        q = a.c_q(h).view(B, T, nh, hd); k = a.c_k(h).view(B, T, nh, hd); v = a.c_v(h).view(B, T, nh, hd)
        if v1 is None: v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        if li == LAYER and vov is not None:
            v = v.clone(); v[:, :, HEAD, :] = vov
        cos, sin = a.rotary(q); cos = cos.float(); sin = sin.float()
        q = F.rms_norm(q, (hd,)); k = F.rms_norm(k, (hd,))
        if li == LAYER and qov is not None:
            q = q.clone(); q[:, :, HEAD, :] = qov
        q = apply_rot(q, cos, sin); k = apply_rot(k, cos, sin)
        sc = torch.einsum("bqhd,bkhd->bhqk", q, k) / hd
        mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        pat = sc.square().masked_fill(~mask, 0.0); pat = pat / pat.sum(-1, keepdim=True).clamp_min(1e-9)
        y = torch.einsum("bhqk,bkhd->bqhd", pat, v).reshape(B, T, C); y = a.c_proj(y); x = x + y
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    x = F.rms_norm(x, (x.size(-1),)); logits = 30 * torch.tanh(m.lm_head(x) / 30)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1))

# ---- data ----
ds = datasets.load_dataset("NeelNanda/pile-10k", split="train", streaming=True); docs = []
for d in ds:
    t = tok(d["text"])["input_ids"]
    if len(t) >= 66: docs.append(t[:66])
    if len(docs) >= 32: break
toks = torch.tensor(docs, device=DEV); idx = toks[:, :-1].contiguous(); tgt = toks[:, 1:].contiguous()
with torch.no_grad():
    ce_ref = float(m(idx, tgt))
    assert abs(ce_ref - float(forward_ce(idx, tgt))) < 1e-3, "custom forward mismatch"

# ---- readouts at LAYER ----
a = m.transformer.h[LAYER].attn; nh, hd = a.n_head, a.head_dim
Wq, Wk, Wv = a.c_q.weight.detach(), a.c_k.weight.detach(), a.c_v.weight.detach()
st = {}; hk = m.transformer.h[LAYER].register_forward_hook(lambda mm, i, o: st.__setitem__("r", i[0].detach()))
with torch.no_grad(): m(idx, tgt)
hk.remove(); resid = st["r"]; B, Tm, dm = resid.shape
Xn = F.rms_norm(resid, (dm,)).detach()
qtil = F.rms_norm((Xn @ Wq.T).view(B, Tm, nh, hd)[:, :, HEAD, :], (hd,)).detach()  # head query (WHERE)
# value pathway: reproduce the mixed head value actually used at LAYER (needs v1 from block 0)
st2 = {}
def cap_v(mm, i, o):
    h_ = F.rms_norm(i[0].detach(), (dm,)); st2["v"] = (h_ @ Wv.T).view(-1, Tm, nh, hd)[:, :, HEAD, :]
hk2 = m.transformer.h[LAYER].register_forward_hook(cap_v)
with torch.no_grad(): m(idx, tgt)
hk2.remove()
# mixed value = (1-lamb)*v + lamb*v1(block0). Recompute v1 (block0 head value, pre-mix, but v1 is full v of block0).
st0 = {}
hk0 = m.transformer.h[0].register_forward_hook(lambda mm, i, o: st0.__setitem__("v0", F.rms_norm(i[0].detach(), (dm,))))
with torch.no_grad(): m(idx, tgt)
hk0.remove()
Wv0 = m.transformer.h[0].attn.c_v.weight.detach()
v1_head = (st0["v0"] @ Wv0.T).view(B, Tm, nh, hd)[:, :, HEAD, :]
lamb = float(a.lamb)
Vmix = ((1 - lamb) * st2["v"] + lamb * v1_head).detach()   # the head value that enters attention (WHAT)

valid = [(b, q) for b in range(B) for q in range(2, Tm)]

def override(feat, lab2d, means):
    ov = feat.clone()
    for (b, q) in valid:
        c = lab2d[b, q]
        if c >= 0: ov[b, q] = means[c]
    return ov

def cluster_means(feat, lab2d, k):
    out = []
    for c in range(k):
        pts = [feat[b, q] for (b, q) in valid if lab2d[b, q] == c]
        out.append(torch.stack(pts).mean(0) if pts else feat.mean((0, 1)))
    return out

def make_labels(featmat, seed):
    lab = KMeans(K, n_init=6, random_state=seed).fit_predict(featmat)
    L2 = np.full((B, Tm), -1)
    for i, (b, q) in enumerate(valid): L2[b, q] = lab[i]
    return L2

def swap_within(feat, L2):
    mns = cluster_means(feat, L2, K)
    ce_w = float(forward_ce(idx, tgt, **{PATH: override(feat, L2, mns)}))
    swapped = [mns[(c + 1) % K] for c in range(K)]
    ce_s = float(forward_ce(idx, tgt, **{PATH: override(feat, L2, swapped)}))
    return ce_s - ce_w

# feature matrices for clustering
Qmat = torch.stack([qtil[b, q] for (b, q) in valid])                 # query vectors
Qn = F.normalize(Qmat, dim=1)
Qcos = Qn.cpu().numpy()                                              # plain cos (tick25 "query")
Qouter = F.normalize(torch.einsum("ni,nj->nij", Qn, Qn).reshape(len(valid), -1), dim=1).cpu().numpy()  # rank-1 form => cos^2 (|cos|)
Xmat = F.normalize(torch.stack([Xn[b, q] for (b, q) in valid]), dim=1).cpu().numpy()
Vmat = torch.stack([Vmix[b, q] for (b, q) in valid])
Vcos = F.normalize(Vmat, dim=1).cpu().numpy()

print(f"L{LAYER}H{HEAD} squared-attn, K={K}, baseline CE {ce_ref:.4f}, value-mix lamb={lamb:.3f}\n")

def run(name, feat, featmat_or_none, use_x=False, use_rand=False):
    vals = []
    for seed in range(5):
        if use_rand:
            rl = np.random.RandomState(seed).randint(0, K, len(valid)); L2 = np.full((B, Tm), -1)
            for i, (b, q) in enumerate(valid): L2[b, q] = rl[i]
        else:
            L2 = make_labels(featmat_or_none, seed)
        vals.append(swap_within(feat, L2))
    vals = np.array(vals)
    print(f"  {name:34s} {vals.mean():+.4f} +- {vals.std():.4f}")
    return vals.mean()

print("=== (A) QUERY pathway (WHERE to look) — override head query ===")
PATH = "qov"
run("query cos (tick25 baseline)", qtil, Qcos)
run("query cos^2 / rank-1 form (SQUARED)", qtil, Qouter)
run("residual x (control)", qtil, Xmat, use_x=True)
run("random (control)", qtil, None, use_rand=True)

print("\n=== (B) OV / VALUE pathway (WHAT is copied) — override head value ===")
PATH = "vov"
run("value v_k cos", Vmix, Vcos)
run("residual x (control)", Vmix, Xmat, use_x=True)
run("random (control)", Vmix, None, use_rand=True)

print("\n=== (C) full-head contraction: is OV redundant WITHIN the head? ===")
print("  Query pathway clusters tokens by WHERE they look; value pathway by WHAT they emit.")
print("  If these two clusterings are ~the same partition, OV adds nothing to query clustering.")
for seed in range(3):
    Lq = make_labels(Qcos, seed); Lv = make_labels(Vcos, seed)
    from sklearn.metrics import adjusted_rand_score
    aq = [Lq[b, q] for (b, q) in valid]; av = [Lv[b, q] for (b, q) in valid]
    print(f"  seed {seed}: ARI(query-clusters, value-clusters) = {adjusted_rand_score(aq, av):+.4f}")
print("  (ARI~0 => WHERE and WHAT are DIFFERENT partitions => OV carries independent structure,")
print("   so folding OV into a per-query clustering is NOT redundant across the two circuits.)")
print("DONE")
