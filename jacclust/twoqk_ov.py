"""tick 29 (autonomous, closes LOG Q1): OV / WHERE x WHAT factorization on the 500M TWO-QK model.

tick 26 (S16) found on the 12l single-QK model: the OV/value pathway ("what is copied") is a causal
clustering axis independent of the query pathway ("where to look"), head = WHERE(query) x WHAT(OV),
ARI(query-clusters, value-clusters) ~ 0.07. Confirm on the validated 500M two-QK model.

Cluster context tokens by their head value v_k; causally swap the written value (own cluster mean vs next
cluster's). Controls: residual-x clusters (same intervention) + random. 5 seeds. Then ARI(query, value)
to test independence. Causal heads from ticks 27-28: L6H3, L8H3, L0H2 (+ L9H3, L11H5 for spread).
Same swap-within metric as ticks 23-28.
"""
import sys, torch, json, numpy as np, torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
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
K = 8; nh, hd = m.transformer.h[0].attn.n_head, m.transformer.h[0].attn.head_dim

def apply_rot(x, c, s):
    d = x.shape[-1] // 2; x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * c + x2 * s, -x1 * s + x2 * c], -1)

def forward_ce(idx, tgt, LAYER=-1, HEAD=-1, vov=None):
    x = m.transformer.wte(idx); x = F.rms_norm(x, (x.size(-1),)); x0 = x; v1 = None
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0; a = blk.attn
        h = F.rms_norm(x, (x.size(-1),)); B, T, C = h.shape
        q = a.c_q(h).view(B, T, nh, hd); k = a.c_k(h).view(B, T, nh, hd)
        q2 = a.c_q2(h).view(B, T, nh, hd); k2 = a.c_k2(h).view(B, T, nh, hd); v = a.c_v(h).view(B, T, nh, hd)
        if v1 is None: v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        if li == LAYER and vov is not None: v = v.clone(); v[:, :, HEAD, :] = vov
        cos, sin = a.rotary(q); cos = cos.float(); sin = sin.float()
        q = F.rms_norm(q, (hd,)); k = F.rms_norm(k, (hd,)); q2 = F.rms_norm(q2, (hd,)); k2 = F.rms_norm(k2, (hd,))
        q = apply_rot(q, cos, sin); k = apply_rot(k, cos, sin); q2 = apply_rot(q2, cos, sin); k2 = apply_rot(k2, cos, sin)
        sc = torch.einsum("bqhd,bkhd->bhqk", q, k) / hd; sc2 = torch.einsum("bqhd,bkhd->bhqk", q2, k2) / hd
        mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool)); pat = (sc * sc2).masked_fill(~mask, 0.0)
        y = torch.einsum("bhqk,bkhd->bqhd", pat, v).reshape(B, T, C); y = a.c_proj(y); x = x + y
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    x = F.rms_norm(x, (x.size(-1),)); logits = 30 * torch.tanh(m.lm_head(x) / 30)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1))

ds = datasets.load_dataset("NeelNanda/pile-10k", split="train", streaming=True); docs = []
for d in ds:
    t = tok(d["text"])["input_ids"]
    if len(t) >= 66: docs.append(t[:66])
    if len(docs) >= 24: break
toks = torch.tensor(docs, device=DEV); idx = toks[:, :-1].contiguous(); tgt = toks[:, 1:].contiguous()
with torch.no_grad(): ce_ref = float(m(idx, tgt)); assert abs(ce_ref - float(forward_ce(idx, tgt))) < 1e-3
B, Tm = idx.shape; valid = [(b, q) for b in range(B) for q in range(2, Tm)]

# block-0 head value (v1) for the value-residual mix
Wv0 = m.transformer.h[0].attn.c_v.weight.detach()
st0 = {}; hk0 = m.transformer.h[0].register_forward_hook(lambda mm, i, o: st0.__setitem__("v0", F.rms_norm(i[0].detach(), (i[0].shape[-1],))))
with torch.no_grad(): m(idx, tgt)
hk0.remove()
v1_full = (st0["v0"] @ Wv0.T).view(B, Tm, nh, hd)

def readouts(L, H):
    a = m.transformer.h[L].attn
    st = {}; hk = m.transformer.h[L].register_forward_hook(lambda mm, i, o: st.__setitem__("r", i[0].detach()))
    with torch.no_grad(): m(idx, tgt)
    hk.remove(); Xn = F.rms_norm(st["r"], (st["r"].shape[-1],))
    q1 = F.rms_norm((Xn @ a.c_q.weight.detach().T).view(B, Tm, nh, hd)[:, :, H, :], (hd,)).detach()
    q2 = F.rms_norm((Xn @ a.c_q2.weight.detach().T).view(B, Tm, nh, hd)[:, :, H, :], (hd,)).detach()
    vh = (Xn @ a.c_v.weight.detach().T).view(B, Tm, nh, hd)[:, :, H, :]
    Vmix = ((1 - float(a.lamb)) * vh + float(a.lamb) * v1_full[:, :, H, :]).detach()
    return Xn.detach(), q1, q2, Vmix

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
def sw_value(L, H, Vmix, L2):
    mns = cmeans(Vmix, L2)
    cw = float(forward_ce(idx, tgt, L, H, override(Vmix, L2, mns)))
    cs = float(forward_ce(idx, tgt, L, H, override(Vmix, L2, [mns[(c + 1) % K] for c in range(K)])))
    return cs - cw

print(f"500M two-QK, OV/value pathway. baseline CE {ce_ref:.4f}\n")
print(f"{'head':7s} {'value-cos':>16s} {'resid-x':>14s} {'random':>12s} {'ARI(qry,val)':>13s}")
for (L, H) in [(0, 2), (6, 3), (8, 3), (9, 3), (11, 5)]:
    Xn, q1, q2, Vmix = readouts(L, H)
    Vcos = F.normalize(torch.stack([Vmix[b, q] for (b, q) in valid]), dim=1).cpu().numpy()
    Xm = F.normalize(torch.stack([Xn[b, q] for (b, q) in valid]), dim=1).cpu().numpy()
    q1n = F.normalize(torch.stack([q1[b, q] for (b, q) in valid]), dim=1)
    q2n = F.normalize(torch.stack([q2[b, q] for (b, q) in valid]), dim=1)
    bothq = torch.cat([q1n, q2n], 1).cpu().numpy()
    vv = np.array([sw_value(L, H, Vmix, labels(Vcos, s)) for s in range(5)])
    xx = np.array([sw_value(L, H, Vmix, labels(Xm, s)) for s in range(5)])
    rr = []
    for s in range(5):
        rl = np.random.RandomState(s).randint(0, K, len(valid)); L2 = np.full((B, Tm), -1)
        for i, (b, q) in enumerate(valid): L2[b, q] = rl[i]
        rr.append(sw_value(L, H, Vmix, L2))
    rr = np.array(rr)
    aris = [adjusted_rand_score([labels(bothq, s)[b, q] for (b, q) in valid],
                                [labels(Vcos, s)[b, q] for (b, q) in valid]) for s in range(3)]
    print(f"L{L}H{H:<4d} {vv.mean():+.4f}+-{vv.std():.4f} {xx.mean():+.4f}+-{xx.std():.4f} {rr.mean():+.4f}+-{rr.std():.4f} {np.mean(aris):+.4f}", flush=True)
print("\nvalue-cos > resid-x > random => OV pathway is a real causal clustering axis.")
print("ARI(query,value) ~ 0 => WHERE and WHAT are independent partitions (head = WHERE x WHAT), as on 12l.")
print("DONE")
