"""tick 56 (Logan): transcoder (z->y, output loss) vs secant-SAE (M=yz+, operator loss), TIED vs UNTIED.

Down-projection D of L8: z = Left(h)*Right(h) (hidden, 4608), y = Dz (1152). Compare:
  A transcoder: a=BatchTopK(W_enc z); yhat=W_dec a; loss ||y-yhat||^2/||y||^2  (z-only, predicts y)
  B secant TIED: s_i=(q_i.z+)(p_i.y); Mhat=Σ s_i p_i q_i^T; loss ||M-Mhat||^2/||M||^2  (sees y, analysis)
  C secant UNTIED: code c_i=(a_i.z+)(b_i.y) separate from decode atoms p_i q_i^T  (Goodhart test)
Metrics (held-out): output-FVU (all; secants' uses y -> flag), operator-FVU (secants), dict MMCS
(transcoder{d_i⊗w_i} vs secant{p_i⊗q_i}), and cross-seed stability tied vs untied (Goodhart signature).
m=1024, BatchTopK k=32. ~40k train / 15k held-out.
"""
import sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/workspace/tensor_language")
from huggingface_hub import hf_hub_download
import jacclust.tt_model as TT
torch.set_default_dtype(torch.float32); DEV = "cuda"

def batch_topk_mask(zsel, k):
    nkeep = zsel.shape[0] * k; flat = zsel.abs().reshape(-1)
    thr = flat.kthvalue(flat.numel() - nkeep).values; return zsel.abs() > thr, float(thr)

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
    if len(docs) >= 460: break
idx = torch.tensor(docs, device=DEV)[:, :-1].contiguous(); dm = cfg["n_embd"]
L = 8; Down = m.transformer.h[L].mlp.Down; Dw = Down.weight.detach()   # (1152, 4608)
dz = Dw.shape[1]; dy = Dw.shape[0]
buf = {"z": []}
hk = Down.register_forward_pre_hook(lambda mm, i: buf["z"].append(i[0].detach().reshape(-1, dz).cpu()))
with torch.no_grad():
    for s0 in range(0, idx.shape[0], 24): m(idx[s0:s0+24], idx[s0:s0+24])
hk.remove()
z = torch.cat(buf["z"], 0); y = (z @ Dw.T.cpu())                       # y = Dz (pre-bias, clean linear map)
n = z.shape[0]; perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)); te = perm[:15000]; tr = perm[15000:]
zte = z[te].to(DEV); yte = y[te].to(DEV); zpte = zte / (zte ** 2).sum(1, keepdim=True).clamp_min(1e-9)
print(f"L{L} down-proj: z(hidden)={dz}, y={dy}. {len(tr)} train / 15000 held. m=1024 BatchTopK k=32.\n")
M = 1024; K = 32; STEPS = 3000

def train_transcoder(seed):
    We = (torch.randn(M, dz, generator=torch.Generator().manual_seed(seed)) / dz ** .5).to(DEV).requires_grad_()
    Wd = (torch.randn(dy, M, generator=torch.Generator().manual_seed(seed + 1)) / M ** .5).to(DEV).requires_grad_()
    b = torch.zeros(M, device=DEV, requires_grad=True); opt = torch.optim.Adam([We, Wd, b], 2e-3); g = torch.Generator().manual_seed(seed)
    for _ in range(STEPS):
        bi = tr[torch.randint(0, len(tr), (2048,), generator=g)]; zb = z[bi].to(DEV); yb = y[bi].to(DEV)
        a = zb @ We.T + b; mask, _ = batch_topk_mask(a, K); ah = a * mask; yh = ah @ Wd.T
        (((yh - yb) ** 2).sum(1) / (yb ** 2).sum(1).clamp_min(1e-9)).mean().backward(); opt.step(); opt.zero_grad()
    with torch.no_grad():
        a = zte @ We.T + b; mask, _ = batch_topk_mask(a, K); yh = (a * mask) @ Wd.T
        ofvu = float(((yh - yte) ** 2).sum() / (yte ** 2).sum())
    return We.detach(), Wd.detach(), ofvu

def train_secant(seed, tied=True):
    g0 = torch.Generator().manual_seed(seed)
    q = (torch.randn(M, dz, generator=g0) / dz ** .5).to(DEV).requires_grad_()
    p = (torch.randn(M, dy, generator=torch.Generator().manual_seed(seed + 1)) / dy ** .5).to(DEV).requires_grad_()
    params = [q, p]
    if not tied:
        a = (torch.randn(M, dz, generator=torch.Generator().manual_seed(seed + 2)) / dz ** .5).to(DEV).requires_grad_()
        bb = (torch.randn(M, dy, generator=torch.Generator().manual_seed(seed + 3)) / dy ** .5).to(DEV).requires_grad_()
        params += [a, bb]
    opt = torch.optim.Adam(params, 2e-3); g = torch.Generator().manual_seed(seed)
    def code(zp, yy):
        if tied: return (zp @ q.T) * (yy @ p.T)
        return (zp @ a.T) * (yy @ bb.T)
    def err(c, zp, yy):
        cross = (c * (yy @ p.T) * (zp @ q.T)).sum(1)
        quad = torch.einsum("ti,ij,tj->t", c, (p @ p.T) * (q @ q.T), c)
        mn = (yy ** 2).sum(1) * (zp ** 2).sum(1); return (mn - 2 * cross + quad).clamp_min(0), mn
    for _ in range(STEPS):
        bi = tr[torch.randint(0, len(tr), (2048,), generator=g)]; zb = z[bi].to(DEV); yb = y[bi].to(DEV)
        zp = zb / (zb ** 2).sum(1, keepdim=True).clamp_min(1e-9)
        c = code(zp, yb); mask, _ = batch_topk_mask(c, K); ck = c * mask
        e, mn = err(ck, zp, yb); (e / mn.clamp_min(1e-9)).mean().backward(); opt.step(); opt.zero_grad()
    with torch.no_grad():
        c = code(zpte, yte); mask, thr = batch_topk_mask(c, K); ck = c * mask
        e, mn = err(ck, zpte, yte); opfvu = float(e.sum() / mn.sum())
        yh = torch.einsum("ti,id->td", ck * (zte @ q.T), p)           # Mhat z (uses y in ck -> flag)
        ofvu = float(((yh - yte) ** 2).sum() / (yte ** 2).sum())
    return q.detach(), p.detach(), opfvu, ofvu

def mmcs_rank1(qa, pa, qb, pb):
    qn = F.normalize(qa, dim=1) @ F.normalize(qb, dim=1).T; pn = F.normalize(pa, dim=1) @ F.normalize(pb, dim=1).T
    sim = (qn.abs() * pn.abs()); return float(sim.max(1).values.mean())

We, Wd, t_ofvu = train_transcoder(0)
q_t, p_t, s_opfvu, s_ofvu = train_secant(0, tied=True)
q_u, p_u, u_opfvu, u_ofvu = train_secant(0, tied=False)
print("=== output-FVU (predict y=Dz) — transcoder is the FAIR predictor; secant sees y (flag) ===")
print(f"  transcoder (z->y, output loss)   output-FVU {t_ofvu:.3f}   [fair: z-only]")
print(f"  secant TIED  (operator loss)     output-FVU {s_ofvu:.3f}   operator-FVU {s_opfvu:.3f}   [sees y]")
print(f"  secant UNTIED                    output-FVU {u_ofvu:.3f}   operator-FVU {u_opfvu:.3f}   [sees y]")
# dictionary agreement transcoder vs secant-tied (transcoder atom = d_i(=Wd col) ⊗ w_i(=We row))
print(f"\n  dict MMCS transcoder{{d⊗w}} vs secant-tied{{p⊗q}} = {mmcs_rank1(We, Wd.T, q_t, p_t):.3f}")
# Goodhart: cross-seed stability tied vs untied
qt2, pt2, *_ = train_secant(1, tied=True); qu2, pu2, *_ = train_secant(1, tied=False)
print(f"\n=== Goodhart test: cross-seed dictionary stability (higher=more canonical) ===")
print(f"  secant TIED   stability (seed0 vs seed1): {mmcs_rank1(q_t, p_t, qt2, pt2):.3f}   (op-FVU {s_opfvu:.3f})")
print(f"  secant UNTIED stability (seed0 vs seed1): {mmcs_rank1(q_u, p_u, qu2, pu2):.3f}   (op-FVU {u_opfvu:.3f})")
print("  Goodhart signature = untied op-FVU LOWER (better recon) but stability LOWER (less canonical).")
print("DONE")
