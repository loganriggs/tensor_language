"""WHY does two-branch attention route induction through MLP1 while softmax does it in heads?
Hypothesis: induction needs a PREVIOUS-TOKEN signal (key at position j encodes token[j-1]); bilin18
(two-branch) builds it in an MLP, softmax models build it in an early attention head. Test on the
two 18-layer models (bilin18 two-branch vs swiglu18 softmax): (a) previous-token decodability from
the residual by depth (linear probe residual[j] -> embedding of token[j-1], R^2); (b) causal
attribution -- at the layer where prev-token info is built, ablate that block's MLP vs its ATTENTION
and see which collapses the signal.
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))


def make_model(short):
    m, cfg = load_elriggs(short)
    NH, D = cfg['n_head'], cfg['n_embd']; HD = D // NH; V = cfg['vocab_size']; NL = len(m.transformer.h)
    two_branch = bool(cfg.get('bilinear_attn')) and bool(cfg.get('squared_attn'))
    sq = bool(cfg.get('squared_attn'))
    wte = m.transformer.wte.weight.detach().float().to(DEV)
    EPC = torch.linalg.svd(F.rms_norm(wte, (D,)) - F.rms_norm(wte, (D,)).mean(0), full_matrices=False).Vh[:64].T.contiguous()
    EMB = F.rms_norm(wte, (D,)) @ EPC   # (V,64) previous-token target codes
    return dict(m=m, cfg=cfg, NH=NH, HD=HD, D=D, V=V, NL=NL, two_branch=two_branch, sq=sq, EMB=EMB)


@torch.no_grad()
def forward(M, idx, ablate_mlp=frozenset(), ablate_attn=frozenset(), MEAN=None, collect_mean=False, cblocks=None):
    m = M['m']; NH, HD, D, NL = M['NH'], M['HD'], M['D'], M['NL']
    B, T = idx.shape; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    means = {}; res = {}
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        if M['two_branch']:
            q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
            pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        elif M['sq']:
            q, k = qk(a.c_q), qk(a.c_k); s = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD
            pat = s.square().masked_fill(~mask, 0.0); pat = pat/pat.sum(-1, keepdim=True).clamp_min(1e-9)
            yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        else:
            qh, kh = qk(a.c_q), qk(a.c_k)
            yh4 = F.scaled_dot_product_attention(qh.transpose(1, 2), kh.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2)
        if collect_mean: means[('a', li)] = yh4.mean((0, 1))
        if li in ablate_attn: yh4 = MEAN[('a', li)].expand_as(yh4)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if collect_mean: means[('m', li)] = mo.mean((0, 1))
        if li in ablate_mlp: mo = MEAN[('m', li)].expand_as(mo)
        x = x + mo
        if cblocks is not None and li in cblocks: res[li] = x.reshape(-1, D)
    return res, means


def prevtok_probe(M, Rtr, Ytr, Rte, Yte):
    Xtr = torch.cat([Rtr, torch.ones(Rtr.shape[0], 1, device=DEV)], 1).double()
    Xte = torch.cat([Rte, torch.ones(Rte.shape[0], 1, device=DEV)], 1).double()
    W = torch.linalg.solve(Xtr.T @ Xtr + 50*torch.eye(Xtr.shape[1], device=DEV, dtype=torch.double), Xtr.T @ Ytr.double())
    pred = Xte @ W
    ss_res = (pred - Yte.double()).pow(2).sum(); ss_tot = (Yte.double() - Yte.double().mean(0)).pow(2).sum()
    return float(1 - ss_res/ss_tot)


def run_model(short):
    M = make_model(short); NL = M['NL']; EMB = M['EMB']
    CB = list(range(min(7, NL)))    # probe residual after blocks 0..6
    # gather train/test residuals + previous-token targets (token[j-1])
    def gather(rng, **kw):
        R = {b: [] for b in CB}; Y = []
        for i in rng:
            idx = FINEWEB[i:i+4, :128].to(DEV)
            res, _ = forward(M, idx, cblocks=CB, **kw)
            prev = idx[:, :-1]                    # token[j-1] for query positions j=1..T-1
            for b in CB: R[b].append(res[b].view(4, 128, -1)[:, 1:, :].reshape(-1, M['D']))
            Y.append(EMB[prev.reshape(-1)])
        return {b: torch.cat(R[b]) for b in CB}, torch.cat(Y)
    _, MEAN = forward(M, FINEWEB[:16, :128].to(DEV), collect_mean=True, cblocks=[])
    Rtr, Ytr = gather(range(0, 160, 4)); Rte, Yte = gather(range(300, 380, 4))
    curve = {b: round(prevtok_probe(M, Rtr[b], Ytr, Rte[b], Yte), 4) for b in CB}
    print(f"[{short}] prev-token R2 by block: " + " ".join(f"b{b}={curve[b]:.3f}" for b in CB), flush=True)
    # find the block with the biggest jump in prev-token R2
    jumps = {b: curve[b] - (curve[b-1] if b-1 in curve else 0.0) for b in CB}
    jb = max(CB, key=lambda b: jumps[b])
    # causal: ablate block jb's MLP vs its attention, re-probe at block jb
    def gather_one(rng, b, **kw):
        R = []; Y = []
        for i in rng:
            idx = FINEWEB[i:i+4, :128].to(DEV)
            res, _ = forward(M, idx, cblocks=[b], MEAN=MEAN, **kw)
            R.append(res[b].view(4, 128, -1)[:, 1:, :].reshape(-1, M['D'])); Y.append(EMB[idx[:, :-1].reshape(-1)])
        return torch.cat(R), torch.cat(Y)
    r2_intact = curve[jb]
    Rtr_m, Ytr_m = gather_one(range(0, 160, 4), jb, ablate_mlp={jb}); Rte_m, Yte_m = gather_one(range(300, 380, 4), jb, ablate_mlp={jb})
    r2_no_mlp = round(prevtok_probe(M, Rtr_m, Ytr_m, Rte_m, Yte_m), 4)
    Rtr_a, Ytr_a = gather_one(range(0, 160, 4), jb, ablate_attn={jb}); Rte_a, Yte_a = gather_one(range(300, 380, 4), jb, ablate_attn={jb})
    r2_no_attn = round(prevtok_probe(M, Rtr_a, Ytr_a, Rte_a, Yte_a), 4)
    print(f"[{short}] biggest-jump block {jb}: R2 intact {r2_intact:.3f} | ablate MLP{jb} {r2_no_mlp:.3f} | ablate ATTN{jb} {r2_no_attn:.3f}", flush=True)
    return {'curve': curve, 'jump_block': jb, 'r2_intact': r2_intact, 'r2_ablate_mlp': r2_no_mlp, 'r2_ablate_attn': r2_no_attn,
            'built_by': 'MLP' if (r2_intact - r2_no_mlp) > (r2_intact - r2_no_attn) else 'ATTN',
            'drop_mlp': round(r2_intact - r2_no_mlp, 4), 'drop_attn': round(r2_intact - r2_no_attn, 4)}

res = {}
for short in ['bilin18', 'swiglu18', 'bilinsm12']:
    res[short] = run_model(short)
    print(f"  -> {short}: previous-token signal built by {res[short]['built_by']} "
          f"(MLP drop {res[short]['drop_mlp']}, ATTN drop {res[short]['drop_attn']})\n", flush=True)
json.dump(res, open(f'{QK}/qk_prevtoken_source.json', 'w'), indent=2)
print("QK PREVTOKEN SOURCE DONE", flush=True)
