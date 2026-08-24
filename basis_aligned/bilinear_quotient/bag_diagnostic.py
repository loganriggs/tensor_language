# bag_diagnostic: WHY did the computed bag fail (§1262)? Compare, position by position, the
# computed far bag's content coordinates against the CLEAN model's actual far-content signal:
# the difference between the clean mid-stream content coords and the W64-masked model's
# content coords (what the crowd actually adds from afar), captured at block-8 entry.
#
# Registered predictions:
#   pred_a THE FORMULA IS WRONG (the §1262-consistent bet): mean per-position cosine between
#          bag coords and the clean-minus-masked content delta <= 0.3.
#   pred_b MAGNITUDE TOO: the bag's coord-norm profile correlates with the true delta's
#          norm profile at r <= 0.5 (not even the "how much" matches).
#   pred_c SANITY: the true delta is nonzero (mean norm >= 0.1 x clean coord norm) — the
#          crowd's far contribution is measurably present at this capture point.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'bag_diagnostic_results.json'
NR = 24; QSTART = 128; WIN = 64; K = 64
NSEQ = 96; SEQ = 256; REF = [8, 10, 12]
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
MASK_W = None
FULL = None


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def content_basis(blocks):
    """standard idiom: top-K PCA of pooled L8-12 mlp-input deviation."""
    cap = {L: [] for L in REF}; hs = []
    for L in REF:
        def mk(L):
            def h(mo, i_, o_): cap[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hs.append(H[L].mlp.register_forward_hook(mk(L)))
    idsL = []
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1)); fwd(idx)
    for h in hs: h.remove()
    tok = torch.cat(idsL, 0); V = int(m.lm_head.weight.shape[0]); devsum = None
    for L in REF:
        X = torch.cat(cap[L], 0); xb = torch.zeros(V, D, device=DEV); cn = torch.zeros(V, device=DEV)
        xb.index_add_(0, tok, X); cn.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
        dv = X - (xb/cn.clamp_min(1).unsqueeze(1))[tok]
        devsum = dv if devsum is None else devsum + dv; cap[L] = []; del X
    dev = devsum/len(REF); dev = dev - dev.mean(0)
    _, _, Vt = torch.linalg.svd(dev, full_matrices=False)
    return Vt[:K].T.contiguous()




@torch.no_grad()
def far_bag(idx, far=True):
    """Recency-weighted bag of v1 codes. far=True: k <= t-WIN; far=False: t-WIN < k <= t."""
    B = idx.shape[0]
    x = F.rms_norm(m.transformer.wte(idx), (D,))
    v1 = m.transformer.h[0].attn.c_v(x)                       # (B,T,D)
    ar = torch.arange(T, device=DEV)
    dist = (ar[:, None] - ar[None, :]).float()
    wmat = torch.exp(-dist / 64.0)
    valid = (dist >= WIN) if far else ((dist > 0) & (dist < WIN))
    wmat = (wmat * valid.float() * torch.tril(torch.ones(T, T, device=DEV))).to(v1.dtype)
    bag = torch.einsum('qk,bkd->bqd', wmat, v1)
    return bag / wmat.sum(-1).clamp_min(1e-6).unsqueeze(0).unsqueeze(-1)




@torch.no_grad()
def stream_at8(idx, masked):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for L, blk in enumerate(m.transformer.h):
        if L == 8:
            return x.detach().float()
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        cos, sin = at.rotary(at.c_q(xin).view(B, T, 9, 128))
        q = are(F.rms_norm(at.c_q(xin).view(B, T, 9, 128), (128,)), cos, sin)
        k = are(F.rms_norm(at.c_k(xin).view(B, T, 9, 128), (128,)), cos, sin)
        q2 = are(F.rms_norm(at.c_q2(xin).view(B, T, 9, 128), (128,)), cos, sin)
        k2 = are(F.rms_norm(at.c_k2(xin).view(B, T, 9, 128), (128,)), cos, sin)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / 128.0) \
            * (torch.einsum('bqhd,bkhd->bhqk', q2.float(), k2.float()) / 128.0)
        pat = pat.masked_fill(~(MASK_W if masked else FULL), 0.0)
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv).reshape(B, T, D)
        x = xm + at.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return None


@torch.no_grad()
def main():
    global MASK_W, FULL
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ar = torch.arange(T, device=DEV)
    vis = ((ar[:, None] - ar[None, :]) < WIN) | (ar[None, :] == 0)
    FULL = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    MASK_W = FULL & vis
    blocks = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    Uc = content_basis(blocks)
    EV = cl.fineweb_rows(NR)[:, :T + 1].contiguous()

    coss = []; bag_norms = []; del_norms = []; clean_norms = []
    for i in range(0, NR, 4):
        idx = EV[i:i + 4, :-1].to(DEV).contiguous()
        xc = stream_at8(idx, False)
        xm = stream_at8(idx, True)
        delta = ((xc - xm) @ Uc)[:, QSTART:]                  # true far-content coords
        bag = far_bag(idx, far=True)
        bagc = (bag.float() @ Uc)[:, QSTART:]
        cs = F.cosine_similarity(delta.reshape(-1, K), bagc.reshape(-1, K), dim=-1)
        coss.append(cs.cpu())
        bag_norms.append(bagc.norm(dim=-1).reshape(-1).cpu())
        del_norms.append(delta.norm(dim=-1).reshape(-1).cpu())
        clean_norms.append((xc @ Uc)[:, QSTART:].norm(dim=-1).reshape(-1).cpu())
    cs = torch.cat(coss); bn = torch.cat(bag_norms); dn = torch.cat(del_norms); cn = torch.cat(clean_norms)
    bz = (bn - bn.mean()) / bn.std().clamp_min(1e-6); dz = (dn - dn.mean()) / dn.std().clamp_min(1e-6)
    rmag = float((bz * dz).mean())
    out = {'n_rows': NR, 'mean_cos': round(float(cs.mean()), 4),
           'mean_abs_cos': round(float(cs.abs().mean()), 4),
           'norm_corr': round(rmag, 4),
           'delta_norm_over_clean': round(float((dn / cn.clamp_min(1e-6)).mean()), 4),
           'pred_a_formula_wrong': bool(float(cs.mean()) <= 0.3),
           'pred_b_magnitude_too': bool(rmag <= 0.5),
           'pred_c_delta_nonzero': bool(float((dn / cn.clamp_min(1e-6)).mean()) >= 0.1),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"mean cos {out['mean_cos']} (|cos| {out['mean_abs_cos']}) | norm corr {rmag:.3f} | delta/clean {out['delta_norm_over_clean']}")
    print(f"pred_a wrong {out['pred_a_formula_wrong']} | pred_b mag {out['pred_b_magnitude_too']} | pred_c nonzero {out['pred_c_delta_nonzero']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
