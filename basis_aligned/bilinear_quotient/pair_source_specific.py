# pair_source_specific: §1299's clean isolation. "Zero ALL front heads at position set X"
# is ~70% generic damage; here the ablation is PAIR-SCOPED — zero ONLY 1.1 and 1.8's
# y-slices at (a) source positions, (b) count-matched random positions, (c) target
# positions. Matched mass, minimal scope: any source excess over random is annotation-
# specific by construction.
#
# Registered predictions:
#   pred_a SOURCE-SPECIFIC: pair-at-sources dCE >= 3x pair-at-random.
#   pred_b SMALL ABSOLUTES: all three pair-scoped dCEs <= 0.10 (redundancy beyond the
#          pair covers most of the function — §1296).
#   pred_c NO QUERY-SIDE ROLE: pair-at-targets <= 1.5x pair-at-random.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'pair_source_specific_results.json'
NR = 192
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
FRONT = [0, 1, 2]


PAIR = ((1, 1), (1, 8))


@torch.no_grad()
def forward_pairzero(idx, posmask):
    """Zero ONLY heads 1.1 and 1.8's y-slices at posmask positions (None = clean)."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    pm = None if posmask is None else posmask.to(x.dtype).unsqueeze(-1)
    for L, blk in enumerate(m.transformer.h):
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
        tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        pat = pat.masked_fill(~tril, 0.0)
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv).reshape(B, T, D)
        if pm is not None:
            zvec = torch.zeros(D, device=DEV, dtype=y.dtype)
            for (LL, hh) in PAIR:
                if LL == L:
                    zvec[hh * 128:(hh + 1) * 128] = 1.0
            if zvec.any():
                y = y * (1 - pm * zvec)
        x = xm + at.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous()
    toks = ROWS[:, :-1]; tgt_all = ROWS[:, 1:]
    Wd = 128
    TGT = torch.zeros_like(toks, dtype=torch.bool)
    OPENMASK = torch.zeros_like(toks, dtype=torch.bool)
    for b0 in range(0, NR, 64):
        tb = toks[b0:b0 + 64]; gb = tgt_all[b0:b0 + 64]
        eq = (tb.unsqueeze(1) == tb.unsqueeze(2)) & (gb.unsqueeze(1) == gb.unsqueeze(2))
        q_i = torch.arange(T).view(1, T, 1); p_i = torch.arange(T).view(1, 1, T)
        band = (q_i < p_i) & (q_i >= p_i - Wd)
        rel = eq & band
        TGT[b0:b0 + 64] = rel.any(1)
        OPENMASK[b0:b0 + 64] = rel.any(2)
    TGT[:, :16] = False
    print(f"source positions {int(OPENMASK.sum())} | induction targets {int(TGT.sum())}", flush=True)

    def run_mask(mask):
        ce_t = 0.0; n_t = 0
        for i in range(0, NR, 4):
            bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = forward_pairzero(idx, None if mask is None else mask[i:i + 4].to(DEV)).float()
            lse = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1), reduction='none').view(tg.shape)
            tm = TGT[i:i + 4].to(DEV)
            ce_t += float(lse[tm].sum()); n_t += int(tm.sum())
        return ce_t / max(n_t, 1)

    base = run_mask(None)
    g = torch.Generator().manual_seed(23)
    RANDPOS = torch.zeros_like(OPENMASK)
    flat = torch.randperm(OPENMASK.numel(), generator=g)[:int(OPENMASK.sum())]
    RANDPOS.view(-1)[flat] = True
    RANDPOS &= ~OPENMASK & ~TGT
    d_src = run_mask(OPENMASK) - base
    d_rnd = run_mask(RANDPOS) - base
    d_tgt = run_mask(TGT) - base
    pa = d_src >= 3 * max(d_rnd, 1e-4)
    pb = max(d_src, d_rnd, d_tgt) <= 0.10
    pc = d_tgt <= 1.5 * max(d_rnd, 1e-4)
    out = {'n_rows': NR, 'base': round(base, 4),
           'dce': {'pair_at_sources': round(d_src, 4), 'pair_at_random': round(d_rnd, 4),
                   'pair_at_targets': round(d_tgt, 4)},
           'pred_a_source_specific': bool(pa), 'pred_b_small': bool(pb),
           'pred_c_no_query_role': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"src {d_src:.4f} | rand {d_rnd:.4f} | tgt {d_tgt:.4f}")
    print(f"pred_a specific {pa} | pred_b small {pb} | pred_c no-query {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
