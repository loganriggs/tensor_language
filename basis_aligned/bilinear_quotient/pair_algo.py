# pair_algo: the §1296 discriminating cell — identity-mark (1.1, local read) vs context-
# signature (1.8, global read) as two interchangeable annotation algorithms. Dataset
# decouples them: SYNTHETIC rows of random tokens with planted repeated bigrams — token
# identity repeats across the pair sites, surrounding context does NOT. If 1.8's
# annotation is a context signature, it has nothing to match here and its keep-one-alive
# restore should collapse; 1.1's identity mark should survive.
#
# Registered predictions (natural-text anchors from writer_cross2: 1.1 70%, 1.8 84%):
#   pred_a ANNOTATION NEEDED ON SYNTH TOO: keep-none costs >= 0.5 nats at planted targets.
#   pred_b ALGORITHM DISCRIMINATION: 1.1's restore exceeds 1.8's by >= 20 points
#          (reversing the natural-text order).
#   pred_c PAIR >= BEST SINGLE (interchangeability where both apply, superiority where
#          one does): pair restore >= max(single restores).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'pair_algo_results.json'
NR = 192
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
FRONT = [0, 1, 2]


@torch.no_grad()
def forward_keep(idx, keep, posmask):
    """Zero y-slices of all FRONT-layer heads at posmask positions, except `keep`=(L,h).
    keep=None -> keep-none; posmask=None -> clean forward."""
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
        if pm is not None and L in FRONT:
            keepvec = torch.zeros(D, device=DEV, dtype=y.dtype)
            for (LL, hh) in (keep or ()):
                if LL == L:
                    keepvec[hh * 128:(hh + 1) * 128] = 1.0
            y = y * (1 - pm) + y * pm * keepvec
        x = xm + at.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ROWS_N = cl.fineweb_rows(NR)[:, :T + 1].contiguous()
    vocab_pool = torch.unique(ROWS_N.reshape(-1))
    g = torch.Generator().manual_seed(17)
    sel = torch.randint(0, len(vocab_pool), (NR, T + 1), generator=g)
    ROWS = vocab_pool[sel]
    TGT = torch.zeros(NR, T, dtype=torch.bool)
    OPENMASK = torch.zeros(NR, T, dtype=torch.bool)
    for b in range(NR):
        for _ in range(3):
            q = int(torch.randint(20, 110, (1,), generator=g))
            d = int(torch.randint(40, 120, (1,), generator=g))
            p = q + d
            if p + 1 >= T:
                continue
            ROWS[b, p] = ROWS[b, q]
            ROWS[b, p + 1] = ROWS[b, q + 1]
            TGT[b, p] = True          # predicting toks[p+1] == toks[q+1]
            OPENMASK[b, q] = True     # the source to be annotated
    toks = ROWS[:, :-1]
    print(f"planted sources {int(OPENMASK.sum())} | planted targets {int(TGT.sum())}", flush=True)

    def run(keep, use_mask=True):
        ce_t = 0.0; n_t = 0
        for i in range(0, NR, 4):
            bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            pmask = OPENMASK[i:i + 4].to(DEV) if use_mask else None
            lo = forward_keep(idx, keep, pmask).float()
            lse = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1), reduction='none').view(tg.shape)
            tm = TGT[i:i + 4].to(DEV)
            ce_t += float(lse[tm].sum()); n_t += int(tm.sum())
        return ce_t / max(n_t, 1)

    base = run(None, use_mask=False)
    anchor = run(None) - base
    print(f"base {base:.4f} | keep-none dCE {anchor:.4f}", flush=True)
    res = {}
    for name, keep in (('k11', {(1, 1)}), ('k18', {(1, 8)}),
                       ('k11_18', {(1, 1), (1, 8)}), ('k10ctrl', {(1, 0)})):
        d = run(keep) - base
        res[name] = {'dce': round(d, 4), 'restore': round(1 - d / max(anchor, 1e-6), 3)}
        print(f"{name}: dCE {d:.4f} restore {res[name]['restore']:.3f}", flush=True)
    pa = anchor >= 0.5
    pb = res['k11']['restore'] - res['k18']['restore'] >= 0.2
    pc = res['k11_18']['restore'] >= max(res['k11']['restore'], res['k18']['restore']) - 0.02
    out = {'n_rows': NR, 'base': round(base, 4), 'keep_none_dce': round(anchor, 4),
           'conds': res,
           'pred_a_annotation_needed': bool(pa), 'pred_b_discrimination': bool(pb),
           'pred_c_pair_best': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a needed {pa} | pred_b discrim {pb} | pred_c pair-best {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
