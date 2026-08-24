# pair_algo2: the MIRROR ARM of §1297's dissociation — context repeats, identity does
# NOT. Plant: copy a 7-token window c1..c7 (+ its continuation y at slot 8) from q to p,
# but replace the pivot token at p+7 with a random token x != c8. At position p+7 the
# current token has never been seen, but the preceding 7-token context exactly matches
# q..q+6 — a context-signature annotation at q+7 can still route the fetch of y; an
# identity mark of c8 cannot (no c8 present to match). Source to annotate: q+7.
#
# Registered predictions:
#   pred_a THE CLEAN MODEL FUZZY-MATCHES AT ALL: base CE at pivot targets <= 6.0
#          (vs ~10.8 chance on random-token rows) — the behaviour exists.
#   pred_b REVERSED DISSOCIATION: 1.8's restore exceeds 1.1's by >= 0.2 (context
#          signature carries the fuzzy match; the identity mark cannot).
#   pred_c CONTROL: kept 1.0 restores <= half of the best single.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'pair_algo2_results.json'
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
    g = torch.Generator().manual_seed(19)
    sel = torch.randint(0, len(vocab_pool), (NR, T + 1), generator=g)
    ROWS = vocab_pool[sel]
    TGT = torch.zeros(NR, T, dtype=torch.bool)
    OPENMASK = torch.zeros(NR, T, dtype=torch.bool)
    K = 8
    for b in range(NR):
        for _ in range(3):
            q = int(torch.randint(20, 100, (1,), generator=g))
            d = int(torch.randint(40, 120, (1,), generator=g))
            p = q + d
            if p + K >= T:
                continue
            ROWS[b, p:p + K - 1] = ROWS[b, q:q + K - 1]   # c1..c7 copied
            # pivot at p+K-1 stays random (identity mismatch); target: predict y=ROWS[q+K]
            ROWS[b, p + K] = ROWS[b, q + K]               # ground truth continuation
            TGT[b, p + K - 1] = True                      # at pivot, predict y
            OPENMASK[b, q + K - 1] = True                 # source c8-position to annotate
    toks = ROWS[:, :-1]
    print(f"planted sources {int(OPENMASK.sum())} | pivot targets {int(TGT.sum())}", flush=True)

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
    pa = base <= 6.0
    pb = res['k18']['restore'] - res['k11']['restore'] >= 0.2
    pc = res['k10ctrl']['restore'] <= 0.5 * max(res['k11']['restore'], res['k18']['restore'], 1e-3)
    out = {'n_rows': NR, 'base': round(base, 4), 'keep_none_dce': round(anchor, 4),
           'conds': res,
           'pred_a_fuzzy_match_exists': bool(pa), 'pred_b_reversed': bool(pb),
           'pred_c_control': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a fuzzy {pa} | pred_b reversed {pb} | pred_c ctrl {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
