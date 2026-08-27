# mlp0_table_rank_curve: WHAT DOES A TOKEN TABLE FOR MLP0 COST AT EACH RANK?
#
# §1324 measured mlp0's token-table CEILING at 86.3% of a 0.7994-nat stake, with the
# curve traced over K = number of token CLUSTERS (1/4/16/64/256/1024 -> 0/7/37/56/65/69%).
# §780 measured the per-token mean table's EFFECTIVE RANK at 22.7 against the
# embedding's 132.4 -- tokens collapse into a ~23-dim space.
#
# Neither traced the curve over the table's RANK, and that is the axis Codex needs.
# Their compiler prices a rank-64 PCA basis plus an affine predictor (153,920 reals per
# site). A token table compressed to rank r is a directly comparable object with a
# directly comparable price, and unlike their P and O arms it is EXECUTABLE -- a lookup
# makes no original-MLP call. This produces the recovery-vs-rank curve so a table and a
# compiled map can be compared at matched price instead of by assertion.
#
# PROTOCOL, stated plainly because it is not Codex's. Fit the per-token mean of mlp0's
# OUTPUT on a fit split, SVD it, and at each rank replace mlp0's output with the
# rank-r table lookup on a disjoint eval split. Recovery is measured against the
# module's own mean-ablation stake on the SAME rows:
#
#     recovery(r) = (CE_meanablate - CE_table_r) / (CE_meanablate - CE_full)
#
# That is §1324's denominator, not Codex's exact-restoration-on-a-frozen-ship
# denominator. The numbers here are NOT importable into their accounting; what
# transfers is the SHAPE of the price curve.
#
# Rows: canonical .rowcache -- fit on n96_skip1200, eval on n192_skip7000, which are
# document-disjoint by construction (different skips) and are the house eval convention.
#
# Registered predictions:
#   pred_a THE §780 EFFECTIVE RANK IS ENOUGH: rank-23 recovers >= 80% of what the FULL
#          (uncompressed) table recovers.
#   pred_b THE CURVE SATURATES: rank-64 buys <= 5 recovery POINTS over rank-23, i.e.
#          quadrupling the rank past the effective rank is nearly free of gain.
#   pred_c EVEN A VERY CHEAP TABLE IS WORTH SOMETHING: rank-8 recovers >= 40% of the
#          full-table recovery.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
SITE = 0
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp0_table_rank_curve_results.json'
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip1200.pt'
EVAL_ROWS = PT + '.rowcache/fineweb_n192_skip7000.pt'
RECEIPT = PT + '.rowcache/fineweb_oracle_v2_receipt.json'
H = m.transformer.h
RANKS = [1, 2, 4, 8, 16, 23, 32, 64, 128]     # 23 = §780's measured effective rank
S780_EFFECTIVE_RANK = 22.7
S1324_CEILING = 0.863


def load(p, n=None):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return (r if n is None else r[:n])[:, :T + 1].contiguous()


@torch.no_grad()
def fit_table(rows):
    """Per-token mean of mlp0's OUTPUT, indexed by the INPUT token at that position."""
    s = torch.zeros(50257, D, device=DEV, dtype=torch.float32)
    c = torch.zeros(50257, device=DEV, dtype=torch.float32)
    cap = {}

    def hook(mod, args, out):
        cap['o'] = out.float()
        return None
    h = H[SITE].mlp.register_forward_hook(hook)
    try:
        for i in range(0, rows.shape[0], 8):
            idx = rows[i:i + 8, :-1].to(DEV).contiguous()
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            for blk in H:
                x, v1 = blk(x, v1, x0)
            o = cap['o'].reshape(-1, D)
            t = idx.reshape(-1)
            s.index_add_(0, t, o)
            c.index_add_(0, t, torch.ones_like(t, dtype=torch.float32))
    finally:
        h.remove()
    seen = c > 0
    tbl = torch.zeros_like(s)
    tbl[seen] = s[seen] / c[seen].unsqueeze(1)
    return tbl, seen, c


def rank_truncate(tbl, seen, r):
    """Rank-r truncation of the table, computed over SEEN rows only."""
    sub = tbl[seen]
    mu = sub.mean(0, keepdim=True)
    U, S, Vh = torch.linalg.svd(sub - mu, full_matrices=False)
    k = min(r, S.shape[0])
    rec = (U[:, :k] * S[:k]) @ Vh[:k] + mu
    out = torch.zeros_like(tbl)
    out[seen] = rec
    return out


@torch.no_grad()
def ce_with(rows, mode, tbl=None, seen=None):
    """mode: 'full' | 'mean' (mean-ablate mlp0) | 'table' (replace with lookup)."""
    h = None
    if mode == 'mean':
        gm = tbl[seen].mean(0) if tbl is not None else None
        def hk(mod, args, out):
            return gm.to(out.dtype).expand_as(out)
        h = H[SITE].mlp.register_forward_hook(hk)
    elif mode == 'table':
        def hk(mod, args, out):
            t = hk.idx.reshape(-1)
            return tbl[t].reshape(out.shape).to(out.dtype)
        h = H[SITE].mlp.register_forward_hook(hk)
    tot, n = 0.0, 0
    try:
        for i in range(0, rows.shape[0], 8):
            bb = rows[i:i + 8]
            idx = bb[:, :-1].to(DEV).contiguous(); tg = bb[:, 1:].to(DEV)
            if mode == 'table':
                hk.idx = idx
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            for blk in H:
                x, v1 = blk(x, v1, x0)
            lg = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
            ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                                 reduction='none').reshape(tg.shape)
            ce = ce[:, 64:]
            tot += float(ce.sum()); n += ce.numel()
    finally:
        if h is not None:
            h.remove()
    return tot / max(n, 1)


@torch.no_grad()
def main():
    import hashlib
    t0 = time.time()
    fit = load(FIT_ROWS); ev = load(EVAL_ROWS)
    rh = hashlib.sha256(open(RECEIPT, 'rb').read()).hexdigest()[:16]
    print(f'mlp{SITE} TABLE RANK CURVE | fit {tuple(fit.shape)} (skip1200) | '
          f'eval {tuple(ev.shape)} (skip7000) | receipt {rh}', flush=True)

    tbl, seen, cnt = fit_table(fit)
    print(f'  table fitted: {int(seen.sum())} distinct tokens seen, '
          f'{int(cnt.sum())} positions', flush=True)

    ce_full = ce_with(ev, 'full')
    ce_mean = ce_with(ev, 'mean', tbl, seen)
    stake = ce_mean - ce_full
    print(f'  CE full {ce_full:.5f} | CE mean-ablated {ce_mean:.5f} | '
          f'STAKE {stake:.5f} nats  (§1324 reported .7994 on its own rows/protocol)',
          flush=True)
    assert stake > 1e-4, 'mean-ablation of mlp0 costs nothing -- instrument broken'

    rows_out = {}
    ce_fulltable = ce_with(ev, 'table', tbl, seen)
    rec_full = (ce_mean - ce_fulltable) / stake
    print(f'  FULL table: CE {ce_fulltable:.5f}  recovery {rec_full:6.2%}  '
          f'(§1324 ceiling {S1324_CEILING:.1%} on its own protocol)', flush=True)

    for r in RANKS:
        tr = rank_truncate(tbl, seen, r)
        ce_r = ce_with(ev, 'table', tr, seen)
        rec = (ce_mean - ce_r) / stake
        rows_out[r] = {'ce': round(ce_r, 5), 'recovery': round(rec, 5),
                       'frac_of_full_table': round(rec / rec_full, 5) if rec_full else None,
                       'reals': int(seen.sum()) * r + r * D}
        print(f'  rank {r:4d}: CE {ce_r:.5f}  recovery {rec:6.2%}  '
              f'= {rec/rec_full:6.2%} of full table  | {rows_out[r]["reals"]:,} reals',
              flush=True)

    f23 = rows_out[23]['frac_of_full_table']
    pa = f23 >= 0.80
    pb = (rows_out[64]['recovery'] - rows_out[23]['recovery']) <= 0.05
    pc = rows_out[8]['frac_of_full_table'] >= 0.40

    print(f'\n  rank-23 (§780 effective rank {S780_EFFECTIVE_RANK}) reaches '
          f'{f23:.1%} of the full table', flush=True)
    print(f'  rank-64 minus rank-23 = '
          f'{(rows_out[64]["recovery"]-rows_out[23]["recovery"])*100:+.2f} recovery points',
          flush=True)
    print(f'  rank-8 reaches {rows_out[8]["frac_of_full_table"]:.1%} of the full table',
          flush=True)

    out = {'config': {'site': SITE, 'ranks': RANKS,
                      'fit_rows': 'fineweb_n96_skip1200.pt', 'eval_rows': 'fineweb_n192_skip7000.pt',
                      'protocol': ('recovery = (CE_meanablate - CE_table_r) / (CE_meanablate - CE_full), '
                                   'measured on the running model. This is §1324-style, NOT Codex\'s '
                                   'exact-restoration-on-a-frozen-ship denominator; numbers are not '
                                   'importable into their accounting.'),
                      's780_effective_rank': S780_EFFECTIVE_RANK, 's1324_ceiling': S1324_CEILING},
           'ce_full': round(ce_full, 5), 'ce_meanablated': round(ce_mean, 5),
           'stake_nats': round(stake, 5), 'full_table_recovery': round(rec_full, 5),
           'tokens_seen': int(seen.sum()), 'by_rank': rows_out,
           'predictions': {'pred_a_rank23_ge_80pct_of_full': bool(pa),
                           'pred_b_rank64_adds_le_5_points': bool(pb),
                           'pred_c_rank8_ge_40pct_of_full': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
