# front_proj_compose4: THE MISSING ARM OF §1656 — measure the PROJECTION family at the
# SAME four sites in the SAME protocol, because §1656's family comparison was
# cross-protocol and I should not have made it.
#
# §1656 measured TABLE composition at mlp0-3 and got joint/sum = 1.1530, then compared
# it to "projections at 1.6". That 1.6 is §541's figure for SIX BLOCKS under rank
# truncation -- a different grain, different rows, different protocol. Comparing my
# n=4-MLP table ratio against it is exactly the cross-denominator error this ledger
# keeps warning about (§1324: "not directly comparable"), and I committed it to the
# registry. This measures the projection family at the SAME four MLPs, SAME rows, SAME
# cost definition, so the comparison becomes apples-to-apples.
#
# PROJECTION ARM: for each site, PCA the module's OUTPUT on the fit rows, keep the top
# r directions, and project the eval-time output onto that fixed basis. Directly
# analogous to the table arm (fitted on fit rows, applied on eval) but in the
# projection family -- it truncates the output rather than indexing it by token.
#
# THREE RANKS, because effect size is a confound. The table arm's individual costs were
# +.218/+1.291/+.336/+.345. If a projection arm's individual costs land in a wildly
# different range, its joint/sum ratio is not comparable to the table's even within one
# protocol. Reporting r in {16, 64, 256} lets the comparison be made at whichever rank
# puts individual costs closest to the table arm's.
#
# ALSO NOTED: my 16:49 board prediction to Codex ("C512 lands above 1.4") is NOT
# scoreable by their C512/MLP1 discriminator, which decomposes the MLP1 mismatch into
# state and write terms rather than producing a joint/sum ratio. That is my error in
# registering it against their run; this experiment produces the number their result
# would have to be compared against, in a protocol I control.
#
# Registered predictions:
#   pred_a THE FAMILY DISTINCTION SURVIVES WITHIN ONE PROTOCOL: at the rank whose
#          individual costs are closest to the table arm's, projection joint/sum
#          EXCEEDS the table arm's 1.1530.
#   pred_b IT IS SUBSTANTIAL, NOT MARGINAL: that same ratio is >= 1.35.
#   pred_c MANIPULATION CHECK: at that rank every individual projection costs >= .01
#          nats, so the sum is signal rather than noise.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
SITES = [0, 1, 2, 3]                 # the four EVALUABLE front MLPs (§1326)
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'front_proj_compose4_results.json'
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip1200.pt'
EVAL_ROWS = PT + '.rowcache/fineweb_n192_skip7000.pt'
RECEIPT = PT + '.rowcache/fineweb_oracle_v2_receipt.json'
H = m.transformer.h
TABLE_RATIO_N4 = 1.1530          # §1656, same four sites, same protocol
TABLE_COSTS_N4 = {0: 0.21791, 1: 1.29080, 2: 0.33570, 3: 0.34464}
RANKS = [16, 64, 256]
S541_CROSS_PROTOCOL = 1.6        # six BLOCKS, different grain -- NOT comparable, kept only as context


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


@torch.no_grad()
def fit_bases(rows):
    """Top-r PCA directions of each site's MLP OUTPUT, fitted on the fit rows.
    Returns {site: (V [D,maxr], mean [D])}; the same fixed basis is applied at eval."""
    maxr = max(RANKS)
    acc = {L: [] for L in SITES}
    cap = {}
    hs = []
    for L in SITES:
        def mk(L):
            def hook(mod, args, out):
                cap[L] = out.float()
                return None
            return hook
        hs.append(H[L].mlp.register_forward_hook(mk(L)))
    try:
        for i in range(0, rows.shape[0], 8):
            idx = rows[i:i + 8, :-1].to(DEV).contiguous()
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            for blk in H:
                x, v1 = blk(x, v1, x0)
            for L in SITES:
                acc[L].append(cap[L].reshape(-1, D))
    finally:
        for h in hs:
            h.remove()
    out = {}
    for L in SITES:
        M = torch.cat(acc[L], 0)
        mu = M.mean(0)
        _, _, Vh = torch.linalg.svd(M - mu, full_matrices=False)
        out[L] = (Vh[:maxr].T.contiguous(), mu)
    return out


@torch.no_grad()
def ce_with_proj(rows, bases, active, r):
    """CE with the given sites' MLP outputs PROJECTED onto their fixed rank-r basis."""
    hs = []
    for L in active:
        def mk(L):
            V, mu = bases[L]
            Vr = V[:, :r]
            def hook(mod, args, out):
                f = out.float()
                d = f.reshape(-1, D) - mu
                rec = (d @ Vr) @ Vr.T + mu
                return rec.reshape(out.shape).to(out.dtype)
            return hook
        hs.append(H[L].mlp.register_forward_hook(mk(L)))
    tot, n = 0.0, 0
    try:
        for i in range(0, rows.shape[0], 8):
            bb = rows[i:i + 8]
            idx = bb[:, :-1].to(DEV).contiguous(); tg = bb[:, 1:].to(DEV)
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            for blk in H:
                x, v1 = blk(x, v1, x0)
            lg = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
            ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                                 reduction='none').reshape(tg.shape)[:, 64:]
            tot += float(ce.sum()); n += ce.numel()
    finally:
        for h in hs:
            h.remove()
    return tot / max(n, 1)


@torch.no_grad()
def main():
    import hashlib
    t0 = time.time()
    fit = load(FIT_ROWS); ev = load(EVAL_ROWS)
    rh = hashlib.sha256(open(RECEIPT, 'rb').read()).hexdigest()[:16]
    print(f'FRONT PROJECTION COMPOSE n=4 | sites {SITES} | fit {tuple(fit.shape)} skip1200 | '
          f'eval {tuple(ev.shape)} skip7000 | receipt {rh}', flush=True)

    bases = fit_bases(fit)
    ce_full = ce_with_proj(ev, bases, [], RANKS[0])
    print(f'  CE full (no substitution) {ce_full:.5f}', flush=True)
    print(f'  TABLE arm reference (§1656, same sites/rows): individual '
          f'{[round(TABLE_COSTS_N4[L],4) for L in SITES]}, joint/sum {TABLE_RATIO_N4}', flush=True)

    by_rank = {}
    for r in RANKS:
        ind = {}
        for L in SITES:
            ce_L = ce_with_proj(ev, bases, [L], r)
            ind[L] = ce_L - ce_full
        ce_j = ce_with_proj(ev, bases, SITES, r)
        joint = ce_j - ce_full
        ssum = sum(ind.values())
        ratio = joint / ssum if ssum > 0 else float('inf')
        # distance from the table arm's individual-cost profile, for matched comparison
        prof = sum(abs(ind[L] - TABLE_COSTS_N4[L]) for L in SITES)
        by_rank[r] = {'individual': {f'mlp{L}': round(ind[L], 5) for L in SITES},
                      'sum': round(ssum, 5), 'joint': round(joint, 5),
                      'joint_over_sum': round(ratio, 4),
                      'profile_distance_to_table': round(prof, 4),
                      'all_ge_01': bool(all(v >= 0.01 for v in ind.values()))}
        print(f'  rank {r:4d}: individual {[round(ind[L],4) for L in SITES]} '
              f'sum {ssum:+.4f} joint {joint:+.4f}  JOINT/SUM {ratio:.4f}  '
              f'|profile-dist {prof:.3f}|', flush=True)

    best = min(RANKS, key=lambda r: by_rank[r]['profile_distance_to_table'])
    b = by_rank[best]
    pa = b['joint_over_sum'] > TABLE_RATIO_N4
    pb = b['joint_over_sum'] >= 1.35
    pc = b['all_ge_01']

    print(f'\n  MATCHED-COST RANK = {best} (closest individual-cost profile to the table arm)',
          flush=True)
    print(f'    projection joint/sum {b["joint_over_sum"]:.4f}  vs  table {TABLE_RATIO_N4}',
          flush=True)
    print(f'    family distinction within ONE protocol: {pa}', flush=True)
    print(f'  (§541\'s 1.6 is six BLOCKS, a different grain -- context only, not the comparator)',
          flush=True)

    out = {'config': {'sites': SITES, 'fit_rows': 'fineweb_n96_skip1200.pt',
                      'eval_rows': 'fineweb_n192_skip7000.pt',
                      'arm': 'PROJECTION -- rank-r PCA of each MLP output, basis fitted on fit rows',
                      'cost_definition': 'CE(table) - CE(full); no ablation constant involved',
                      'table_arm_costs': TABLE_COSTS_N4, 's541_cross_protocol_context_only': S541_CROSS_PROTOCOL},
           'ce_full': round(ce_full, 5), 'by_rank': by_rank,
           'matched_cost_rank': best, 'table_ratio_n4': TABLE_RATIO_N4,
           'predictions': {'pred_a_projection_exceeds_table': bool(pa),
                           'pred_b_ratio_ge_135': bool(pb),
                           'pred_c_manipulation_all_ge_01': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
