# artifact_ceiling: HOW BIG CAN THE EFFECT-SIZE ARTIFACT GET? — bounding it, so the
# flag §1657 raised against §1616's 8.9x can be resolved rather than left hanging.
#
# §1657 showed joint/sum tracks total effect size: sum 2.55 -> 1.03, 2.19 -> 1.15,
# 0.66 -> 2.14, 0.055 -> 2.16. I then flagged §1616's "early block is a COUPLED causal
# program" (joint .5115 vs singleton sum .0573 = 8.9x) as possibly carrying the same
# artifact, since .0573 is squarely in the small-effect regime.
#
# But TWO of my points sat near 2.15 across a 12x range of effect size (sum .66 and
# .055), which HINTS the artifact plateaus rather than growing without bound. If it
# does plateau near ~2, then an artifact cannot produce 8.9x and §1616's coupling is
# real. If it grows without bound as effects shrink, §1616 needs re-examination.
#
# Two points do not establish a plateau. This measures the curve across eight ranks
# spanning a ~100x range of effect size, at the same four front MLPs, same rows, same
# cost definition as §1656/§1657.
#
# WHAT THIS CAN AND CANNOT SETTLE. It bounds the artifact IN THIS PROTOCOL. §1616 is
# exact restoration on a frozen ship -- a different protocol, and §1656 cost three
# withdrawn board posts for exactly the sin of transferring a curve across protocols.
# So a plateau here does NOT prove §1616 is artifact-free; it establishes whether the
# artifact has a ceiling at all, which is the prior question. If the mechanism is the
# ratio's algebra rather than anything protocol-specific -- as §1657 argued -- then a
# ceiling is the kind of property that plausibly transfers, and its ABSENCE would be
# the more alarming result.
#
# Registered predictions:
#   pred_a THE ARTIFACT IS BOUNDED: the maximum joint/sum across all eight ranks is
#          <= 3.0, far below §1616's 8.9x.
#   pred_b IT PLATEAUS EARLY: the ratios at the four smallest-effect ranks all sit
#          within 0.5 of each other.
#   pred_c IT IS MONOTONE IN EFFECT SIZE: the ratio is non-decreasing as rank rises
#          (and individual effects shrink), allowing at most one inversion for noise.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
SITES = [0, 1, 2, 3]                 # the four EVALUABLE front MLPs (§1326)
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'artifact_ceiling_results.json'
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip1200.pt'
EVAL_ROWS = PT + '.rowcache/fineweb_n192_skip7000.pt'
RECEIPT = PT + '.rowcache/fineweb_oracle_v2_receipt.json'
H = m.transformer.h
TABLE_RATIO_N4 = 1.1530          # §1656, same four sites, same protocol
TABLE_COSTS_N4 = {0: 0.21791, 1: 1.29080, 2: 0.33570, 3: 0.34464}
RANKS = [4, 8, 16, 32, 64, 128, 256, 512]
S1616_RATIO = 8.9               # exact restoration, singleton sum .0573 -- DIFFERENT protocol
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
    print(f'ARTIFACT CEILING (projection, n=4) | sites {SITES} | fit {tuple(fit.shape)} skip1200 | '
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

    ratios = [by_rank[r]['joint_over_sum'] for r in RANKS]
    sums = [by_rank[r]['sum'] for r in RANKS]
    mx = max(ratios)
    tail = ratios[-4:]
    spread_tail = max(tail) - min(tail)
    inversions = sum(1 for i in range(len(ratios) - 1) if ratios[i + 1] < ratios[i] - 1e-9)

    pa = mx <= 3.0
    pb = spread_tail <= 0.5
    pc = inversions <= 1

    print(f'\n  ARTIFACT CURVE across {len(RANKS)} ranks '
          f'(effect sizes {min(sums):.4f} to {max(sums):.4f}, {max(sums)/max(min(sums),1e-9):.0f}x range):',
          flush=True)
    for r in RANKS:
        b = by_rank[r]
        print(f'    rank {r:4d}  sum {b["sum"]:8.4f}  joint/sum {b["joint_over_sum"]:7.4f}', flush=True)
    print(f'  MAXIMUM ratio {mx:.4f}   (§1616 reports {S1616_RATIO} in a DIFFERENT protocol)', flush=True)
    print(f'  tail spread over the four smallest-effect ranks: {spread_tail:.4f}', flush=True)
    print(f'  monotone inversions: {inversions}', flush=True)

    out = {'config': {'sites': SITES, 'fit_rows': 'fineweb_n96_skip1200.pt',
                      'eval_rows': 'fineweb_n192_skip7000.pt',
                      'arm': 'PROJECTION -- rank-r PCA of each MLP output, basis fitted on fit rows',
                      'cost_definition': 'CE(table) - CE(full); no ablation constant involved',
                      'table_arm_costs': TABLE_COSTS_N4, 's541_cross_protocol_context_only': S541_CROSS_PROTOCOL},
           'ce_full': round(ce_full, 5), 'by_rank': by_rank,
           'max_ratio': round(mx, 4), 'tail_spread': round(spread_tail, 4),
           'monotone_inversions': inversions, 's1616_ratio_other_protocol': S1616_RATIO,
           'table_ratio_n4': TABLE_RATIO_N4,
           'predictions': {'pred_a_artifact_bounded_le_3': bool(pa),
                           'pred_b_plateaus_tail_within_05': bool(pb),
                           'pred_c_monotone_in_effect_size': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
