# front_table_compose4: DOES TABLE-ADDITIVITY SURVIVE PAST n=2? — testing a claim I
# just put in the registry myself.
#
# §546 (`front_table_compose`) measured how two table substitutions compose:
#     block 0 table alone                        +0.1666
#     block 1 table alone                        +0.5218
#     both, each fitted against the real model   +0.6654   (sum would be 0.6884)
#     both, block 1 REFITTED with block 0 active +1.0647
# and contrasted it with rank truncation, where six blocks cost 1.6x their sum (§541).
# From that I wrote into `registry/_front_band_account` that "the substitution FAMILY
# determines the composition law: tables ~additive, projections superadditive". That
# generalisation rests on ONE measurement at n=2. Seven of my generalisations died on
# their first independent test today; this one should be tested rather than trusted.
#
# EXTENSION: the same question at n=4, at MLP grain rather than block grain, with one
# index type throughout. Per-token mean tables for mlp0, mlp1, mlp2, mlp3 -- the four
# EVALUABLE front modules (§1326: the other fourteen are below the instrument's floor).
# Individual costs, then all four jointly, each table fitted against the REAL model
# (§546 found refitting against an already-substituted model makes things WORSE, so
# naive fitting is the right arm to extend).
#
# COMPLEMENTARY TO CODEX, NOT DUPLICATIVE: they are testing whether a rank-512
# PROJECTION composes from MLP0 to MLP1 (C512/MLP1 discriminator, 16:32). This tests
# whether TABLES compose across four front sites. Together the two settle the family
# distinction the registry entry asserts.
#
# TWO BUGS FROM THE DISCARDED §1655 RUN ARE FIXED HERE, EXPLICITLY:
#   1. Unseen tokens fall back to the module's POSITION-WEIGHTED mean output, never to
#      a zero vector. §1655's run gave 23.4% of eval positions a zero and thereby
#      penalised the table for a coverage failure rather than a modelling one.
#   2. No mean-ablation constant is computed here at all -- cost is measured directly
#      as CE(table) - CE(full), so the unweighted-mean error that inflated §1655's
#      stake cannot occur.
#
# Registered predictions:
#   pred_a ADDITIVITY SURVIVES AT n=4: joint cost <= 1.15x the sum of the four
#          individual costs.
#   pred_b AND IT IS CLEARLY UNLIKE THE PROJECTION FAMILY: joint/sum < 1.6, the
#          rank-truncation figure §541 measured over six blocks.
#   pred_c MANIPULATION CHECK -- the test is not vacuous: every one of the four
#          individual table substitutions costs at least 0.01 nats. If a module's table
#          is free, its contribution to the sum is noise and the ratio is meaningless.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
SITES = [0, 1, 2, 3]                 # the four EVALUABLE front MLPs (§1326)
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'front_table_compose4_results.json'
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip1200.pt'
EVAL_ROWS = PT + '.rowcache/fineweb_n192_skip7000.pt'
RECEIPT = PT + '.rowcache/fineweb_oracle_v2_receipt.json'
H = m.transformer.h
S546_PAIR = {"b0_alone": 0.1666, "b1_alone": 0.5218, "both": 0.6654, "sum": 0.6884}
S541_PROJECTION_RATIO = 1.6


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


@torch.no_grad()
def fit_tables(rows):
    """Per-token mean of each site's MLP output, plus the POSITION-WEIGHTED global mean
    as the unseen-token fallback (§1655 fix 1)."""
    s = {L: torch.zeros(50257, D, device=DEV) for L in SITES}
    c = {L: torch.zeros(50257, device=DEV) for L in SITES}
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
            t = idx.reshape(-1)
            for L in SITES:
                s[L].index_add_(0, t, cap[L].reshape(-1, D))
                c[L].index_add_(0, t, torch.ones_like(t, dtype=torch.float32))
    finally:
        for h in hs:
            h.remove()
    tables = {}
    for L in SITES:
        seen = c[L] > 0
        gmean = s[L].sum(0) / c[L].sum()          # POSITION-WEIGHTED, the right constant
        tbl = gmean.unsqueeze(0).repeat(50257, 1)  # unseen tokens -> weighted mean, NOT zero
        tbl[seen] = s[L][seen] / c[L][seen].unsqueeze(1)
        tables[L] = tbl
    return tables


@torch.no_grad()
def ce_with_tables(rows, tables, active):
    """CE with the given sites' MLP outputs replaced by their token tables."""
    hs = []
    state = {}
    for L in active:
        def mk(L):
            def hook(mod, args, out):
                return tables[L][state['idx'].reshape(-1)].reshape(out.shape).to(out.dtype)
            return hook
        hs.append(H[L].mlp.register_forward_hook(mk(L)))
    tot, n = 0.0, 0
    try:
        for i in range(0, rows.shape[0], 8):
            bb = rows[i:i + 8]
            idx = bb[:, :-1].to(DEV).contiguous(); tg = bb[:, 1:].to(DEV)
            state['idx'] = idx
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
    print(f'FRONT TABLE COMPOSE n=4 | sites {SITES} | fit {tuple(fit.shape)} skip1200 | '
          f'eval {tuple(ev.shape)} skip7000 | receipt {rh}', flush=True)

    tables = fit_tables(fit)
    ce_full = ce_with_tables(ev, tables, [])
    print(f'  CE full (no substitution) {ce_full:.5f}', flush=True)

    individual = {}
    for L in SITES:
        ce_L = ce_with_tables(ev, tables, [L])
        individual[L] = ce_L - ce_full
        print(f'  mlp{L} table alone: CE {ce_L:.5f}  cost {individual[L]:+.5f}', flush=True)

    ce_joint = ce_with_tables(ev, tables, SITES)
    joint = ce_joint - ce_full
    ssum = sum(individual.values())
    ratio = joint / ssum if ssum > 0 else float('inf')

    pa = ratio <= 1.15
    pb = ratio < S541_PROJECTION_RATIO
    pc = all(v >= 0.01 for v in individual.values())

    print(f'\n  ALL FOUR jointly: CE {ce_joint:.5f}  cost {joint:+.5f}', flush=True)
    print(f'  sum of individual costs: {ssum:+.5f}', flush=True)
    print(f'  JOINT / SUM = {ratio:.4f}   (§546 at n=2: {S546_PAIR["both"]/S546_PAIR["sum"]:.4f}; '
          f'projections §541: {S541_PROJECTION_RATIO})', flush=True)
    print(f'  manipulation check -- all four individual costs >= .01 nats: {pc}', flush=True)

    out = {'config': {'sites': SITES, 'fit_rows': 'fineweb_n96_skip1200.pt',
                      'eval_rows': 'fineweb_n192_skip7000.pt',
                      'unseen_token_fallback': 'POSITION-WEIGHTED global mean (§1655 fix)',
                      'cost_definition': 'CE(table) - CE(full); no ablation constant involved',
                      's546_pair': S546_PAIR, 's541_projection_ratio': S541_PROJECTION_RATIO},
           'ce_full': round(ce_full, 5), 'ce_joint': round(ce_joint, 5),
           'individual_costs': {f'mlp{L}': round(v, 5) for L, v in individual.items()},
           'joint_cost': round(joint, 5), 'sum_of_individual': round(ssum, 5),
           'joint_over_sum': round(ratio, 4),
           'predictions': {'pred_a_additive_within_15pct': bool(pa),
                           'pred_b_unlike_projection_family': bool(pb),
                           'pred_c_manipulation_all_ge_01': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
