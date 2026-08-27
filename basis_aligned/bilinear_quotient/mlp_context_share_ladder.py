# mlp_context_share_ladder: HOW MUCH OF EACH FRONT MLP IS A TOKEN TABLE, AND HOW MUCH IS
# ATTENTION? — generalising §1661's mlp0 decomposition down the front band.
#
# §1661 established, with a passing instrument check, that mlp0 decomposes cleanly:
#   covered-position table ceiling, attn0 live    90.27%
#   covered-position table ceiling, attn0 frozen 100.00%  (ce_table == ce_live to 5dp)
# i.e. mlp0 is ~90% a current-token lookup, and ALL of the remaining ~10% is context
# that attn0 delivered. mlp0 computes no context of its own.
#
# The obvious question is whether that is special to mlp0 -- which sits directly on the
# embedding -- or whether every front MLP is a token table plus an attention-delivered
# correction, with the correction growing as context accumulates.
#
# METHOD, per site L in {0,1,2,3} (the evaluable front MLPs, §1326):
#   live   condition: model as-is; fit mlp_L's per-token table; measure the ceiling
#   frozen condition: freeze attn_0..attn_L at their optimal constants, refit, measure
# Freezing every attention at or below L makes mlp_L's input a deterministic function of
# the current token (the residual stream below mlp_L is then embedding + constants, and
# MLPs are position-wise), so the frozen ceiling has a KNOWN ANSWER of 1.0 at every site.
# That is the instrument check, and it is what caught the two broken versions of §1661.
#
# CONTEXT SHARE at site L := 1 - (live covered ceiling). The frozen arm certifies that
# this quantity is attributable to attention rather than to a defect of the table.
#
# THE BUG THIS INHERITS THE FIX FOR (§1661, LESSONS 27): the table is substituted ONLY
# at positions whose token was seen at fit time, and mlp_L runs LIVE elsewhere. Excluding
# uncovered positions from the CE average instead is NOT sufficient and is not a
# conservative approximation: a wrong MLP output at an uncovered position propagates
# through the layers above and attention mixes it into the predictions at covered
# positions. That error made §1661 v1 report mlp0's ceiling FALLING 25 points under the
# freeze, the exact opposite of the truth.
#
# HONEST LIMITATION, stated before the numbers: at L=3 the frozen arm has four attention
# modules replaced by constants, which is a large intervention -- the frozen model is
# badly damaged and its CE is high. The ceiling is a ratio computed WITHIN each condition
# so it remains well defined, but the frozen and live arms are not the same model and
# only the live arm's context share is a statement about bilin18 as it runs. Both stakes
# are reported so the size of the damage is visible.
#
# Registered predictions:
#   pred_a THE INSTRUMENT HOLDS AT EVERY DEPTH: all four frozen covered ceilings >= 0.97.
#          If any fails, that site's live reading is not interpretable either.
#   pred_b CONTEXT SHARE GROWS WITH DEPTH: mlp3's context share exceeds mlp0's by >= 5
#          percentage points. The front band accumulates context, so a token table should
#          buy less as you climb.
#   pred_c MANIPULATION CHECK -- every site's stake is >= 0.05 nats in BOTH conditions.
#          A ceiling computed against a stake near zero is a ratio of noise.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
SITES = [0, 1, 2, 3]
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp_context_share_ladder_results.json'
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip1200.pt'
EVAL_ROWS = PT + '.rowcache/fineweb_n192_skip7000.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
S1661_MLP0 = {'live_covered': 0.9027, 'frozen_covered': 1.0}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def freeze_hook(const):
    def hook(mod, args, out):
        if isinstance(out, tuple):
            y = out[0]
            return (const.to(y.dtype).expand_as(y),) + tuple(out[1:])
        return const.to(out.dtype).expand_as(out)
    return hook


def attn_freezes(K, upto):
    """Hooks freezing attn_0..attn_upto at their optimal constants."""
    return [H[b].attn.register_forward_hook(freeze_hook(K[f'attn{b}'].to(DEV).float()))
            for b in range(upto + 1)]


@torch.no_grad()
def run(rows, hooks_fn, mlp_hook=None, score=None):
    """One pass. hooks_fn installs condition hooks; mlp_hook is the site intervention;
    score(logits, targets, idx) accumulates CE, or None to only run the mlp_hook."""
    hs = hooks_fn()
    if mlp_hook is not None:
        hs.append(mlp_hook)
    try:
        for i in range(0, rows.shape[0], 8):
            bb = rows[i:i + 8]
            idx = bb[:, :-1].to(DEV).contiguous()
            STATE['idx'] = idx
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            for blk in H:
                x, v1 = blk(x, v1, x0)
            if score is not None:
                lg = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
                score(lg, bb[:, 1:].to(DEV), idx)
    finally:
        for h in hs:
            h.remove()


STATE = {}


@torch.no_grad()
def fit_table(rows, L, hooks_fn):
    s = torch.zeros(50257, D, device=DEV)
    c = torch.zeros(50257, device=DEV)

    def collect(mod, args, out):
        t = STATE['idx'].reshape(-1)
        s.index_add_(0, t, out.float().reshape(-1, D))
        c.index_add_(0, t, torch.ones_like(t, dtype=torch.float32))
        return None
    run(rows, hooks_fn, mlp_hook=H[L].mlp.register_forward_hook(collect))
    seen = c > 0
    gmean = s.sum(0) / c.sum()
    tbl = gmean.unsqueeze(0).repeat(50257, 1)
    tbl[seen] = s[seen] / c[seen].unsqueeze(1)
    return tbl, seen


@torch.no_grad()
def ce(rows, L, hooks_fn, mode, const_m, tbl=None, seen=None):
    """mode: live | const | table. Scored on COVERED positions only; the table is applied
    ONLY at covered positions and the MLP runs live elsewhere (§1661 / LESSONS 27)."""
    mh = None
    if mode == 'const':
        mh = H[L].mlp.register_forward_hook(
            lambda mo, a, o: const_m.to(o.dtype).expand_as(o))
    elif mode == 'table':
        def th(mo, a, o):
            sub = tbl[STATE['idx'].reshape(-1)].reshape(o.shape).to(o.dtype)
            return torch.where(seen.to(DEV)[STATE['idx']].unsqueeze(-1), sub, o)
        mh = H[L].mlp.register_forward_hook(th)
    acc = {'t': 0.0, 'n': 0}

    def score(lg, tg, idx):
        e = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                            reduction='none').reshape(tg.shape)[:, 64:]
        cov = seen.to(DEV)[idx[:, 64:]]
        acc['t'] += float(e[cov].sum()); acc['n'] += int(cov.sum())
    run(rows, hooks_fn, mlp_hook=mh, score=score)
    return acc['t'] / max(acc['n'], 1)


@torch.no_grad()
def main():
    t0 = time.time()
    fit = load(FIT_ROWS); ev = load(EVAL_ROWS)
    K = torch.load(CONSTS, map_location='cpu')
    print(f'MLP CONTEXT SHARE LADDER | sites {SITES} | covered-position scoring with '
          f'HYBRID substitution (§1661 fix) | fit skip1200, eval skip7000', flush=True)

    out = {}
    for L in SITES:
        const_m = K[f'mlp{L}'].to(DEV).float()
        row = {}
        for cond in ('live', 'frozen'):
            hf = (lambda: []) if cond == 'live' else (lambda L=L: attn_freezes(K, L))
            tbl, seen = fit_table(fit, L, hf)
            cl = ce(ev, L, hf, 'live', const_m, seen=seen)
            cc = ce(ev, L, hf, 'const', const_m, seen=seen)
            ct = ce(ev, L, hf, 'table', const_m, tbl, seen)
            stake = cc - cl
            row[cond] = {'ce_live': round(cl, 5), 'ce_const': round(cc, 5),
                         'ce_table': round(ct, 5), 'stake': round(stake, 5),
                         'ceiling': round((cc - ct) / stake, 5) if stake > 1e-6 else None}
            print(f'  mlp{L} {cond:6s} stake {stake:7.4f} | CE live {cl:.5f} const {cc:.5f} '
                  f'table {ct:.5f} | CEILING {row[cond]["ceiling"]:7.2%}', flush=True)
        row['context_share'] = round(1.0 - row['live']['ceiling'], 5)
        out[f'mlp{L}'] = row
        print(f'    -> context share {row["context_share"]:.2%}', flush=True)

    frozen = [out[f'mlp{L}']['frozen']['ceiling'] for L in SITES]
    shares = [out[f'mlp{L}']['context_share'] for L in SITES]
    stakes = [out[f'mlp{L}'][c]['stake'] for L in SITES for c in ('live', 'frozen')]

    pa = all(f is not None and f >= 0.97 for f in frozen)
    pb = (shares[-1] - shares[0]) >= 0.05
    pc = all(s >= 0.05 for s in stakes)

    print(f'\n  INSTRUMENT CHECK (all frozen ceilings ~1.0): '
          f'{[f"{f:.2%}" for f in frozen]}  -> {pa}', flush=True)
    print(f'  CONTEXT SHARE LADDER: ' +
          '  '.join(f'mlp{L} {s:.2%}' for L, s in zip(SITES, shares)), flush=True)
    print(f'  mlp0 replicates §1661 (live {S1661_MLP0["live_covered"]:.2%}): '
          f'{out["mlp0"]["live"]["ceiling"]:.2%}', flush=True)
    print(f'  min stake across all arms: {min(stakes):.4f} nats', flush=True)

    res = {'config': {'sites': SITES, 'fit_rows': 'fineweb_n96_skip1200.pt',
                      'eval_rows': 'fineweb_n192_skip7000.pt',
                      'substitution': 'HYBRID -- table at covered positions, MLP live elsewhere',
                      'scoring': 'covered positions only',
                      'frozen_arm': 'attn_0..attn_L at optimal constants (known ceiling 1.0)',
                      'ablation_constants': 'opt_ablation_consts_all.pt',
                      's1661_mlp0': S1661_MLP0,
                      'limitation': 'the frozen arm at L=3 replaces four attention modules; '
                                    'it is a damaged model, and only the live arm describes bilin18'},
           'sites': out, 'context_shares': {f'mlp{L}': s for L, s in zip(SITES, shares)},
           'predictions': {'pred_a_instrument_holds_all_sites': bool(pa),
                           'pred_b_context_share_grows_ge_5pts': bool(pb),
                           'pred_c_all_stakes_ge_05_nats': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
