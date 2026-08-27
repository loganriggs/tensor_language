# mlp1_residue_attribution: WHICH ATTENTION DELIVERS MLP1'S UN-TABLEABLE RESIDUE? — the
# first question in this thread whose answer is NOT forced by the architecture.
#
# §1662's ladder gives the front band's live covered-position table ceilings:
#     mlp0 90.27%  mlp1 96.01%  mlp2 76.98%  mlp3 67.55%
# and mlp1 is the anomaly twice over. It is the most TABLEABLE front module despite being
# deeper than mlp0, and its stake is 7.005 nats against 0.86/0.77/0.62 for its
# neighbours -- eight times the size. In absolute terms its 3.99% residue is 0.279 nats,
# the LARGEST un-tableable quantity in the front band.
#
# §1662 also retracted the framing of §1661. That a module with all its attention inputs
# frozen becomes a token function is architecturally forced -- attention is the only thing
# in a transformer that moves information across positions -- so the frozen arm is an
# instrument check, not an attribution. Saying "the residue is attention" states the
# architecture back.
#
# THIS is the attribution question with a real answer: mlp1 sits above eighteen attention
# heads (attn0.0-8, attn1.0-8). Is its 0.279-nat residue spread evenly across them, or
# delivered by a few? Nothing about the architecture fixes that.
#
# METHOD. For each condition, freeze the named component(s) at their optimal constants
# from opt_ablation_consts_all.pt, REFIT mlp1's per-token table under that condition, and
# measure the covered-position ceiling with the §1661 hybrid hook (table only where the
# token was seen at fit; mlp1 LIVE elsewhere). Head freezing follows the established
# pattern: a pre-hook on attn.c_proj replacing that head's 128-dim slice (circuit_holdout).
#   conditions: live | attn0 | attn1 | attn0+attn1 | each of 18 heads
# ATTRIBUTION of a condition := (ceiling_condition - ceiling_live) / (1 - ceiling_live),
# the fraction of the live residue that freezing that component removes.
#
# Freezing attn0+attn1 leaves mlp1 token-deterministic, so its ceiling has a KNOWN ANSWER
# of 1.0 and attribution 1.0. That is the instrument check; the two versions it caught in
# §1659 are the reason every run in this thread carries one.
#
# HONEST LIMITATION: each freeze changes the model, so the refitted stakes differ across
# conditions and the attributions are not guaranteed to sum to 1. Their sum is reported
# precisely so that the departure is visible rather than hidden by normalising it away.
#
# Registered predictions:
#   pred_a INSTRUMENT CHECK: freezing attn0+attn1 gives a ceiling >= 0.97. If it fails
#          nothing else in the run is interpretable.
#   pred_b THE RESIDUE IS CONCENTRATED, NOT DIFFUSE: the single best head accounts for
#          >= 33% of the live residue. Uniform spread over eighteen heads would be 5.6%.
#   pred_c MANIPULATION CHECK -- head grain is above the instrument's floor: at least one
#          head moves mlp1's ceiling by >= 0.5 percentage points. If no single head does,
#          per-head attribution is not measurable here whatever the ranking looks like.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; SITE = 1; NH = 9; HD = 128
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp1_residue_attribution_results.json'
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip1200.pt'
EVAL_ROWS = PT + '.rowcache/fineweb_n192_skip7000.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
S1662_LADDER = {'mlp0': 0.90265, 'mlp1': 0.96010, 'mlp2': 0.76980, 'mlp3': 0.67550}
STATE = {}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def module_freeze(const):
    def hook(mod, args, out):
        if isinstance(out, tuple):
            y = out[0]
            return (const.to(y.dtype).expand_as(y),) + tuple(out[1:])
        return const.to(out.dtype).expand_as(out)
    return hook


def head_freeze(L, heads, K):
    """Pre-hook on attn.c_proj replacing each named head's 128-dim slice (circuit_holdout)."""
    def hook(mod, args):
        x = args[0].clone()
        for hh in heads:
            x[:, :, hh * HD:(hh + 1) * HD] = \
                K[f'head{L}.{hh}'].to(DEV).float().to(x.dtype)
        return (x,)
    return hook


def make_hooks(spec, K):
    """spec: [] | ['attn0'] | ['attn0','attn1'] | [('head', L, h)]"""
    hs = []
    for s in spec:
        if isinstance(s, tuple):
            hs.append(H[s[1]].attn.c_proj.register_forward_pre_hook(
                head_freeze(s[1], [s[2]], K)))
        else:
            hs.append(H[int(s[4:])].attn.register_forward_hook(
                module_freeze(K[s].to(DEV).float())))
    return hs


@torch.no_grad()
def sweep(rows, spec, K, mlp_hook=None, score=None):
    hs = make_hooks(spec, K)
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


@torch.no_grad()
def fit_table(rows, spec, K):
    s = torch.zeros(50257, D, device=DEV)
    c = torch.zeros(50257, device=DEV)

    def collect(mod, args, out):
        t = STATE['idx'].reshape(-1)
        s.index_add_(0, t, out.float().reshape(-1, D))
        c.index_add_(0, t, torch.ones_like(t, dtype=torch.float32))
        return None
    sweep(rows, spec, K, mlp_hook=H[SITE].mlp.register_forward_hook(collect))
    seen = c > 0
    tbl = (s.sum(0) / c.sum()).unsqueeze(0).repeat(50257, 1)
    tbl[seen] = s[seen] / c[seen].unsqueeze(1)
    return tbl, seen


@torch.no_grad()
def ce(rows, spec, K, mode, const_m, tbl=None, seen=None):
    mh = None
    if mode == 'const':
        mh = H[SITE].mlp.register_forward_hook(
            lambda mo, a, o: const_m.to(o.dtype).expand_as(o))
    elif mode == 'table':
        def th(mo, a, o):
            sub = tbl[STATE['idx'].reshape(-1)].reshape(o.shape).to(o.dtype)
            return torch.where(seen.to(DEV)[STATE['idx']].unsqueeze(-1), sub, o)
        mh = H[SITE].mlp.register_forward_hook(th)
    acc = {'t': 0.0, 'n': 0}

    def score(lg, tg, idx):
        e = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                            reduction='none').reshape(tg.shape)[:, 64:]
        cov = seen.to(DEV)[idx[:, 64:]]
        acc['t'] += float(e[cov].sum()); acc['n'] += int(cov.sum())
    sweep(rows, spec, K, mlp_hook=mh, score=score)
    return acc['t'] / max(acc['n'], 1)


@torch.no_grad()
def ceiling(fit, ev, spec, K, const_m):
    tbl, seen = fit_table(fit, spec, K)
    cl = ce(ev, spec, K, 'live', const_m, seen=seen)
    cc = ce(ev, spec, K, 'const', const_m, seen=seen)
    ct = ce(ev, spec, K, 'table', const_m, tbl, seen)
    st = cc - cl
    return ((cc - ct) / st if st > 1e-6 else float('nan')), st


@torch.no_grad()
def main():
    t0 = time.time()
    fit = load(FIT_ROWS); ev = load(EVAL_ROWS)
    K = torch.load(CONSTS, map_location='cpu')
    const_m = K[f'mlp{SITE}'].to(DEV).float()
    print(f'MLP{SITE} RESIDUE ATTRIBUTION | hybrid substitution, covered-position scoring '
          f'(§1661) | fit skip1200, eval skip7000', flush=True)

    base, base_st = ceiling(fit, ev, [], K, const_m)
    resid = 1.0 - base
    print(f'  live: ceiling {base:.2%}  stake {base_st:.4f} nats  -> residue {resid:.2%} '
          f'= {resid*base_st:.4f} nats  (§1662 ladder: {S1662_LADDER["mlp1"]:.2%})', flush=True)

    def attribution(c):
        return (c - base) / resid if resid > 1e-9 else float('nan')

    mods = {}
    for spec, name in (([f'attn{0}'], 'attn0'), (['attn1'], 'attn1'),
                       (['attn0', 'attn1'], 'attn0+attn1')):
        c, st = ceiling(fit, ev, spec, K, const_m)
        mods[name] = {'ceiling': round(c, 5), 'stake': round(st, 5),
                      'attribution': round(attribution(c), 4)}
        print(f'  freeze {name:12s} ceiling {c:7.2%}  stake {st:7.4f} | '
              f'attribution {attribution(c):7.2%}', flush=True)

    heads = {}
    for L in (0, 1):
        for hh in range(NH):
            c, st = ceiling(fit, ev, [('head', L, hh)], K, const_m)
            heads[f'head{L}.{hh}'] = {'ceiling': round(c, 5), 'stake': round(st, 5),
                                      'attribution': round(attribution(c), 4)}
    rank = sorted(heads.items(), key=lambda kv: -kv[1]['attribution'])
    print('\n  PER-HEAD (top 6 by attribution):', flush=True)
    for k, v in rank[:6]:
        print(f'    {k:9s} ceiling {v["ceiling"]:7.2%}  attribution {v["attribution"]:7.2%}',
              flush=True)
    print(f'    ... worst: {rank[-1][0]} attribution {rank[-1][1]["attribution"]:.2%}', flush=True)

    best = rank[0][1]['attribution']
    max_shift = max(abs(v['ceiling'] - base) for v in heads.values())
    hsum = sum(v['attribution'] for v in heads.values())

    pa = mods['attn0+attn1']['ceiling'] >= 0.97
    pb = best >= 0.33
    pc = max_shift >= 0.005

    print(f'\n  INSTRUMENT CHECK (attn0+attn1 frozen, known answer 1.0): '
          f'{mods["attn0+attn1"]["ceiling"]:.2%} -> {pa}', flush=True)
    print(f'  best single head {rank[0][0]} accounts for {best:.2%} of the residue '
          f'(uniform over 18 heads would be {1/18:.2%})', flush=True)
    print(f'  sum of 18 head attributions: {hsum:.2%}  (not constrained to 100% -- each '
          f'freeze is a different model)', flush=True)
    print(f'  largest single-head ceiling shift: {max_shift:.2%}', flush=True)

    res = {'config': {'site': SITE, 'fit_rows': 'fineweb_n96_skip1200.pt',
                      'eval_rows': 'fineweb_n192_skip7000.pt',
                      'substitution': 'HYBRID -- table at covered positions, mlp live elsewhere',
                      'scoring': 'covered positions only',
                      'head_freeze': 'pre-hook on attn.c_proj, 128-dim slice (circuit_holdout pattern)',
                      'attribution_def': '(ceiling_frozen - ceiling_live) / (1 - ceiling_live)',
                      's1662_ladder': S1662_LADDER,
                      'limitation': 'each freeze is a different model; refitted stakes differ '
                                    'and attributions are not constrained to sum to 1'},
           'live': {'ceiling': round(base, 5), 'stake': round(base_st, 5),
                    'residue_frac': round(resid, 5), 'residue_nats': round(resid * base_st, 5)},
           'modules': mods, 'heads': heads,
           'head_ranking': [k for k, _ in rank],
           'head_attribution_sum': round(hsum, 4),
           'max_single_head_ceiling_shift': round(max_shift, 5),
           'predictions': {'pred_a_instrument_check_ge_097': bool(pa),
                           'pred_b_residue_concentrated_best_head_ge_33': bool(pb),
                           'pred_c_head_grain_above_floor': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
