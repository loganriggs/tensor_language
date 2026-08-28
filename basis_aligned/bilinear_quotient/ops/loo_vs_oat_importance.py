# LEAVE-ONE-OUT vs ONE-AT-A-TIME SITE IMPORTANCE -- do they even rank the sites the same way?
#
# §1736 certified that the two stacks compose in opposite directions: the sum of the 18 one-at-a-time
# (OAT) constant-ablation removals is 2.36x the joint MLP removal and 0.40x the joint attention
# removal. Every single-site importance number in this arc is an OAT number, so they are inflated for
# MLP sites and deflated for attention sites relative to the stack.
#
# A SCALE factor would be annoying. A RANKING error would be worse, and nothing has checked for one.
# The natural alternative measure is LEAVE-ONE-OUT: how much MORE the stack costs when site i is
# ablated too, given the other seventeen already are.
#
#     OAT_i = removal({i})                          what one site costs on its own
#     LOO_i = removal(stack) - removal(stack \ {i})  what one site adds once the rest are gone
#
# For a redundant stack these pull apart hard: a site whose work seventeen others can absorb has a
# LARGE OAT relative to its LOO... or the reverse, depending on whether the redundancy is in the
# source or the sink. Which way it goes is the measurement, not something to assert in a comment.
#
# ROLES. This is the same family as §1736, whose confirmation role was skip11000, so BOTH large roles
# are spent here. DISCOVERY ONLY: this run confirms nothing and certifies nothing. Its product is a
# frozen ranked list and a frozen prediction for whichever clean role appears next. Running a
# same-family question on burned rows and calling it held-out is the error Codex retired two of my
# scripts for, and the answer is to label it, not to skip the measurement.
#
# Registered predictions, TWO-SIDED per LESSONS 31:
#   pred_a THE TWO MEASURES RANK SITES DIFFERENTLY: the Spearman correlation between OAT and LOO
#          across all 36 sites is below 0.8 on both roles. If it is HIGH, §1736's factor is a scale
#          problem only, every existing ranking survives, and that is the more reassuring result --
#          which is why the bar is set where a pass means my existing tables are WRONG.
#   pred_b THE STACK-LEVEL ASYMMETRY MIRRORS: sum of LOO over joint is BELOW 1 for the MLP stack and
#          ABOVE 1 for the attention stack -- the mirror of §1736's 2.36 and 0.40. If FALSE the two
#          measures are not two views of one redundancy structure and the reading of §1736 is too
#          simple.
#   pred_c CONTROLS: the joint stack removals reproduce §1662/§1682's 4.3301 and 3.5570 within 0.01,
#          the OAT column reproduces §1736's per-site removals within 0.001, and every
#          `stack minus one site` removal lies between 0 and the joint removal plus 0.5 nats.
#   pred_d THE DISAGREEMENT TRACKS THE REDUNDANCY: the median |OAT - LOO| gap is larger for the 18
#          MLP sites than for the 18 attention sites. If FALSE, the disagreement is not explained by
#          which stack is redundant and needs another account.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/loo_vs_oat_importance_results.json'
S1736 = PT + 'ops/site_additivity_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
S1662_JOINT = {'mlp': 4.3301, 'attn': 3.5570}
COV = {}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def const_hook(c):
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = c.to(y.dtype).expand_as(y)
        return (sub,) + tuple(out[1:]) if isinstance(out, tuple) else sub
    return hook


def mod_of(kind, L):
    return H[L].mlp if kind == 'mlp' else H[L].attn


@torch.no_grad()
def ce(rows, hooks=()):
    """float64 accumulation: these removals are small differences of ~95,000-nat totals (§1736)."""
    tot = 0.0; cnt = 0
    hs = list(hooks)
    try:
        for i in range(0, rows.shape[0], 8):
            bb = rows[i:i + 8]
            idx = bb[:, :-1].to(DEV).contiguous()
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            for blk in H:
                x, v1 = blk(x, v1, x0)
            lg = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
            tg = bb[:, 1:].to(DEV)
            e = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                                reduction='none').reshape(tg.shape)[:, 64:].double()
            cov = COV['seen'][idx[:, 64:]]
            tot += float(e[cov].sum()); cnt += int(cov.sum())
    finally:
        for h in hs:
            h.remove()
    return tot / cnt


def spearman(a, b):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((x - mb) ** 2 for x in rb) ** 0.5
    return num / (da * db) if da * db else float('nan')


@torch.no_grad()
def main():
    t0 = time.time()
    K = torch.load(CONSTS, map_location='cpu')
    fit = load(FIT_ROWS)
    seen = torch.zeros(50257, dtype=torch.bool)
    seen[fit[:, :T].reshape(-1).long()] = True
    COV['seen'] = seen.to(DEV)
    prior = json.load(open(S1736))['results']
    sites = [(k, L) for k in ('mlp', 'attn') for L in range(18)]
    stacks = {'mlp': [s for s in sites if s[0] == 'mlp'],
              'attn': [s for s in sites if s[0] == 'attn']}

    def hooks_for(sl):
        return [mod_of(*s).register_forward_hook(
            const_hook(K[f'{s[0]}{s[1]}'].to(DEV).float())) for s in sl]

    out = {}
    print('LOO vs OAT SITE IMPORTANCE | DISCOVERY ONLY -- both large roles are spent for this '
          'family (§1736). Confirms nothing.', flush=True)

    for ename, epath, ce_ref in EVAL_SETS:
        ev = load(epath)
        cl = ce(ev)
        assert abs(cl - ce_ref) <= 1e-2, f'{ename} baseline CE {cl:.5f} != {ce_ref}'
        jnt = {kind: ce(ev, hooks_for(sl)) - cl for kind, sl in stacks.items()}
        rows, sane = {}, True
        for kind, sl in stacks.items():
            for st in sl:
                oat = ce(ev, hooks_for([st])) - cl
                minus = ce(ev, hooks_for([s for s in sl if s != st])) - cl
                sane = sane and (0.0 <= minus <= jnt[kind] + 0.5)
                rows[f'{st[0]}{st[1]}'] = {'oat': round(oat, 5), 'loo': round(jnt[kind] - minus, 5),
                                           'stack_minus_site': round(minus, 5), 'kind': kind}
        sums = {kind: {'oat': round(sum(rows[f'{s[0]}{s[1]}']['oat'] for s in sl), 5),
                       'loo': round(sum(rows[f'{s[0]}{s[1]}']['loo'] for s in sl), 5),
                       'joint': round(jnt[kind], 5)} for kind, sl in stacks.items()}
        names = list(rows)
        rho = spearman([rows[n]['oat'] for n in names], [rows[n]['loo'] for n in names])
        gapm = {kind: sorted(abs(rows[f'{s[0]}{s[1]}']['oat'] - rows[f'{s[0]}{s[1]}']['loo'])
                             for s in sl)[9] for kind, sl in stacks.items()}
        print(f'\n  {ename} [discovery]: baseline CE {cl:.5f} | joint mlp {jnt["mlp"]:.4f} attn '
              f'{jnt["attn"]:.4f}', flush=True)
        for kind in stacks:
            s = sums[kind]
            print(f'    {kind:4s}  sum OAT {s["oat"]:8.4f} ({s["oat"]/s["joint"]:5.3f}x joint)  '
                  f'sum LOO {s["loo"]:8.4f} ({s["loo"]/s["joint"]:5.3f}x joint)  '
                  f'median |OAT-LOO| {gapm[kind]:.4f}', flush=True)
        print(f'    Spearman(OAT, LOO) over 36 sites: {rho:.4f}', flush=True)
        top_o = sorted(names, key=lambda n: -rows[n]['oat'])[:5]
        top_l = sorted(names, key=lambda n: -rows[n]['loo'])[:5]
        print(f'    top 5 by OAT: {top_o}', flush=True)
        print(f'    top 5 by LOO: {top_l}', flush=True)
        out[ename] = {'baseline_ce': round(cl, 5), 'joint': {k: round(v, 5) for k, v in jnt.items()},
                      'sites': rows, 'stack_sums': sums, 'spearman_oat_loo': round(rho, 5),
                      'median_abs_gap': {k: round(v, 5) for k, v in gapm.items()},
                      'top5_oat': top_o, 'top5_loo': top_l, 'sane': bool(sane)}
        del ev
        torch.cuda.empty_cache()

    pa = all(out[e]['spearman_oat_loo'] < 0.8 for e in out)
    pb = all(out[e]['stack_sums']['mlp']['loo'] < out[e]['stack_sums']['mlp']['joint']
             and out[e]['stack_sums']['attn']['loo'] > out[e]['stack_sums']['attn']['joint']
             for e in out)
    oat_ok = all(abs(out[e]['sites'][n]['oat'] - prior[e]['per_site_removal_cov'][n]) <= 1e-3
                 for e in out for n in out[e]['sites'])
    pc = (all(abs(out['skip7000']['joint'][k] - v) <= 0.01 for k, v in S1662_JOINT.items())
          and oat_ok and all(out[e]['sane'] for e in out))
    pd = all(out[e]['median_abs_gap']['mlp'] > out[e]['median_abs_gap']['attn'] for e in out)

    frozen = sorted(out['skip7000']['sites'],
                    key=lambda n: -min(out['skip7000']['sites'][n]['loo'],
                                       out['skip11000']['sites'][n]['loo']))[:6]
    print(f'\n  Spearman below 0.8 on both roles -> the two measures rank differently {pa}',
          flush=True)
    print(f'  sum LOO below joint for MLPs and above for attention -> mirrors §1736 {pb}', flush=True)
    print(f'  §1662 joints + §1736 OAT column + sanity bounds -> control {pc}', flush=True)
    print(f'  median |OAT-LOO| larger for MLPs -> tracks redundancy {pd}', flush=True)
    print(f'\n  FROZEN: top 6 sites by the WORSE of the two roles\' LOO, for the next clean role: '
          f'{frozen}', flush=True)

    r = {'config': {'eval_sets': [e[0] for e in EVAL_SETS],
                    'OAT': 'removal({i}) -- what one site costs on its own',
                    'LOO': 'removal(stack) - removal(stack minus {i}) -- what one site adds once '
                           'the other seventeen are already ablated',
                    'ROLE_NOTE': 'DISCOVERY ONLY. Same family as §1736, whose confirmation role was '
                                 'skip11000, so both large roles are spent. Confirms nothing, '
                                 'certifies nothing; the product is the frozen list above.',
                    'float64': 'CE accumulated in float64 (§1736)'},
         'results': out, 'frozen_for_next_clean_role': frozen,
         'predictions': {'pred_a_measures_rank_differently': bool(pa),
                         'pred_b_stack_asymmetry_mirrors': bool(pb),
                         'pred_c_controls': bool(pc),
                         'pred_d_gap_tracks_redundancy': bool(pd)},
         'runtime_s': round(time.time() - t0, 1)}
    json.dump(r, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
