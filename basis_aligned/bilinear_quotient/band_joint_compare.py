# band_joint_compare: THE FRONT/MIDDLE TABLEABILITY CONTRAST, WITH THE PROTOCOLS MATCHED
#
# §1665 found the middle band (mlp4-15) carries a 2.645-nat JOINT stake against twelve
# individual stakes of 0.026-0.106, and that a per-token table recovers only 21.73% of it.
# I contrasted that with the front band's 68-96% and concluded the front is a token
# function and the middle is not.
#
# THAT COMPARISON IS NOT PROTOCOL-MATCHED, and it is the exact error I have flagged twice
# on the board and committed once myself (§1656: an n=4 table ratio against an n=6 block
# figure). The front-band ceilings are PER-SITE -- one module tabled, seventeen live. The
# middle-band ceiling is JOINT over twelve simultaneous substitutions, which is a strictly
# harder problem. A joint ceiling and a per-site ceiling are different quantities and the
# gap between 21.73% and 90% is not evidence until they are the same measurement.
#
# So: joint stake and joint table ceiling for each band, one protocol throughout.
#   front mlp0-3   middle mlp4-15   late mlp16-17   all eighteen
# Every arm uses the §1661 hybrid hook (table only at covered positions, MLP live
# elsewhere) and covered-position scoring, and each band gets an INSTRUMENT CHECK: freeze
# attn0..attn_max(band) at their optimal constants and the joint covered table must be
# exact, ceiling 1.0. The all-eighteen arm is also the strongest available check on the
# whole apparatus, since it freezes every attention module in the model.
#
# §1657's caution applies to the ratios and is honoured here: joint/sum ratios track total
# effect size, so a band's ratio is NOT comparable to another band's measured at a
# different scale. The ratios are reported per band and are not compared across bands. The
# comparison this run makes is between CEILINGS, which are normalised within their own
# condition and do not have that defect.
#
# Registered predictions:
#   pred_a THE CONTRAST SURVIVES PROTOCOL MATCHING: the front band's JOINT table ceiling
#          exceeds the middle band's by >= 30 percentage points. If it does not, §1665's
#          reading was an artifact of comparing per-site against joint and I withdraw it.
#   pred_b SUPERADDITIVITY COMPOUNDS ACROSS BANDS: the all-eighteen joint stake exceeds the
#          sum of the three per-band joint stakes.
#   pred_c INSTRUMENT CHECK ON EVERY BAND: all four frozen-arm ceilings are >= 0.97,
#          including the all-eighteen arm with every attention module in the model frozen.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
BANDS = {'front': list(range(0, 4)), 'middle': list(range(4, 16)),
         'late': list(range(16, 18)), 'all18': list(range(0, 18))}
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'band_joint_compare_results.json'
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip1200.pt'
EVAL_ROWS = PT + '.rowcache/fineweb_n192_skip7000.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
S1665_MIDDLE = {'joint_stake': 2.6453, 'joint_ceiling': 0.2173, 'sum_individual': 0.6208}
S1662_PER_SITE = {'mlp0': 0.90265, 'mlp1': 0.96010, 'mlp2': 0.76980, 'mlp3': 0.67550}
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


@torch.no_grad()
def sweep(rows, K, freeze_upto=None, mlp_hooks=(), score=None):
    hs = []
    if freeze_upto is not None:
        hs += [H[b].attn.register_forward_hook(module_freeze(K[f'attn{b}'].to(DEV).float()))
               for b in range(freeze_upto + 1)]
    hs += list(mlp_hooks)
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
def fit_tables(rows, K, sites, freeze_upto=None):
    s = {L: torch.zeros(50257, D, device=DEV) for L in sites}
    c = torch.zeros(50257, device=DEV)
    fired = {'n': 0}

    def mk(L):
        def hook(mod, args, out):
            t = STATE['idx'].reshape(-1)
            s[L].index_add_(0, t, out.float().reshape(-1, D))
            if L == sites[0]:
                c.index_add_(0, t, torch.ones_like(t, dtype=torch.float32))
                fired['n'] += 1
            return None
        return hook
    sweep(rows, K, freeze_upto,
          mlp_hooks=[H[L].mlp.register_forward_hook(mk(L)) for L in sites])
    assert fired['n'] > 0, 'count hook never fired -- token counts would be all zero'
    seen = c > 0
    tables = {}
    for L in sites:
        mean = s[L].sum(0) / c.sum()
        tbl = mean.unsqueeze(0).repeat(50257, 1)
        tbl[seen] = s[L][seen] / c[seen].unsqueeze(1)
        tables[L] = tbl
    return tables, seen


@torch.no_grad()
def ce(rows, K, seen, mode, sites=(), tables=None, freeze_upto=None):
    hooks = []
    for L in sites:
        if mode == 'const':
            hooks.append(H[L].mlp.register_forward_hook(
                (lambda cst: (lambda mo, a, o: cst.to(o.dtype).expand_as(o)))(
                    K[f'mlp{L}'].to(DEV).float())))
        elif mode == 'table':
            def mk(L):
                def th(mo, a, o):
                    sub = tables[L][STATE['idx'].reshape(-1)].reshape(o.shape).to(o.dtype)
                    return torch.where(seen.to(DEV)[STATE['idx']].unsqueeze(-1), sub, o)
                return th
            hooks.append(H[L].mlp.register_forward_hook(mk(L)))
    acc = {'t': 0.0, 'n': 0}

    def score(lg, tg, idx):
        e = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                            reduction='none').reshape(tg.shape)[:, 64:]
        cov = seen.to(DEV)[idx[:, 64:]]
        acc['t'] += float(e[cov].sum()); acc['n'] += int(cov.sum())
    sweep(rows, K, freeze_upto, mlp_hooks=hooks, score=score)
    return acc['t'] / max(acc['n'], 1)


@torch.no_grad()
def band(fit, ev, K, sites, want_individual):
    tables, seen = fit_tables(fit, K, sites)
    cl = ce(ev, K, seen, 'live')
    cc = ce(ev, K, seen, 'const', sites=sites)
    ct = ce(ev, K, seen, 'table', sites=sites, tables=tables)
    st = cc - cl
    ceil = (cc - ct) / st if st > 1e-6 else float('nan')
    indiv = None
    if want_individual:
        indiv = {f'mlp{L}': round(ce(ev, K, seen, 'const', sites=[L]) - cl, 5) for L in sites}
    up = max(sites)
    tf, sf = fit_tables(fit, K, sites, freeze_upto=up)
    clf = ce(ev, K, sf, 'live', freeze_upto=up)
    ccf = ce(ev, K, sf, 'const', sites=sites, freeze_upto=up)
    ctf = ce(ev, K, sf, 'table', sites=sites, tables=tf, freeze_upto=up)
    stf = ccf - clf
    return {'joint_stake': round(st, 5), 'joint_ceiling': round(ceil, 5),
            'ce_live': round(cl, 5), 'ce_const': round(cc, 5), 'ce_table': round(ct, 5),
            'individual_stakes': indiv,
            'instrument_check': {'known_answer': 1.0,
                                 'observed': round((ccf - ctf) / stf, 5) if stf > 1e-6 else None,
                                 'frozen_arm_stake': round(stf, 5)}}


@torch.no_grad()
def main():
    t0 = time.time()
    fit = load(FIT_ROWS); ev = load(EVAL_ROWS)
    K = torch.load(CONSTS, map_location='cpu')
    print(f'BAND JOINT COMPARE | one protocol for every band: §1661 hybrid substitution, '
          f'covered-position scoring, optimal-constant ablation | fit skip1200, eval skip7000',
          flush=True)

    out = {}
    for name, sites in BANDS.items():
        out[name] = band(fit, ev, K, sites, want_individual=(name != 'all18'))
        r = out[name]
        ss = sum(r['individual_stakes'].values()) if r['individual_stakes'] else None
        r['sum_individual'] = round(ss, 5) if ss is not None else None
        r['joint_over_sum'] = round(r['joint_stake'] / ss, 4) if ss and ss > 1e-9 else None
        print(f'  {name:7s} mlp{sites[0]}-{sites[-1]:<2d} joint stake {r["joint_stake"]:7.4f} | '
              f'JOINT CEILING {r["joint_ceiling"]:7.2%} | instrument '
              f'{r["instrument_check"]["observed"]:.2%}'
              + (f' | joint/sum {r["joint_over_sum"]:.3f}' if r['joint_over_sum'] else ''),
              flush=True)

    gap = out['front']['joint_ceiling'] - out['middle']['joint_ceiling']
    band_sum = sum(out[b]['joint_stake'] for b in ('front', 'middle', 'late'))
    checks = [out[b]['instrument_check']['observed'] for b in BANDS]

    pa = gap >= 0.30
    pb = out['all18']['joint_stake'] > band_sum
    pc = all(v is not None and v >= 0.97 for v in checks)

    print(f'\n  FRONT vs MIDDLE, same protocol: {out["front"]["joint_ceiling"]:.2%} vs '
          f'{out["middle"]["joint_ceiling"]:.2%}  gap {gap:+.2%}  -> contrast survives {pa}',
          flush=True)
    print(f'  (§1665 middle joint ceiling {S1665_MIDDLE["joint_ceiling"]:.2%}; per-SITE front '
          f'ceilings were {[f"{v:.0%}" for v in S1662_PER_SITE.values()]} -- not the comparator)',
          flush=True)
    print(f'  all-18 joint stake {out["all18"]["joint_stake"]:.4f} vs sum of band joints '
          f'{band_sum:.4f}  -> compounds {pb}', flush=True)
    print(f'  instrument checks: {[f"{v:.2%}" for v in checks]}  -> {pc}', flush=True)

    res = {'config': {'bands': BANDS, 'fit_rows': 'fineweb_n96_skip1200.pt',
                      'eval_rows': 'fineweb_n192_skip7000.pt',
                      'substitution': 'HYBRID -- table at covered positions, MLP live elsewhere (§1661)',
                      'scoring': 'covered positions only',
                      'instrument_check': 'freeze attn0..attn_max(band); joint covered table must be exact',
                      's1657_caution': 'joint/sum ratios track total effect size and are NOT compared '
                                       'across bands here; the comparison is between CEILINGS',
                      's1665_middle': S1665_MIDDLE, 's1662_per_site_front': S1662_PER_SITE},
           'bands': out, 'front_minus_middle_ceiling': round(gap, 5),
           'sum_of_band_joint_stakes': round(band_sum, 5),
           'predictions': {'pred_a_contrast_survives_matching_ge_30pts': bool(pa),
                           'pred_b_superadditivity_compounds': bool(pb),
                           'pred_c_instrument_check_all_bands': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
