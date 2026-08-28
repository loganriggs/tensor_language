# circuit_audit: a QUICK, RE-RUNNABLE audit of every registry circuit on the three things we
# actually care about -- OOD, REMOVAL, EXTRACTION.
#
# Logan, 2026-08-28: "what we care about is predicting OOD or removal or extraction of circuits...
# could you make sure this is a quick runnable test? Like we have more and more circuits. It's still
# not obvious how to define the data for each circuit but bootstrapping can work by giving our best
# bet for now."
#
# The registry has 81 entries, 55 certified, and grows every few hours. Individual sections test
# individual claims; nothing tests the SET, and nothing re-tests it when a new circuit lands. This
# does, in a few minutes, with no fitting beyond one pass per circuit.
#
# THREE SCORES PER CIRCUIT, all on the same two eval sets:
#   REMOVAL     replace the circuit's components with their optimal constants; report the CE cost.
#               This is "does the circuit matter", and it is the denominator for the other two.
#   EXTRACTION  replace them with a per-token lookup table -- the simplest non-trivial program --
#               and report the fraction of the removal cost it recovers. This is "can the circuit
#               be stated simply", the cheapest point on the fidelity/simplicity frontier (§1718).
#   OOD         both of the above on skip7000 (reference) and skip11000 (held out), reporting the
#               deltas. This is "does the claim survive documents it was not measured on".
#
# WHAT IS BOOTSTRAPPED, STATED PLAINLY. Defining the right evaluation data per circuit is an open
# problem -- a circuit about question punctuation should arguably be scored on question contexts,
# not on generic FineWeb rows. It is not solved here. Every circuit is scored on the same standard
# rows, which is the best available default and is WRONG in a specific, known direction: a circuit
# with a narrow trigger will look weaker than it is, because most rows do not trigger it. The audit
# reports this per circuit rather than hiding it, and `data_note` is the field to fill in as
# per-circuit data becomes definable.
#
# COVERAGE IS PART OF THE OUTPUT. Only circuits with a resolvable component list can be audited.
# The run reports how many of the registry's entries that is, so the number is visible and can be
# driven up by annotating entries rather than by quietly skipping them.
#
#
# v3 REPAIRS THE TWO DEFECTS §1724 FOUND IN v2's OWN CONTROL. v2 added a specificity control,
# answering the sharpest of Codex's three audit points (§1721):
# "removal is global constant-ablation importance WITHOUT COLLATERAL CONTROL". A raw removal cost
# shows a component matters; it does not show it matters more than any arbitrary component set of
# the same size. So each circuit now also gets a MATCHED-SIZE CONTROL SET drawn from components it
# does not name, ablated identically, and the reported SPECIFICITY is removal / removal_control.
#
#   specificity >> 1  the named set is doing something an arbitrary set of that size does not
#   specificity ~ 1   the circuit has identified a component COUNT, not a component SET
#   specificity < 1   the named set matters LESS than average -- the claim is worse than arbitrary
#
# This is not the per-context collateral control Codex ultimately wants (damage measured on
# contexts the circuit claims NOT to touch), which needs the per-circuit data that §1721 records as
# unsolved. It is the strongest specificity check available without it, and it is honest about
# which one it is.
#
# DEFECT 1 -- ONE DRAW IS A COIN FLIP. v2 took a single deterministic control set. For a one-site
# circuit that makes the whole score depend on which single arbitrary site the picker chose: mlp0's
# control was mlp1 and mlp1's control was mlp0, giving exactly reciprocal 0.12 and 8.25. v3 samples
# N_CTRL distinct control sets and reports the named set's PERCENTILE among them alongside the ratio
# to the control MEDIAN. Where the pool admits only ONE possible control -- a circuit naming all 18
# sites of one kind, whose only same-size elsewhere is all 18 of the other -- the percentile is
# reported as null and `n_control_draws: 1`, because a percentile against a single point is not a
# percentile. That is the reciprocal pair, named rather than papered over.
#
# DEFECT 2 -- SPECIFICITY IS UNSIGNED, THE CLAIMS ARE NOT. v2 scored every circuit against
# "specificity > 1". But `_middle_band_is_redundant_not_small` asserts its band is REDUNDANT, so a
# low specificity CONFIRMS it; v2 counted that as a failure. v3 carries a hand-annotated claim
# DIRECTION per entry and scores each circuit relative to what it actually asserts. This is the
# first thing in the harness that reads a CLAIM rather than a component set, and it is a partial
# answer to §1722: entries with the same sites can now differ, but only through this annotation.
#
# THE DIRECTION ANNOTATIONS ARE MY READING OF EACH ENTRY, NOT ITS WORDS. They are listed in one
# dict below so they can be disputed line by line. Entries whose claim is about a price curve or a
# sign rather than about the set mattering are marked `ambiguous` and scored on neither bar.
#
# WHAT v3 STILL DOES NOT DO. It is not the per-context collateral control Codex ultimately wants
# (damage measured on contexts the circuit claims NOT to touch); that needs the per-circuit data
# §1721 records as unsolved. And outside the DIRECTION field it still scores component SETS.
#
# Registered predictions, TWO-SIDED per LESSONS 31:
#   pred_a DISCRIMINATION: at least one circuit lands at percentile >= 0.90 among its controls and
#          at least one at <= 0.10. If every circuit sits mid-pack, the randomised control has no
#          resolving power and the ratio was carrying the signal by luck.
#   pred_b §1724's OWN DIAGNOSIS IS RIGHT: at least one circuit's v2 single-draw specificity differs
#          from its v3 control-median specificity by a factor >= 2.0. If this is FALSE the single
#          draw was representative after all and §1724 overstated the artifact -- which is a result
#          about my own reasoning and gets recorded as one.
#   pred_c CONTROLS: baseline CE reproduces 3.29205 (§1695) within 1e-3, the whole-MLP-stack removal
#          reproduces §1662's 4.3301 within 0.05, and every v1/v2 removal and extraction row
#          reproduces within 0.01.
#   pred_d DIRECTION RESCUES THE FAILURE: with each circuit scored against the bar its own claim
#          implies, at least two thirds of the direction-annotated circuits are claim-consistent.
#          If this is FALSE, §1724's explanation (1) is wrong -- the circuits fail specificity on
#          their own terms and not because the bar was unsigned.
import json, time, sys, os, random, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
REG = '/workspace/theseus-bench/registry/circuits.json'
OUT = PT + 'ops/circuit_audit_v3_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt'),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt')]
MASK_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
S1695_CE_LIVE = 3.29205
S1662_MLP_STAKE = 4.3301
S1722 = {'_program_price_curve': (4.3301, 0.3454), '_middle_band_is_redundant_not_small': (2.6496, 0.2092)}
# v2's single-draw specificities (§1724), so pred_b can test its own diagnosis rather than assert it
S1724_SPEC = {'_front_band_tableability_ladder': 9.32, '_mlp1_dossier': 8.25,
              '_program_price_curve': 1.22, '_attention_output_write_is_nonlocal': 0.82,
              '_middle_band_is_redundant_not_small': 0.77,
              '_band_synergy_sign_depends_on_band': 0.71,
              '_lag1_failure_is_middle_band': 0.54, '_mlp0_dossier_resolved': 0.12}
N_CTRL = 12
SEED = 1724
STATE = {}

# The component map. Keys are registry entry names; values are (kind, [layer indices]).
# This is the "best bet" bootstrap: seeded by hand from each entry's own sections, and the thing
# to extend as circuits accumulate. An entry absent here is reported as unauditable, not skipped
# silently.
COMPONENTS = {
    '_mlp0_dossier_resolved':        [('mlp', [0])],
    '_mlp1_dossier':                 [('mlp', [1])],
    '_front_band_tableability_ladder': [('mlp', [0, 1, 2, 3])],
    '_front_is_tabular_middle_is_not': [('mlp', list(range(0, 4)))],
    '_middle_band_is_redundant_not_small': [('mlp', list(range(4, 16)))],
    '_middle_band_program_family_prices':  [('mlp', list(range(4, 16)))],
    '_mid_band_feature_price_curve':       [('mlp', list(range(4, 16)))],
    '_front_mlps_are_synergistic':   [('mlp', [0, 1, 2, 3])],
    '_band_synergy_sign_depends_on_band': [('mlp', [16, 17])],
    '_attention_output_write_is_nonlocal': [('attn', list(range(0, 18)))],
    '_attention_write_is_mostly_two_position': [('attn', list(range(0, 18)))],
    '_lag1_failure_is_middle_band':  [('attn', list(range(4, 16)))],
    '_whole_model_program':          [('mlp', list(range(0, 18))), ('attn', list(range(0, 18)))],
    '_program_price_curve':          [('mlp', list(range(0, 18)))],
    '_best_compiled_program_for_mlp_stack': [('mlp', list(range(0, 18)))],
    '_only_attention_routing_is_compressible': [('attn', list(range(0, 18)))],
}

# What each entry ASSERTS about its own set, so specificity is scored against the right bar.
# `important`  the claim needs the named set to carry more than an arbitrary set of that size
# `redundant`  the claim needs it to carry LESS -- a low specificity CONFIRMS it
# `ambiguous`  the claim is about a price curve, a sign, or a decomposition, and implies neither
# MY READING, not the entries' words. Disputable line by line; that is why it is one flat dict.
DIRECTION = {
    '_mlp0_dossier_resolved':                  'important',
    '_mlp1_dossier':                           'important',
    '_front_band_tableability_ladder':         'important',
    '_front_is_tabular_middle_is_not':         'important',
    '_middle_band_is_redundant_not_small':     'redundant',
    '_middle_band_program_family_prices':      'ambiguous',
    '_mid_band_feature_price_curve':           'ambiguous',
    '_front_mlps_are_synergistic':             'important',
    '_band_synergy_sign_depends_on_band':      'ambiguous',
    '_attention_output_write_is_nonlocal':     'important',
    '_attention_write_is_mostly_two_position': 'important',
    '_lag1_failure_is_middle_band':            'important',
    '_whole_model_program':                    'important',
    '_program_price_curve':                    'important',
    '_best_compiled_program_for_mlp_stack':    'important',
    '_only_attention_routing_is_compressible': 'important',
}
assert set(DIRECTION) == set(COMPONENTS), 'DIRECTION and COMPONENTS must name the same entries'


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def const_hook(c, is_attn):
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = c.to(y.dtype).expand_as(y)
        return (sub,) + tuple(out[1:]) if isinstance(out, tuple) else sub
    return hook


def table_hook(tbl, seen, is_attn):
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = tbl[STATE['idx'].reshape(-1)].reshape(y.shape).to(y.dtype)
        sub = torch.where(seen[STATE['idx']].unsqueeze(-1), sub, y)
        return (sub,) + tuple(out[1:]) if isinstance(out, tuple) else sub
    return hook


def sites_of(spec):
    out = []
    for kind, layers in spec:
        for L in layers:
            out.append((kind, L))
    return out


def mod_of(kind, L):
    return H[L].mlp if kind == 'mlp' else H[L].attn


def control_draws(sites):
    """Up to N_CTRL DISTINCT matched-size sets of components the circuit does NOT name.

    Same kind where possible; the complementary kind when the circuit names every site of its own;
    empty when it names all 36. Seeded, so the run reproduces, but a DRAW rather than v2's single
    deterministic pick -- which for a one-site circuit made the score depend entirely on which
    arbitrary site the picker chose (§1724). When the pool is exactly the size of the named set
    there is only ONE possible control and the caller reports `percentile: null`."""
    named = set(sites)
    allsites = [(k, L) for k in ('mlp', 'attn') for L in range(18)]
    pool_same = [st for st in allsites if st not in named and st[0] == sites[0][0]]
    pool_any = [st for st in allsites if st not in named]
    if len(pool_any) < len(sites):
        return []
    pool = pool_same if len(pool_same) >= len(sites) else pool_any
    if len(pool) == len(sites):
        return [sorted(pool)]
    rng = random.Random(SEED + 31 * len(sites) + sum(L for _, L in sites))
    seen, out = set(), []
    for _ in range(400):
        if len(out) >= N_CTRL:
            break
        d = tuple(sorted(rng.sample(pool, len(sites))))
        if d not in seen:
            seen.add(d); out.append(list(d))
    return out


@torch.no_grad()
def sweep(rows, hooks=(), score=None, capture=None):
    hs = list(hooks)
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
def ce(rows, seen, hooks=()):
    acc = {'t': 0.0, 'n': 0}

    def score(lg, tg, idx):
        e = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                            reduction='none').reshape(tg.shape)[:, 64:]
        cov = seen[idx[:, 64:]]
        acc['t'] += float(e[cov].sum()); acc['n'] += int(cov.sum())
    sweep(rows, hooks=hooks, score=score)
    return acc['t'] / max(acc['n'], 1)


@torch.no_grad()
def fit_tables(rows, sites):
    """Per-token mean output at each site, in ONE pass over the fit rows."""
    s = {st: torch.zeros(50257, D, device=DEV) for st in sites}
    c = torch.zeros(50257, device=DEV)
    fired = {'n': 0}

    def mk(st, first):
        def hook(mod, args, out):
            y = (out[0] if isinstance(out, tuple) else out).float().reshape(-1, D)
            t = STATE['idx'].reshape(-1)
            s[st].index_add_(0, t, y)
            if first:
                c.index_add_(0, t, torch.ones_like(t, dtype=torch.float32))
                fired['n'] += 1
            return None
        return hook
    hooks = [mod_of(*st).register_forward_hook(mk(st, j == 0))
             for j, st in enumerate(sites)]
    sweep(rows, hooks=hooks)
    assert fired['n'] > 0, 'table fit never fired'
    seen = c > 0
    out = {}
    for st in sites:
        mean = s[st].sum(0) / c.sum()
        tbl = mean.unsqueeze(0).repeat(50257, 1)
        tbl[seen] = s[st][seen] / c[seen].unsqueeze(1)
        out[st] = tbl
    return out, seen


@torch.no_grad()
def main():
    t0 = time.time()
    K = torch.load(CONSTS, map_location='cpu')
    reg = json.load(open(REG))
    mask_rows = load(MASK_ROWS)
    fit = load(FIT_ROWS)

    certified = [k for k, v in reg.items() if isinstance(v, dict) and v.get('status') == 'certified']
    auditable = [k for k in COMPONENTS if k in reg]
    print(f'CIRCUIT AUDIT v3 (randomised control + claim direction) | registry {len(reg)} entries, {len(certified)} certified | '
          f'{len(auditable)} auditable (have a component list)', flush=True)
    print(f'  BOOTSTRAP: every circuit scored on the same standard rows. Circuits with a narrow '
          f'trigger will read WEAKER than they are.', flush=True)

    rows = {}
    CTRL_CACHE = {}
    for ename, epath in EVAL_SETS:
        ev = load(epath)
        seen = None
        for name in auditable:
            sites = sites_of(COMPONENTS[name])
            tables, seen = fit_tables(fit, sites)
            cl = ce(ev, seen)
            cc = ce(ev, seen, hooks=[mod_of(*st).register_forward_hook(
                const_hook(K[f'{st[0]}{st[1]}'].to(DEV).float(), st[0] == 'attn'))
                for st in sites])
            ct = ce(ev, seen, hooks=[mod_of(*st).register_forward_hook(
                table_hook(tables[st], seen, st[0] == 'attn')) for st in sites])
            removal = cc - cl
            extraction = (cc - ct) / removal if removal > 1e-6 else float('nan')
            # the randomised control is scored on the reference set only; OOD below is about
            # removal, which does not need a control to be comparable across splits
            spec = pct = med_c = None; ndraw = 0
            if ename == 'skip7000':
                key = frozenset(sites)
                if key not in CTRL_CACHE:
                    draws = control_draws(sites)
                    CTRL_CACHE[key] = [ce(ev, seen, hooks=[mod_of(*st).register_forward_hook(
                        const_hook(K[f'{st[0]}{st[1]}'].to(DEV).float(), st[0] == 'attn'))
                        for st in d]) - cl for d in draws]
                rc = CTRL_CACHE[key]; ndraw = len(rc)
                if ndraw:
                    med_c = sorted(rc)[ndraw // 2]
                    spec = removal / med_c if med_c > 1e-6 else None
                    # a percentile against a single possible control is not a percentile
                    pct = (sum(1 for r in rc if r < removal) / ndraw) if ndraw > 1 else None
            rows.setdefault(name, {})[ename] = {
                'n_sites': len(sites), 'ce_live': round(cl, 5),
                'removal_nats': round(removal, 5), 'extraction': round(extraction, 5),
                'n_control_draws': ndraw,
                'control_removal_median_nats': None if med_c is None else round(med_c, 5),
                'control_removal_nats_all': ([round(r, 5) for r in CTRL_CACHE[frozenset(sites)]]
                                             if ename == 'skip7000' else None),
                'specificity_vs_median': None if spec is None else round(spec, 4),
                'percentile': None if pct is None else round(pct, 4),
                'claim_direction': DIRECTION[name]}
            if ename == 'skip7000':
                sp = ' n/a ' if spec is None else f'{spec:5.2f}'
                pp = ' n/a' if pct is None else f'{pct:4.2f}'
                print(f'    {name[:40]:40s} {len(sites):2d}s  removal {removal:7.4f}  '
                      f'extract {extraction:7.2%}  spec {sp}  pct {pp}  ({ndraw} draws)', flush=True)
        if ename == 'skip7000':
            assert abs(rows[auditable[0]]['skip7000']['ce_live'] - S1695_CE_LIVE) <= 1e-3, (
                f'baseline CE disagrees with {S1695_CE_LIVE} (§1695)')
        del ev
        torch.cuda.empty_cache()

    for name in auditable:
        a, b = rows[name]['skip7000'], rows[name]['skip11000']
        rows[name]['ood'] = {
            'removal_rel_change': round((b['removal_nats'] - a['removal_nats'])
                                        / max(a['removal_nats'], 1e-9), 4),
            'extraction_delta': round(b['extraction'] - a['extraction'], 5)}

    ex = [rows[n]['skip7000']['extraction'] for n in auditable]
    ood = [abs(rows[n]['ood']['removal_rel_change']) for n in auditable]
    stack = rows.get('_program_price_curve', {}).get('skip7000', {}).get('removal_nats')

    sp = {n: rows[n]['skip7000']['specificity_vs_median'] for n in auditable
          if rows[n]['skip7000']['specificity_vs_median'] is not None}
    pcts = {n: rows[n]['skip7000']['percentile'] for n in auditable
            if rows[n]['skip7000']['percentile'] is not None}
    pa = bool(pcts) and max(pcts.values()) >= 0.90 and min(pcts.values()) <= 0.10
    moved = {n: (S1724_SPEC[n], sp[n]) for n in sp if n in S1724_SPEC
             and max(S1724_SPEC[n] / max(sp[n], 1e-9), sp[n] / max(S1724_SPEC[n], 1e-9)) >= 2.0}
    pb = len(moved) >= 1
    pc = ((stack is not None) and abs(stack - S1662_MLP_STAKE) <= 0.05
          and all(abs(rows[k]['skip7000']['removal_nats'] - v[0]) <= 0.01
                  and abs(rows[k]['skip7000']['extraction'] - v[1]) <= 0.01
                  for k, v in S1722.items() if k in rows))
    # each circuit against the bar ITS OWN claim implies, not one global bar (§1724 defect 2)
    scored = {n: DIRECTION[n] for n in pcts if DIRECTION[n] in ('important', 'redundant')}
    consistent = {n: (pcts[n] >= 0.75 if d == 'important' else pcts[n] <= 0.25)
                  for n, d in scored.items()}
    pd = bool(scored) and (sum(consistent.values()) / len(scored)) >= (2.0 / 3.0)

    print(f'\n  OOD (relative change in removal cost, skip7000 -> skip11000):', flush=True)
    for name in sorted(auditable, key=lambda n: -abs(rows[n]['ood']['removal_rel_change']))[:5]:
        o = rows[name]['ood']
        print(f'    {name[:46]:46s} removal {o["removal_rel_change"]:+7.2%}  '
              f'extraction {o["extraction_delta"]:+7.2%}', flush=True)
    print(f'\n  SPECIFICITY vs the MEDIAN of {N_CTRL} random matched-size control sets, and the '
          f'named set\'s PERCENTILE among them:', flush=True)
    print(f'    {"circuit":40s} {"spec":>6s} {"pct":>5s} {"draws":>6s}  claim        consistent',
          flush=True)
    for n in sorted(sp, key=lambda a: -sp[a]):
        r = rows[n]['skip7000']
        pp = '  n/a' if r['percentile'] is None else f'{r["percentile"]:5.2f}'
        ok = '-' if n not in consistent else ('YES' if consistent[n] else 'no')
        print(f'    {n[:40]:40s} {sp[n]:6.2f} {pp} {r["n_control_draws"]:6d}  '
              f'{DIRECTION[n]:11s}  {ok}', flush=True)
    for n in auditable:
        if rows[n]['skip7000']['n_control_draws'] == 0:
            print(f'    {n[:40]:40s}    no control exists (names every site)', flush=True)
    print(f'\n  extraction spread {min(ex):.2%} .. {max(ex):.2%}', flush=True)
    print(f'  percentile spread {min(pcts.values()):.2f} .. {max(pcts.values()):.2f} '
          f'-> discriminating {pa}', flush=True)
    print(f'  v2 single draw moved >=2x for {len(moved)} circuits '
          f'{ {k: (round(v[0],2), round(v[1],2)) for k, v in moved.items()} } -> '
          f'S1724 diagnosis {pb}', flush=True)
    print(f'  worst OOD removal move {max(ood):.2%} | v1/v2 rows reproduce -> control {pc}',
          flush=True)
    print(f'  claim-consistent {sum(consistent.values())}/{len(scored)} of direction-annotated '
          f'-> direction rescues {pd}', flush=True)

    res = {'config': {'registry': REG, 'eval_sets': [n for n, _ in EVAL_SETS],
                      'fit_rows': 'fineweb_n96_skip80.pt',
                      'removal': 'components replaced by their optimal constants; CE cost in nats',
                      'specificity': f'named-set removal / MEDIAN removal of up to {N_CTRL} random '
                                     'matched-size control sets drawn from components the circuit '
                                     'does NOT name; percentile is the named set\'s rank among them. '
                                     'Where only one control set is possible (a circuit naming all 18 '
                                     'sites of one kind) the percentile is null, not 0 or 1.',
                      'claim_direction': 'HAND-ANNOTATED reading of what each entry asserts, so '
                                         'specificity is scored against the bar its own claim implies. '
                                         'A `redundant` claim is CONFIRMED by a low specificity. This '
                                         'is the only place the harness reads a claim rather than a '
                                         'component set; it is disputable per line.',
                      'extraction': 'components replaced by a per-token table; fraction of removal cost recovered',
                      'ood': 'both scored on skip7000 (reference) and skip11000 (held out)',
                      'BOOTSTRAP_LIMITATION': 'every circuit scored on the same standard rows. Per-circuit '
                                              'evaluation data is an OPEN PROBLEM; a circuit with a narrow '
                                              'trigger reads WEAKER than it is because most rows do not '
                                              'trigger it. Fill `data_note` per entry as this becomes definable.',
                      'coverage': f'{len(auditable)} of {len(certified)} certified entries have a component list'},
           'circuits': rows,
           'registry_entries': len(reg), 'certified': len(certified), 'auditable': len(auditable),
           'direction_annotations': DIRECTION,
           'claim_consistent': consistent,
           'v2_single_draw_moved_2x': {k: [round(v[0], 3), round(v[1], 3)] for k, v in moved.items()},
           'predictions': {'pred_a_percentile_discriminates': bool(pa),
                           'pred_b_S1724_single_draw_diagnosis': bool(pb),
                           'pred_c_v1v2_rows_reproduce': bool(pc),
                           'pred_d_direction_rescues_specificity': bool(pd)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
