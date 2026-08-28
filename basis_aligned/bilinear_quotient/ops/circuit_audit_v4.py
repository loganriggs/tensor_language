# circuit_audit v4 -- WHAT KIND OF TOKEN does each circuit's damage land on?
#
# v1 gave every circuit a removal cost and a per-token-table extraction score. v2 added a
# specificity control; v3 (§1725) fixed that control by randomising it, which reversed two of v2's
# headline rows and left one clean negative: `_lag1_failure_is_middle_band` sits at percentile 0.00
# among matched controls -- the CHEAPEST twelve sites to ablate -- while its claim is about lag-1
# copying failing there. That is not a refutation of the entry. It is a demonstration that GLOBAL
# REMOVAL IS THE WRONG ESTIMAND FOR IT, which is Codex's standing point: an `important/redundant`
# annotation picks the bar, it does not change what is being measured.
#
# v4 changes what is measured. Every circuit's removal cost is DECOMPOSED over target-side token
# classes, computed inside the same forwards, so this costs no extra GPU time:
#
#   induction  the target token appears earlier in the context AND is preceded there by the current
#              token -- the strict copy/induction case, where a routing mechanism can win
#   repeat     the target appears earlier in the context but not in an induction position -- it is
#              retrievable from context without the strict pattern
#   novel      the target has not appeared in the context at all -- nothing to copy; the answer has
#              to come from the weights
#
# The three are DISJOINT AND EXHAUSTIVE over scored positions, and the run asserts their counts sum
# to the total, so nothing is being quietly dropped into an unscored fourth bucket.
#
# WHY THIS IS THE COLLATERAL CONTROL, AND WHERE IT STOPS. §1721 recorded Codex's ask: damage
# measured on contexts a circuit claims NOT to touch. For any circuit annotated with a claimed class,
# v4 reports SELECTIVITY -- per-token removal on the claimed class over per-token removal on its
# complement. That is the first per-context collateral number in this arc. It stops well short of
# the general case: only claims whose context is a target-token property can be expressed this way,
# so the three attention/lag entries get a real class and the rest are annotated `all` and get a
# profile but no selectivity. Marked per entry rather than assumed.
#
# The bootstrap limitation of v1 is now PARTLY addressed rather than only declared: a circuit with a
# narrow trigger no longer has to read weaker than it is, PROVIDED its trigger is one of these three
# classes. Logan, 2026-08-28: "it's still not obvious how to define the data for each circuit but
# bootstrapping can work by giving our best bet for now." This is the best bet, made explicit.
#
# Registered predictions, TWO-SIDED per LESSONS 31:
#   pred_a THE PROFILE DISCRIMINATES: two circuits' shares of damage falling on `induction` targets
#          differ by >= 10 percentage points. If every circuit damages the same mix of token
#          classes, the decomposition adds an axis with no resolving power and should be dropped.
#   pred_b ATTENTION IS COPY-CONCENTRATED: the 18-attention stack's induction/novel per-token
#          removal RATIO exceeds the 18-MLP stack's. §1707 found attention routes content rather
#          than positions; if the effect is real it should be visible target-side. If FALSE,
#          attention's damage is spread like the MLPs' and the routing story does not show up here.
#   pred_c CONTROLS: baseline CE reproduces 3.29205 (§1695) within 1e-3, total removal reproduces
#          §1722/§1725 rows within 0.01, and the three class counts sum EXACTLY to the scored count.
#   pred_d A PER-CONTEXT ESTIMAND RESCUES THE LAG-1 ENTRY: `_lag1_failure_is_middle_band` damages
#          induction targets more per token than novel targets, by >= 10% relative. This is the
#          direct test of §1725's reading. If FALSE, the entry's percentile-0.00 result stands as a
#          problem for the entry and not only for the estimand, and gets recorded as one.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
REG = '/workspace/theseus-bench/registry/circuits.json'
OUT = PT + 'ops/circuit_audit_v4_results.json'
EVAL = ('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt')
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
S1695_CE_LIVE = 3.29205
S1725 = {'_program_price_curve': 4.3301, '_middle_band_is_redundant_not_small': 2.6496,
         '_mlp1_dossier': 7.0213, '_mlp0_dossier_resolved': 0.8513,
         '_front_band_tableability_ladder': 4.3928, '_attention_output_write_is_nonlocal': 3.5570,
         '_lag1_failure_is_middle_band': 2.1137, '_whole_model_program': 5.5684,
         '_band_synergy_sign_depends_on_band': 0.7112}
CLASSES = ('induction', 'repeat', 'novel')
STATE = {}

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

# The token class each entry's claim is ABOUT, where its claim names one. `all` means the claim is
# not about a context and gets a profile but no selectivity number. MY READING of each entry, kept
# in one flat dict so it can be disputed line by line, same as v3's DIRECTION.
CLAIM_CLASS = {
    '_lag1_failure_is_middle_band':            'induction',
    '_attention_write_is_mostly_two_position': 'induction',
    '_attention_output_write_is_nonlocal':     'repeat',
    '_mlp0_dossier_resolved':                  'all',
    '_mlp1_dossier':                           'all',
    '_front_band_tableability_ladder':         'all',
    '_front_is_tabular_middle_is_not':         'all',
    '_middle_band_is_redundant_not_small':     'all',
    '_middle_band_program_family_prices':      'all',
    '_mid_band_feature_price_curve':           'all',
    '_front_mlps_are_synergistic':             'all',
    '_band_synergy_sign_depends_on_band':      'all',
    '_whole_model_program':                    'all',
    '_program_price_curve':                    'all',
    '_best_compiled_program_for_mlp_stack':    'all',
    '_only_attention_routing_is_compressible': 'all',
}
assert set(CLAIM_CLASS) == set(COMPONENTS), 'CLAIM_CLASS and COMPONENTS must name the same entries'


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
    return [(kind, L) for kind, layers in spec for L in layers]


def mod_of(kind, L):
    return H[L].mlp if kind == 'mlp' else H[L].attn


@torch.no_grad()
def token_classes(idx, tg):
    """Disjoint target-side classes over ALL positions of a batch, before the >=64 crop.

    induction  exists p < j with idx[p] == idx[j] and idx[p+1] == tg[j]
    repeat     exists p <= j with idx[p] == tg[j], and not induction
    novel      everything else
    """
    B, L = idx.shape
    ar = torch.arange(L, device=idx.device)
    causal = ar.unsqueeze(1) < ar.unsqueeze(0)                 # [j, p] with p < j
    causal_incl = ar.unsqueeze(1) >= ar.unsqueeze(0)           # [j, p] with p <= j
    nxt = torch.cat([idx[:, 1:], torch.full((B, 1), -1, device=idx.device, dtype=idx.dtype)], 1)
    prev_match = idx.unsqueeze(1) == idx.unsqueeze(2)          # [b, j, p] idx[p] == idx[j]
    copy_match = nxt.unsqueeze(1) == tg.unsqueeze(2)           # [b, j, p] idx[p+1] == tg[j]
    induction = (prev_match & copy_match & causal.unsqueeze(0)).any(2)
    seen_tg = ((idx.unsqueeze(1) == tg.unsqueeze(2)) & causal_incl.unsqueeze(0)).any(2)
    return {'induction': induction,
            'repeat': seen_tg & ~induction,
            'novel': ~seen_tg & ~induction}


@torch.no_grad()
def sweep(rows, hooks=(), score=None):
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
def ce_by_class(rows, seen, hooks=()):
    """Total CE and per-class CE in ONE pass. Classes cost no extra forward."""
    acc = {'t': 0.0, 'n': 0}
    for c in CLASSES:
        acc[c] = [0.0, 0]

    def score(lg, tg, idx):
        e = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                            reduction='none').reshape(tg.shape)[:, 64:]
        cov = seen[idx[:, 64:]]
        acc['t'] += float(e[cov].sum()); acc['n'] += int(cov.sum())
        cls = token_classes(idx, tg)
        for c in CLASSES:
            msk = cls[c][:, 64:] & cov
            acc[c][0] += float(e[msk].sum()); acc[c][1] += int(msk.sum())
    sweep(rows, hooks=hooks, score=score)
    return acc


@torch.no_grad()
def fit_tables(rows, sites):
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
    hooks = [mod_of(*st).register_forward_hook(mk(st, j == 0)) for j, st in enumerate(sites)]
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


def per_tok(a, c):
    return a[c][0] / a[c][1] if a[c][1] else float('nan')


@torch.no_grad()
def main():
    t0 = time.time()
    K = torch.load(CONSTS, map_location='cpu')
    reg = json.load(open(REG))
    fit = load(FIT_ROWS)
    ev = load(EVAL[1])

    certified = [k for k, v in reg.items() if isinstance(v, dict) and v.get('status') == 'certified']
    auditable = [k for k in COMPONENTS if k in reg]
    print(f'CIRCUIT AUDIT v4 (damage decomposed by target token class) | registry {len(reg)} '
          f'entries, {len(certified)} certified | {len(auditable)} auditable', flush=True)
    print(f'  classes are DISJOINT and EXHAUSTIVE over scored positions; counts are asserted to sum',
          flush=True)
    print(f'  {"circuit":40s} {"removal":>8s} | per-token removal (nats/tok)      | share of damage',
          flush=True)
    print(f'  {"":40s} {"":>8s} | {"induc":>7s} {"repeat":>7s} {"novel":>7s} | '
          f'{"induc":>6s} {"repeat":>6s} {"novel":>6s}', flush=True)

    rows = {}
    counts_checked = False
    for name in auditable:
        sites = sites_of(COMPONENTS[name])
        _, seen = fit_tables(fit, sites)
        live = ce_by_class(ev, seen)
        const = ce_by_class(ev, seen, hooks=[mod_of(*st).register_forward_hook(
            const_hook(K[f'{st[0]}{st[1]}'].to(DEV).float(), st[0] == 'attn')) for st in sites])
        if not counts_checked:
            assert abs(live['t'] / live['n'] - S1695_CE_LIVE) <= 1e-3, (
                f'baseline CE {live["t"]/live["n"]:.5f} disagrees with {S1695_CE_LIVE} (§1695)')
            counts_checked = True
        assert sum(live[c][1] for c in CLASSES) == live['n'], (
            f'class counts {[live[c][1] for c in CLASSES]} do not sum to {live["n"]}')

        removal = const['t'] / const['n'] - live['t'] / live['n']
        pt = {c: per_tok(const, c) - per_tok(live, c) for c in CLASSES}
        dmg = {c: const[c][0] - live[c][0] for c in CLASSES}
        tot = sum(dmg.values())
        share = {c: (dmg[c] / tot if tot else float('nan')) for c in CLASSES}
        cc = CLAIM_CLASS[name]
        if cc == 'all':
            sel = None
        else:
            comp_d = sum(dmg[c] for c in CLASSES if c != cc)
            comp_n = sum(const[c][1] for c in CLASSES if c != cc)
            sel = (dmg[cc] / max(const[cc][1], 1)) / (comp_d / max(comp_n, 1)) if comp_n else None
        rows[name] = {'n_sites': len(sites), 'removal_nats': round(removal, 5),
                      'per_token_removal': {c: round(pt[c], 5) for c in CLASSES},
                      'damage_share': {c: round(share[c], 4) for c in CLASSES},
                      'class_token_counts': {c: live[c][1] for c in CLASSES},
                      'claim_class': cc,
                      'selectivity_claimed_vs_complement': None if sel is None else round(sel, 4)}
        print(f'  {name[:40]:40s} {removal:8.4f} | {pt["induction"]:7.4f} {pt["repeat"]:7.4f} '
              f'{pt["novel"]:7.4f} | {share["induction"]:6.1%} {share["repeat"]:6.1%} '
              f'{share["novel"]:6.1%}', flush=True)

    cnt = rows[auditable[0]]['class_token_counts']
    tot_n = sum(cnt.values())
    print(f'\n  scored tokens by class: induction {cnt["induction"]} ({cnt["induction"]/tot_n:.1%})  '
          f'repeat {cnt["repeat"]} ({cnt["repeat"]/tot_n:.1%})  novel {cnt["novel"]} '
          f'({cnt["novel"]/tot_n:.1%})', flush=True)

    print(f'\n  SELECTIVITY (per-token removal on the claimed class / on its complement):', flush=True)
    for n in auditable:
        s = rows[n]['selectivity_claimed_vs_complement']
        if s is not None:
            print(f'    {n[:44]:44s} claims {rows[n]["claim_class"]:9s} selectivity {s:6.3f}',
                  flush=True)

    # EVERY aggregate is computed over DISTINCT MEASUREMENTS, not registry rows. Codex found
    # (2026-08-28) that v3's pred_d passed only because three prose entries share the MLP0-3
    # component set and were counted three times; deduplicated it fails 3/5. §1722 named that
    # duplication as a limitation and then a predicate of mine was inflated by it anyway. Any
    # statistic over `auditable` double-counts; `distinct` is the only honest denominator.
    canon, distinct = {}, []
    for n in auditable:
        k = frozenset(sites_of(COMPONENTS[n]))
        canon[n] = k
        if k not in [canon[d] for d in distinct]:
            distinct.append(n)
    print(f'\n  {len(auditable)} registry rows collapse to {len(distinct)} DISTINCT component sets; '
          f'every aggregate below uses the distinct set', flush=True)
    ind_share = {n: rows[n]['damage_share']['induction'] for n in distinct}
    pa = (max(ind_share.values()) - min(ind_share.values())) >= 0.10
    att, mlp = '_attention_output_write_is_nonlocal', '_program_price_curve'
    ratio = {k: rows[k]['per_token_removal']['induction'] / rows[k]['per_token_removal']['novel']
             for k in (att, mlp)}
    pb = ratio[att] > ratio[mlp]
    pc = all(abs(rows[k]['removal_nats'] - v) <= 0.01 for k, v in S1725.items() if k in rows)
    lg1 = rows['_lag1_failure_is_middle_band']['per_token_removal']
    pd = lg1['induction'] >= 1.10 * lg1['novel']

    print(f'\n  induction damage share spread {min(ind_share.values()):.1%} .. '
          f'{max(ind_share.values()):.1%} -> discriminating {pa}', flush=True)
    print(f'  induction/novel per-token ratio: attention {ratio[att]:.3f} vs MLP stack '
          f'{ratio[mlp]:.3f} -> attention copy-concentrated {pb}', flush=True)
    print(f'  removals reproduce §1722/§1725 -> control {pc}', flush=True)
    print(f'  lag-1 entry induction {lg1["induction"]:.4f} vs novel {lg1["novel"]:.4f} nats/tok '
          f'-> per-context estimand rescues it {pd}', flush=True)

    res = {'config': {'eval_set': EVAL[0], 'fit_rows': 'fineweb_n96_skip80.pt',
                      'classes': 'target-side, disjoint, exhaustive: induction (target appears '
                                 'earlier preceded by the current token), repeat (target appears '
                                 'earlier, not in an induction position), novel (target absent from '
                                 'the context)',
                      'selectivity': 'per-token removal on the class the entry CLAIMS over per-token '
                                     'removal on the other two. The first per-context collateral '
                                     'number in this arc (§1721). Only claims whose context is a '
                                     'target-token property can be expressed; the rest are `all`.',
                      'claim_class': 'HAND-ANNOTATED reading of each entry, disputable per line'},
           'circuits': rows, 'claim_class': CLAIM_CLASS,
           'distinct_component_sets': distinct,
           'DEDUPLICATION_NOTE': 'aggregates use DISTINCT component sets, not registry rows. '
                                 'Sixteen entries collapse to fewer measurements; counting rows '
                                 'inflated v3 pred_d from a fail to a pass (Codex, 2026-08-28).',
           'registry_entries': len(reg), 'certified': len(certified), 'auditable': len(auditable),
           'predictions': {'pred_a_profile_discriminates': bool(pa),
                           'pred_b_attention_copy_concentrated': bool(pb),
                           'pred_c_removals_reproduce': bool(pc),
                           'pred_d_per_context_rescues_lag1': bool(pd)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
