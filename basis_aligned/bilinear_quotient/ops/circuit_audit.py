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
# Registered predictions, TWO-SIDED per LESSONS 31:
#   pred_a THE AUDIT IS INFORMATIVE, not a formality: at least one circuit's extraction score
#          differs from another's by >= 20 percentage points. If every circuit extracts equally
#          well, the metric is not discriminating and the harness needs a different one.
#   pred_b OOD IS MILD FOR REMOVAL: every circuit's removal cost moves by <= 25% relatively between
#          the two eval sets. Removal is a property of the model, so a large move would mean the
#          eval sets differ more than the arc has assumed (§1683/§1693/§1701 found ~1.3 points on
#          levels with gains stable).
#   pred_c CONTROLS: the whole-MLP-stack row reproduces §1662's 4.3301-nat stake within 0.05, and
#          the baseline CE reproduces 3.29205 (§1695) on skip7000.
#   pred_d NON-VACUITY: at least 8 circuits are auditable, so the table is worth reading at all.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
REG = '/workspace/theseus-bench/registry/circuits.json'
OUT = PT + 'ops/circuit_audit_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt'),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt')]
MASK_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
S1695_CE_LIVE = 3.29205
S1662_MLP_STAKE = 4.3301
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
    print(f'CIRCUIT AUDIT | registry {len(reg)} entries, {len(certified)} certified | '
          f'{len(auditable)} auditable (have a component list)', flush=True)
    print(f'  BOOTSTRAP: every circuit scored on the same standard rows. Circuits with a narrow '
          f'trigger will read WEAKER than they are.', flush=True)

    rows = {}
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
            rows.setdefault(name, {})[ename] = {
                'n_sites': len(sites), 'ce_live': round(cl, 5),
                'removal_nats': round(removal, 5), 'extraction': round(extraction, 5)}
            if ename == 'skip7000':
                print(f'    {name[:46]:46s} {len(sites):2d} sites  removal {removal:7.4f}  '
                      f'extraction {extraction:7.2%}', flush=True)
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

    pa = (max(ex) - min(ex)) >= 0.20
    pb = max(ood) <= 0.25
    pc = (stack is not None) and abs(stack - S1662_MLP_STAKE) <= 0.05
    pd = len(auditable) >= 8

    print(f'\n  OOD (relative change in removal cost, skip7000 -> skip11000):', flush=True)
    for name in sorted(auditable, key=lambda n: -abs(rows[n]['ood']['removal_rel_change']))[:5]:
        o = rows[name]['ood']
        print(f'    {name[:46]:46s} removal {o["removal_rel_change"]:+7.2%}  '
              f'extraction {o["extraction_delta"]:+7.2%}', flush=True)
    print(f'\n  extraction spread {min(ex):.2%} .. {max(ex):.2%} -> discriminating {pa}', flush=True)
    print(f'  worst OOD removal move {max(ood):.2%} -> mild {pb}', flush=True)
    print(f'  MLP-stack removal {stack} vs §1662 {S1662_MLP_STAKE} -> control {pc}', flush=True)
    print(f'  {len(auditable)} circuits auditable of {len(certified)} certified -> {pd}', flush=True)

    res = {'config': {'registry': REG, 'eval_sets': [n for n, _ in EVAL_SETS],
                      'fit_rows': 'fineweb_n96_skip80.pt',
                      'removal': 'components replaced by their optimal constants; CE cost in nats',
                      'extraction': 'components replaced by a per-token table; fraction of removal cost recovered',
                      'ood': 'both scored on skip7000 (reference) and skip11000 (held out)',
                      'BOOTSTRAP_LIMITATION': 'every circuit scored on the same standard rows. Per-circuit '
                                              'evaluation data is an OPEN PROBLEM; a circuit with a narrow '
                                              'trigger reads WEAKER than it is because most rows do not '
                                              'trigger it. Fill `data_note` per entry as this becomes definable.',
                      'coverage': f'{len(auditable)} of {len(certified)} certified entries have a component list'},
           'circuits': rows,
           'registry_entries': len(reg), 'certified': len(certified), 'auditable': len(auditable),
           'predictions': {'pred_a_discriminating': bool(pa),
                           'pred_b_ood_mild_for_removal': bool(pb),
                           'pred_c_controls_hold': bool(pc),
                           'pred_d_enough_circuits': bool(pd)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
