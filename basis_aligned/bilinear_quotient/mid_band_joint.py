# mid_band_joint: ARE THE TWELVE MIDDLE MLPs SMALL, OR ONLY INDIVIDUALLY SMALL?
#
# FINDINGS 13 carries an explicit caveat: of bilin18's eighteen MLPs only six (mlp0-3,
# mlp16, mlp17) clear the instrument's ~0.15-nat floor under single-module ablation. The
# other twelve are UNMEASURED, not measured-as-small, and I have been careful to say so.
#
# §1663 supplies the reason that caveat matters. mlp1's contextual input turned out to be
# REDUNDANT across the attention beneath it: freezing attn0 alone removed none of it,
# freezing attn1 alone removed 19%, freezing both removed all of it. A redundantly
# implemented function is invisible to single-component ablation -- every member looks
# free because the others cover for it. If the middle band is built that way, twelve
# individually-negligible ablations are exactly what a large, redundant computation looks
# like.
#
# TEST: ablate mlp4..mlp15 JOINTLY at their optimal constants and compare the joint stake
# against the sum of the twelve individual stakes. Then ask whether the group, taken
# together, is a token function: fit a per-token table for each of the twelve and measure
# the joint ceiling with the §1661 hybrid hook.
#
# INSTRUMENT CHECK with a known answer: with attn0..attn15 all frozen at their optimal
# constants, every module at or below mlp15 is a deterministic function of the current
# token, so the twelve covered tables must jointly be exact and the ceiling must be 1.0.
# That arm is a badly damaged model and is used for nothing else -- it certifies the
# instrument, which is the only thing that separated §1661 from its two broken versions.
#
# Registered predictions:
#   pred_a THE MIDDLE BAND IS NOT SMALL: joint optimal-constant stake for mlp4..mlp15 is
#          >= 1.0 nats, i.e. larger than mlp0, mlp2 and mlp3 put together.
#   pred_b IT IS REDUNDANT, THE §1663 PATTERN AT BAND SCALE: the joint stake exceeds the
#          sum of the twelve individual stakes. Redundant components each read as free
#          alone and expensive together.
#   pred_c INSTRUMENT CHECK: with attn0..attn15 frozen, the joint covered-position table
#          ceiling is >= 0.97. If it fails, the live joint ceiling is not interpretable
#          and only the stake arithmetic survives.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
BAND = list(range(4, 16))
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mid_band_joint_results.json'
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip1200.pt'
EVAL_ROWS = PT + '.rowcache/fineweb_n192_skip7000.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
FRONT_STAKES = {'mlp0': 0.8552, 'mlp1': 7.0050, 'mlp2': 0.7719, 'mlp3': 0.6201}
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
def sweep(rows, K, freeze_attn_upto=None, mlp_hooks=(), score=None):
    hs = []
    if freeze_attn_upto is not None:
        hs += [H[b].attn.register_forward_hook(module_freeze(K[f'attn{b}'].to(DEV).float()))
               for b in range(freeze_attn_upto + 1)]
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
def fit_tables(rows, K, freeze_attn_upto=None):
    """All twelve tables in a single pass."""
    s = {L: torch.zeros(50257, D, device=DEV) for L in BAND}
    c = torch.zeros(50257, device=DEV)
    first = {'done': False}

    def mk(L):
        def hook(mod, args, out):
            t = STATE['idx'].reshape(-1)
            s[L].index_add_(0, t, out.float().reshape(-1, D))
            if L == BAND[0]:
                c.index_add_(0, t, torch.ones_like(t, dtype=torch.float32))
                first['done'] = True
            return None
        return hook
    sweep(rows, K, freeze_attn_upto,
          mlp_hooks=[H[L].mlp.register_forward_hook(mk(L)) for L in BAND])
    assert first['done'], 'count hook never fired -- token counts would be all zero'
    seen = c > 0
    tables = {}
    for L in BAND:
        mean = s[L].sum(0) / c.sum()
        tbl = mean.unsqueeze(0).repeat(50257, 1)
        tbl[seen] = s[L][seen] / c[seen].unsqueeze(1)
        tables[L] = tbl
    return tables, seen


@torch.no_grad()
def ce(rows, K, seen, mode, sites=(), tables=None, freeze_attn_upto=None):
    """mode: live | const | table, applied at every site in `sites`."""
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
    sweep(rows, K, freeze_attn_upto, mlp_hooks=hooks, score=score)
    return acc['t'] / max(acc['n'], 1)


@torch.no_grad()
def main():
    t0 = time.time()
    fit = load(FIT_ROWS); ev = load(EVAL_ROWS)
    K = torch.load(CONSTS, map_location='cpu')
    print(f'MID BAND JOINT | sites mlp{BAND[0]}..mlp{BAND[-1]} | hybrid substitution, '
          f'covered-position scoring (§1661) | fit skip1200, eval skip7000', flush=True)

    tables, seen = fit_tables(fit, K)
    cl = ce(ev, K, seen, 'live')
    print(f'  CE live (covered positions) {cl:.5f} | {int(seen.sum())} tokens seen', flush=True)

    indiv = {}
    for L in BAND:
        cc = ce(ev, K, seen, 'const', sites=[L])
        indiv[f'mlp{L}'] = round(cc - cl, 5)
        print(f'    mlp{L:<2d} individual stake {cc - cl:7.4f}', flush=True)

    cc_j = ce(ev, K, seen, 'const', sites=BAND)
    joint = cc_j - cl
    ssum = sum(indiv.values())
    ct_j = ce(ev, K, seen, 'table', sites=BAND, tables=tables)
    ceil_live = (cc_j - ct_j) / joint if joint > 1e-6 else float('nan')

    print(f'\n  JOINT stake mlp4-15: {joint:.4f} nats', flush=True)
    print(f'  sum of twelve individual stakes: {ssum:.4f} nats   -> joint/sum '
          f'{joint / ssum if ssum > 1e-9 else float("nan"):.3f}', flush=True)
    print(f'  front band for scale: mlp0 {FRONT_STAKES["mlp0"]:.3f}  mlp1 '
          f'{FRONT_STAKES["mlp1"]:.3f}  mlp2 {FRONT_STAKES["mlp2"]:.3f}  mlp3 '
          f'{FRONT_STAKES["mlp3"]:.3f}', flush=True)
    print(f'  JOINT table ceiling (live model): {ceil_live:.2%}', flush=True)

    tf, seen_f = fit_tables(fit, K, freeze_attn_upto=BAND[-1])
    clf = ce(ev, K, seen_f, 'live', freeze_attn_upto=BAND[-1])
    ccf = ce(ev, K, seen_f, 'const', sites=BAND, freeze_attn_upto=BAND[-1])
    ctf = ce(ev, K, seen_f, 'table', sites=BAND, tables=tf, freeze_attn_upto=BAND[-1])
    stf = ccf - clf
    ceil_frozen = (ccf - ctf) / stf if stf > 1e-6 else float('nan')
    print(f'  INSTRUMENT CHECK -- attn0..attn15 frozen, known answer 1.0: {ceil_frozen:.2%} '
          f'(that arm: CE live {clf:.4f}, stake {stf:.4f})', flush=True)

    pa = joint >= 1.0
    pb = joint > ssum
    pc = ceil_frozen >= 0.97

    res = {'config': {'band': BAND, 'fit_rows': 'fineweb_n96_skip1200.pt',
                      'eval_rows': 'fineweb_n192_skip7000.pt',
                      'substitution': 'HYBRID -- table at covered positions, MLP live elsewhere (§1661)',
                      'scoring': 'covered positions only',
                      'ablation': 'optimal constants, opt_ablation_consts_all.pt',
                      'front_band_stakes': FRONT_STAKES,
                      'motivation': 'S1663 -- redundant computation is invisible to single-component ablation'},
           'ce_live': round(cl, 5), 'individual_stakes': indiv,
           'sum_of_individual': round(ssum, 5), 'joint_stake': round(joint, 5),
           'joint_over_sum': round(joint / ssum, 4) if ssum > 1e-9 else None,
           'joint_table_ceiling_live': round(ceil_live, 5),
           'instrument_check': {'known_answer': 1.0, 'observed': round(ceil_frozen, 5),
                                'frozen_arm_ce_live': round(clf, 5),
                                'frozen_arm_stake': round(stf, 5),
                                'passed': bool(pc)},
           'predictions': {'pred_a_band_not_small_ge_1nat': bool(pa),
                           'pred_b_redundant_joint_gt_sum': bool(pb),
                           'pred_c_instrument_check_ge_097': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
