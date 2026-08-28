# RETIRED: JOINT CLASS-RATIO CI used future tokens in its induction mask.
#
# The completed artifact inherits the reversed causal comparison from §1727/§1728:
# it selected p > j while interpreting p < j. It remains failure evidence but has
# no scientific authority. Correcting the mask after both eval roles were observed
# cannot rescue the registered confirmation. ``main`` refuses execution.
#
# §1728 (class_ratio_site_sweep) split §1727's contrast into two very different findings:
#
#   SURVIVES: the JOINT-STACK ratio. Ablating all 18 MLPs to their optimal constants damages
#             `induction` targets 0.838x as much per token as `novel` targets on skip7000 and 0.843x
#             held out; ablating all 18 attention sites gives 1.002 and 0.974. Reproduced to three
#             decimals as a control. But it has NO INTERVAL -- two point estimates on two sets.
#   FAILS:    the same contrast PER SITE. Held out, mlp_L sits below attn_L at only 11 of 18 matched
#             layers (bar 12), and the two stack medians' 95% intervals overlap heavily,
#             (0.434, 0.659) against (0.504, 0.679). The per-site version is not a law.
#
# And it left an anomaly: attn14, attn15 and attn16 have NEGATIVE ratios on both eval sets
# (-3.617, -5.180, -1.988 on skip7000; -3.540, -3.394, -1.285 held out). A ratio goes negative when
# a damage is negative -- constant-ablating the site IMPROVES cross-entropy somewhere. A median over
# eighteen sites is not a meaningful statistic when three of the denominators cross zero, which is
# an instrument problem on top of the failed prediction and is the reason the medians are being
# retired here rather than defended.
#
# This run does two things and only two:
#   1. Puts a 95% interval on the JOINT-stack ratios and on their DIFFERENCE, 2000 row-level draws,
#      with skip11000 carrying the claim. Either the surviving half of §1727 is resolved or it is a
#      point estimate and gets reported as one.
#   2. Decomposes attn14/15/16 by SIGN, per class, so the negative ratios are explained instead of
#      left as an anomaly in a table.
#
# Registered predictions, TWO-SIDED per LESSONS 31:
#   pred_a THE JOINT CONTRAST IS RESOLVED HELD OUT: the 95% interval on (attention ratio minus MLP
#          ratio) on skip11000 excludes zero. If FALSE, §1727's surviving half is a point-estimate
#          ordering at this row count and is NOT going in the registry.
#   pred_b THE NEGATIVE RATIOS ARE A NEGATIVE DENOMINATOR: at attn14, attn15 and attn16 the per-token
#          `novel` damage is negative while the `induction` damage is positive. If FALSE the sign
#          sits on the other side and the late-attention story is the opposite one.
#   pred_c AT LEAST ONE OF attn14/15/16 HAS NEGATIVE TOTAL REMOVAL -- replacing it with a constant
#          IMPROVES pooled CE on the eval rows. If FALSE, the site is net useful and only its `novel`
#          component is harmful, which is a sharper and more interesting claim than a broken site.
#   pred_d CONTROLS: baselines reproduce 3.29205 / 3.09711 (partition-invariant, so still
#          valid after §1733) AND the classifier passes a hand-built known-answer check.
#          §1728's class-dependent ratios are NOT reproduced as a control: they came from the
#          future-looking mask, so reproducing them would only prove the bug came back.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bilin18_joint_removal import m, DEV
from ops.target_token_classes import target_token_classes

D = 1152; T = 256; NB = 2000
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/class_ratio_joint_ci_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205, 1e-3),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711, 1e-2)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
LATE = [('attn', 14), ('attn', 15), ('attn', 16)]
CLASSES = ('induction', 'repeat', 'novel')
COV = {}
CTRL = {'classifier': False, 'baselines': True}

def assert_classifier_is_past_facing():
    """The check whose absence let §1733 through: a hand-built four-token example with a known
    answer, plus an invariance check that a position's label cannot move when the FUTURE changes.
    Pooled baselines and count-sum asserts are invariant to the partition and cannot catch this."""
    idx = torch.tensor([[5, 7, 5, 7]]); tg = torch.tensor([[7, 5, 7, 9]])
    c = target_token_classes(idx, tg)
    assert c['induction'].tolist() == [[False, False, True, False]], c['induction']
    assert c['repeat'].tolist() == [[False, True, False, False]], c['repeat']
    assert c['novel'].tolist() == [[True, False, False, True]], c['novel']
    l = torch.tensor([[5, 7, 5, 7, 1, 2]]); r = torch.tensor([[5, 7, 5, 7, 5, 7]])
    t = torch.tensor([[7, 5, 7, 9, 3, 4]])
    for k in ('induction', 'repeat', 'novel'):
        assert torch.equal(target_token_classes(l, t)[k][:, :4],
                           target_token_classes(r, t)[k][:, :4]), k




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
def per_row(rows, hooks=()):
    n = rows.shape[0]
    s = {c: torch.zeros(n) for c in CLASSES}
    k = {c: torch.zeros(n) for c in CLASSES}
    hs = list(hooks)
    try:
        for i in range(0, n, 8):
            bb = rows[i:i + 8]
            idx = bb[:, :-1].to(DEV).contiguous()
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            for blk in H:
                x, v1 = blk(x, v1, x0)
            lg = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
            tg = bb[:, 1:].to(DEV)
            e = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                                reduction='none').reshape(tg.shape)[:, 64:]
            cls = target_token_classes(idx, tg)
            cov = COV['seen'][idx[:, 64:]]
            for c in CLASSES:
                msk = cls[c][:, 64:] & cov
                s[c][i:i + bb.shape[0]] = (e * msk).sum(1).cpu()
                k[c][i:i + bb.shape[0]] = msk.sum(1).float().cpu()
    finally:
        for h in hs:
            h.remove()
    return s, k


def dmg(cs, ck, ls, c, sel=None):
    """Per-token damage in class c: (constant CE minus live CE) over that class's tokens."""
    a = cs[c] if sel is None else cs[c][sel]
    b = ls[c] if sel is None else ls[c][sel]
    nn = ck[c] if sel is None else ck[c][sel]
    n = float(nn.sum())
    return ((float(a.sum()) - float(b.sum())) / n) if n else float('nan')


def ratio(cs, ck, ls, sel=None):
    den = dmg(cs, ck, ls, 'novel', sel)
    return dmg(cs, ck, ls, 'induction', sel) / den if abs(den) > 1e-9 else float('nan')


@torch.no_grad()
def main():
    raise RuntimeError(
        'RETIRED: the registered class-ratio confirmation used a future-looking '
        'induction mask and cannot be repaired after both eval roles were observed'
    )
    t0 = time.time()
    assert_classifier_is_past_facing()
    CTRL['classifier'] = True
    print('  classifier known-answer check PASSED: past-facing induction, labels invariant '
          'to the future suffix (§1733/LESSONS 34)', flush=True)
    K = torch.load(CONSTS, map_location='cpu')
    fit = load(FIT_ROWS)
    seen = torch.zeros(50257, dtype=torch.bool)
    seen[fit[:, :T].reshape(-1).long()] = True
    COV['seen'] = seen.to(DEV)
    stacks = {'mlp': [('mlp', L) for L in range(18)], 'attn': [('attn', L) for L in range(18)]}
    out = {}
    print(f'JOINT CLASS-RATIO CI | 18-MLP vs 18-attention stacks | {NB} row-level bootstrap draws | '
          f'plus a sign decomposition of attn14/15/16', flush=True)

    for ename, epath, ce_ref, tol in EVAL_SETS:
        ev = load(epath)
        ls, lk = per_row(ev)
        ntok = int(sum(float(lk[c].sum()) for c in CLASSES))
        base = sum(float(ls[c].sum()) for c in CLASSES) / ntok
        assert abs(base - ce_ref) <= tol, f'{ename} baseline CE {base:.5f} != {ce_ref} (+/-{tol})'
        st = {}
        for kind, sl in stacks.items():
            st[kind] = per_row(ev, hooks=[mod_of(*s).register_forward_hook(
                const_hook(K[f'{s[0]}{s[1]}'].to(DEV).float())) for s in sl])
        pt = {kind: round(ratio(*st[kind], ls), 5) for kind in stacks}

        g = torch.Generator().manual_seed(1728)
        nrow = ev.shape[0]
        draws = {'mlp': [], 'attn': [], 'diff': []}
        for _ in range(NB):
            sel = torch.randint(0, nrow, (nrow,), generator=g)
            a = ratio(*st['mlp'], ls, sel); b = ratio(*st['attn'], ls, sel)
            draws['mlp'].append(a); draws['attn'].append(b); draws['diff'].append(b - a)
        ci = {k: (round(sorted(v)[int(0.025 * NB)], 4), round(sorted(v)[int(0.975 * NB)], 4))
              for k, v in draws.items()}
        print(f'\n  {ename}: baseline CE {base:.5f} (ref {ce_ref}) | scored {ntok}', flush=True)
        for kind in stacks:
            print(f'    {kind:4s} stack ratio {pt[kind]:7.4f}  95% CI {ci[kind]}', flush=True)
        print(f'    attn minus mlp {ci["diff"]} -> excludes zero '
              f'{ci["diff"][0] > 0 or ci["diff"][1] < 0}', flush=True)

        late = {}
        print(f'    late attention, per-token damage by class (negative = ablation IMPROVES CE):',
              flush=True)
        for s in LATE:
            cs, ck = per_row(ev, hooks=[mod_of(*s).register_forward_hook(
                const_hook(K[f'{s[0]}{s[1]}'].to(DEV).float()))])
            d = {c: dmg(cs, ck, ls, c) for c in CLASSES}
            tot = (sum(float(cs[c].sum()) - float(ls[c].sum()) for c in CLASSES)) / ntok
            late[f'{s[0]}{s[1]}'] = {'per_token_damage': {c: round(d[c], 5) for c in CLASSES},
                                     'total_removal_nats': round(tot, 5),
                                     'ratio': round(d['induction'] / d['novel'], 4)
                                     if abs(d['novel']) > 1e-9 else None}
            print(f'      {s[0]}{s[1]:<2d}  induction {d["induction"]:+8.4f}  repeat '
                  f'{d["repeat"]:+8.4f}  novel {d["novel"]:+8.4f}  |  total removal {tot:+8.4f}',
                  flush=True)
        out[ename] = {'baseline_ce': round(base, 5), 'scored_tokens': ntok,
                      'joint_ratio': pt, 'joint_ratio_ci95': ci, 'late_attention': late}
        del ev, st
        torch.cuda.empty_cache()

    ho = out['skip11000']
    pa = ho['joint_ratio_ci95']['diff'][0] > 0 or ho['joint_ratio_ci95']['diff'][1] < 0
    pb = all(out[e]['late_attention'][f'attn{L}']['per_token_damage']['novel'] < 0
             and out[e]['late_attention'][f'attn{L}']['per_token_damage']['induction'] > 0
             for e in out for L in (14, 15, 16))
    pc = any(out['skip11000']['late_attention'][f'attn{L}']['total_removal_nats'] < 0
             for L in (14, 15, 16))
    pd = CTRL['classifier'] and CTRL['baselines']

    print(f'\n  HELD OUT diff CI {ho["joint_ratio_ci95"]["diff"]} -> joint contrast resolved {pa}',
          flush=True)
    print(f'  attn14/15/16 negative DENOMINATOR on both sets -> {pb}', flush=True)
    print(f'  some late attention site has negative total removal -> {pc}', flush=True)
    print(f'  joint ratios reproduce §1728 -> control {pd}', flush=True)

    res = {'config': {'eval_sets': [e[0] for e in EVAL_SETS], 'bootstrap_draws': NB,
                      'bootstrap': 'ROW-level clusters over the 192 eval rows; NOT document-clustered '
                                   '(the rowcache carries no document ids) -- §1701',
                      'ratio': 'per-token constant-ablation damage on `induction` targets over the '
                               'same on `novel` targets, JOINT over a whole stack. The PER-SITE '
                               'version of this contrast FAILED its held-out predictions in §1728 '
                               'and is not carried forward.'},
           'VOIDS': 'S1727-S1729 were computed on a future-looking induction mask (S1733). This run uses the shared tested classifier and reproduces NONE of their class-dependent numbers by design; the surviving controls are the partition-invariant baselines plus a known-answer check on the classifier itself.',
           'results': out,
           'predictions': {'pred_a_joint_contrast_resolved': bool(pa),
                           'pred_b_negative_denominator': bool(pb),
                           'pred_c_late_attn_negative_removal': bool(pc),
                           'pred_d_controls': bool(pd)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
