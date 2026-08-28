# CLASS-GAP SITE DISCOVERY -- which individual sites drive the stack contrast, using a statistic
# that does not blow up.
#
# §1734 certified, on a clean confirmation role, that ablating the attention stack costs MORE per
# token on `induction` targets than on `novel` ones while ablating the MLP stack costs LESS. That is
# a STACK claim. The per-site version failed in §1728 and has not been retested since the causal-mask
# correction, and it failed for two separate reasons worth separating:
#
#   1. it did not replicate (11 of 18 matched layers held out, against a bar of 12), and
#   2. the statistic was ill-conditioned. §1728 used a RATIO, and three attention sites had a `novel`
#      damage that crosses zero, producing ratios of -3.6, -5.2 and -2.0 and making a median over
#      eighteen of them meaningless.
#
# Reason 2 is fixable and reason 1 is not, yet. This run fixes the statistic -- induction damage
# MINUS novel damage, in nats, which has no denominator to cross zero and is comparable across sites
# -- and measures every site on both large roles.
#
# IT CONFIRMS NOTHING. All three eval roles are spent for this hypothesis family: §1734 used
# skip1200 as its confirmation and skip7000/skip11000 were burned before that. Running the per-site
# question on burned rows and calling the result held-out is exactly the error Codex retired two of
# my scripts for. So this run is DISCOVERY ONLY, says so in its own JSON, and its product is a
# FROZEN RANKED HYPOTHESIS LIST -- the sites whose gap interval is entirely positive on BOTH roles --
# to be tested on whichever clean role becomes available next, without re-deciding what to look at
# after seeing that role.
#
# Registered predictions, TWO-SIDED per LESSONS 31:
#   pred_a THE STATISTIC IS WELL-CONDITIONED WHERE THE RATIO WAS NOT: all 36 gap intervals are finite
#          and narrower than 0.5 nats, with no NaN. If FALSE the difference is no better behaved than
#          the ratio and the per-site question needs a third formulation, not a third run.
#   pred_b THE ATTENTION EFFECT IS CONCENTRATED, NOT DIFFUSE: the top three attention sites by
#          positive gap hold at least half of the total positive attention gap. If FALSE the stack
#          contrast is spread thinly across all eighteen sites, which would explain why the per-site
#          test failed in §1728 and would mean no small set of sites is worth naming.
#   pred_c CONTROLS: the classifier passes its known-answer check, the per-site total removals sum to
#          §1662/§1682's 4.3301 and 3.5570 within 0.02, and the class counts reproduce §1734's
#          3394/9127/15453 and 3534/8804/15159 exactly. The counts are the one control here that a
#          class error would actually move.
#   pred_d THE SIGN IS STABLE: the gap has the same sign on both roles at at least 30 of 36 sites.
#          If FALSE, per-site gaps are not stable enough to freeze a hypothesis list from, and the
#          list this run produces should not be spent on a clean role.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bilin18_joint_removal import m, DEV
from ops.target_token_classes import target_token_classes

D = 1152; T = 256; NB = 2000
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/class_gap_site_discovery_results.json'
# skip7000/skip11000 are DISCOVERY here and nothing else: their class-ratio numbers were observed
# on the void partition (§1733), so neither can confirm anything about this hypothesis again.
# skip1200 is the CONFIRMATION role -- a pinned rowcache never scored for the class hypothesis. It
# has no published baseline CE, so it carries no baseline assert; the partition-INVARIANT §1662
# stakes below carry the run instead.
# ALL THREE ROLES ARE SPENT for the class hypothesis family (§1734 used skip1200 as its
# confirmation). This run is therefore DISCOVERY ONLY on the two large roles and says so in its own
# output; it confirms nothing and certifies nothing. Its product is a FROZEN RANKED HYPOTHESIS LIST
# for whichever clean role becomes available next.
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205, 1e-3, 'discovery'),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711, 1e-2, 'discovery')]
S1662_STAKE = {'mlp': 4.3301, 'attn': 3.5570}   # partition-invariant, untouched by §1733
# corrected-partition class counts published in §1734; a class error would move these
S1734_COUNTS = {'skip7000': {'induction': 3394, 'repeat': 9127, 'novel': 15453},
                'skip11000': {'induction': 3534, 'repeat': 8804, 'novel': 15159}}
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
CLASSES = ('induction', 'repeat', 'novel')
COV = {}
CTRL = {'classifier': False, 'baselines': True, 'counts': {}, 'stake': {}}

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


def gap(cs, ck, ls, sel=None):
    """induction per-token damage MINUS novel per-token damage.

    §1728 measured the per-site version of this contrast as a RATIO and it was ill-conditioned:
    three attention sites had a `novel` damage that crossed zero, so their ratios came out at -3.6,
    -5.2, -2.0 and a median over eighteen of them meant nothing. A DIFFERENCE has no denominator to
    cross zero. Same sign convention as the ratio -- positive means the site matters more for
    copyable targets -- and it is comparable across sites in nats, which a ratio is not.
    """
    return dmg(cs, ck, ls, 'induction', sel) - dmg(cs, ck, ls, 'novel', sel)


@torch.no_grad()
def main():
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
    sites = [(k, L) for k in ('mlp', 'attn') for L in range(18)]
    stacks = {'mlp': [s for s in sites if s[0] == 'mlp'],
              'attn': [s for s in sites if s[0] == 'attn']}
    out = {}
    print(f'CLASS-GAP SITE DISCOVERY | all 36 sites | induction MINUS novel per-token damage | '
          f'{NB} row-level draws | DISCOVERY ONLY, confirms nothing', flush=True)

    for ename, epath, ce_ref, tol, role in EVAL_SETS:
        ev = load(epath)
        ls, lk = per_row(ev)
        ntok = int(sum(float(lk[c].sum()) for c in CLASSES))
        base = sum(float(ls[c].sum()) for c in CLASSES) / ntok
        assert abs(base - ce_ref) <= tol, f'{ename} baseline CE {base:.5f} != {ce_ref} (+/-{tol})'
        CTRL['counts'][ename] = {c: int(lk[c].sum()) for c in CLASSES}
        g = torch.Generator().manual_seed(1734)
        nrow = ev.shape[0]
        sels = [torch.randint(0, nrow, (nrow,), generator=g) for _ in range(NB)]
        rows = {}
        for st in sites:
            cs, ck = per_row(ev, hooks=[mod_of(*st).register_forward_hook(
                const_hook(K[f'{st[0]}{st[1]}'].to(DEV).float()))])
            gp = gap(cs, ck, ls, None)
            dr = sorted(gap(cs, ck, ls, sl) for sl in sels)
            rows[f'{st[0]}{st[1]}'] = {
                'gap_nats': round(gp, 5),
                'gap_ci95': (round(dr[int(0.025 * NB)], 5), round(dr[int(0.975 * NB)], 5)),
                'induction_damage': round(dmg(cs, ck, ls, 'induction'), 5),
                'novel_damage': round(dmg(cs, ck, ls, 'novel'), 5),
                'total_removal_nats': round(
                    sum(float(cs[c].sum()) - float(ls[c].sum()) for c in CLASSES) / ntok, 5)}
        CTRL['stake'][ename] = {kind: round(sum(rows[f'{s[0]}{s[1]}']['total_removal_nats']
                                                for s in sl), 5) for kind, sl in stacks.items()}
        print(f'\n  {ename} [{role}]: baseline CE {base:.5f} | scored {ntok} | classes '
              f'{CTRL["counts"][ename]}', flush=True)
        rk = sorted(rows, key=lambda n: -rows[n]['gap_nats'])
        print(f'    top 6 by gap (site matters MORE for copyable targets):', flush=True)
        for n in rk[:6]:
            r = rows[n]
            print(f'      {n:7s} gap {r["gap_nats"]:+8.4f} CI {r["gap_ci95"]}  induction '
                  f'{r["induction_damage"]:+8.4f} novel {r["novel_damage"]:+8.4f}', flush=True)
        print(f'    bottom 3 by gap (site matters MORE for weight-resident targets):', flush=True)
        for n in rk[-3:]:
            r = rows[n]
            print(f'      {n:7s} gap {r["gap_nats"]:+8.4f} CI {r["gap_ci95"]}', flush=True)
        out[ename] = {'role': role, 'baseline_ce': round(base, 5), 'scored_tokens': ntok,
                      'class_counts': CTRL['counts'][ename], 'sites': rows, 'ranked': rk}
        del ev
        torch.cuda.empty_cache()

    a, b = out['skip7000']['sites'], out['skip11000']['sites']
    names = list(a)
    finite = all(abs(a[n]['gap_ci95'][1] - a[n]['gap_ci95'][0]) < 0.5
                 and a[n]['gap_nats'] == a[n]['gap_nats'] for n in names)
    pa = finite
    pos = [a[f'attn{L}']['gap_nats'] for L in range(18) if a[f'attn{L}']['gap_nats'] > 0]
    top3 = sum(sorted(pos, reverse=True)[:3])
    pb = bool(pos) and top3 >= 0.5 * sum(pos)
    pc = (CTRL['classifier']
          and all(abs(CTRL['stake']['skip7000'][k] - v) <= 0.02 for k, v in S1662_STAKE.items())
          and all(CTRL['counts'][e] == c for e, c in S1734_COUNTS.items()))
    agree = sum(1 for n in names if (a[n]['gap_nats'] > 0) == (b[n]['gap_nats'] > 0))
    pd = agree >= 30

    # the product of this run: a frozen ranked list for the NEXT clean role to test
    hypothesis = [n for n in out['skip7000']['ranked']
                  if a[n]['gap_ci95'][0] > 0 and b[n]['gap_ci95'][0] > 0]
    print(f'\n  all 36 gap CIs finite and narrower than 0.5 nats -> well-conditioned {pa}',
          flush=True)
    print(f'  top 3 attention sites hold {top3:.4f} of {sum(pos):.4f} positive attention gap '
          f'-> concentrated {pb}', flush=True)
    print(f'  §1662 stakes {CTRL["stake"]["skip7000"]} + §1734 class counts -> control {pc}',
          flush=True)
    print(f'  gap sign agrees across the two roles at {agree}/36 sites -> stable {pd}', flush=True)
    print(f'\n  FROZEN HYPOTHESIS for the next clean role ({len(hypothesis)} sites with both CIs '
          f'entirely positive): {hypothesis}', flush=True)

    res = {'config': {'eval_sets': [e[0] for e in EVAL_SETS], 'bootstrap_draws': NB,
                      'bootstrap': 'ROW-level clusters over the 192 eval rows; NOT document-clustered '
                                   '(the rowcache carries no document ids) -- §1701',
                      'ratio': 'per-token constant-ablation damage on `induction` targets over the '
                               'same on `novel` targets, JOINT over a whole stack. The PER-SITE '
                               'version of this contrast FAILED its held-out predictions in §1728 '
                               'and is not carried forward.'},
           'VOIDS': 'S1727-S1729 were computed on a future-looking induction mask (S1733). This run uses the shared tested classifier and reproduces NONE of their class-dependent numbers by design; the surviving controls are the partition-invariant baselines plus a known-answer check on the classifier itself.',
           'results': out, 'frozen_hypothesis_for_next_clean_role': hypothesis,
           'ROLE_NOTE': 'DISCOVERY ONLY. All three eval roles are now spent for the class hypothesis '
                        'family (S1734 used skip1200 as its confirmation). This run confirms nothing '
                        'and certifies nothing; its product is the frozen ranked list above.',
           'predictions': {'pred_a_gap_is_well_conditioned': bool(pa),
                           'pred_b_attention_gap_is_concentrated': bool(pb),
                           'pred_c_controls': bool(pc),
                           'pred_d_sign_stable_across_roles': bool(pd)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
