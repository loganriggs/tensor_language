# CLASS-RATIO CONFIRMATION ON A ROLE THAT HAS NEVER SEEN THIS HYPOTHESIS
#
# §1733: the induction mask compared `j < p` where the spec said `p < j`, so it searched FUTURE
# positions and voided §1727-§1729 plus two registry entries. Codex then retired both follow-on
# scripts on a second, separate ground that is also correct: the class-ratio numbers were OBSERVED
# on skip7000 AND skip11000, so neither role can confirm this hypothesis again. A corrected re-run
# on those rows would be a discovery measurement wearing a held-out label. Their guards stand --
# this is a new script, not an edit to them.
#
# ROLES, DECLARED BEFORE THE RUN:
#   skip7000, skip11000  DISCOVERY ONLY. Reported, never used to confirm. They are here to show the
#                        corrected numbers beside the void ones and to carry the §1662 controls.
#   skip1200             CONFIRMATION. `fineweb_n96_skip1200`, a pinned rowcache used as a role
#                        elsewhere in the arc but NEVER scored for the class hypothesis. Half the
#                        rows of the others, so its interval will be wider, and no published
#                        baseline CE, so it carries no baseline assert.
#
# WHY THE CONTROLS ARE DIFFERENT THIS TIME. §1733's post-mortem: every control I had was either
# invariant to the partition (pooled CE, total removal) or reproduced my own wrong computation. The
# controls here are chosen so at least one CAN fail on a class error:
#   - a hand-built known-answer check on the classifier, run in-process before anything loads
#   - the §1662/§1682 total constant-ablation stakes 4.3301 and 3.5570, which ARE partition-
#     invariant and therefore genuinely untouched by §1733 -- a real control, not a self-comparison
#   - NO comparison against any §1727-§1729 class number. Reproducing one would only prove the bug
#     came back. The single void number quoted, 0.8382, MEASURES the size of the error in pred_d.
#
# Registered predictions, TWO-SIDED per LESSONS 31:
#   pred_a CONFIRMED ON A CLEAN ROLE: on skip1200 the 95% interval on (attention ratio minus MLP
#          ratio) lies entirely ABOVE zero. If FALSE, the contrast does not survive the corrected
#          partition on a role that never saw it, and nothing from this thread returns to the
#          registry -- which after §1733 is the outcome to expect rather than resist.
#   pred_b THE SIGN AGREES ACROSS ALL THREE ROLES: attention above MLP on all three, or below on all
#          three. A sign that flips between roles is not stable enough to name.
#   pred_c CONTROLS: the classifier passes its known-answer check, and the skip7000 total removals
#          reproduce §1662/§1682's 4.3301 and 3.5570 within 0.01.
#   pred_d THE BUG WAS MATERIAL: the corrected skip7000 MLP-stack ratio differs from the void 0.8382
#          by at least 0.02. If FALSE the future-looking mask barely moved the answer, making §1733
#          a bookkeeping correction rather than a substantive one. Either way it does NOT
#          rehabilitate the void numbers.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bilin18_joint_removal import m, DEV
from ops.target_token_classes import target_token_classes

D = 1152; T = 256; NB = 2000
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/class_ratio_confirm_skip1200_results.json'
# skip7000/skip11000 are DISCOVERY here and nothing else: their class-ratio numbers were observed
# on the void partition (§1733), so neither can confirm anything about this hypothesis again.
# skip1200 is the CONFIRMATION role -- a pinned rowcache never scored for the class hypothesis. It
# has no published baseline CE, so it carries no baseline assert; the partition-INVARIANT §1662
# stakes below carry the run instead.
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205, 1e-3, 'discovery'),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711, 1e-2, 'discovery'),
             ('skip1200', PT + '.rowcache/fineweb_n96_skip1200.pt', None, None, 'CONFIRMATION')]
S1662_STAKE = {'mlp': 4.3301, 'attn': 3.5570}   # partition-invariant, untouched by §1733
VOID_S1728_MLP_RATIO = 0.8382  # quoted ONLY to measure how large the bug was, never as a target
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
LATE = [('attn', 14), ('attn', 15), ('attn', 16)]
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
    stacks = {'mlp': [('mlp', L) for L in range(18)], 'attn': [('attn', L) for L in range(18)]}
    out = {}
    print(f'JOINT CLASS-RATIO CI | 18-MLP vs 18-attention stacks | {NB} row-level bootstrap draws | '
          f'plus a sign decomposition of attn14/15/16', flush=True)

    for ename, epath, ce_ref, tol, role in EVAL_SETS:
        ev = load(epath)
        ls, lk = per_row(ev)
        ntok = int(sum(float(lk[c].sum()) for c in CLASSES))
        base = sum(float(ls[c].sum()) for c in CLASSES) / ntok
        if ce_ref is not None:
            assert abs(base - ce_ref) <= tol, f'{ename} baseline CE {base:.5f} != {ce_ref} (+/-{tol})'
        CTRL['counts'][ename] = {c: int(lk[c].sum()) for c in CLASSES}
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
        CTRL['stake'][ename] = {kind: round(
            sum(float(st[kind][0][c].sum()) - float(ls[c].sum()) for c in CLASSES) / ntok, 5)
            for kind in stacks}
        print(f'\n  {ename} [{role}]: baseline CE {base:.5f} (ref {ce_ref}) | scored {ntok} | '
              f'classes {CTRL["counts"][ename]} | total removal {CTRL["stake"][ename]}', flush=True)
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
        out[ename] = {'role': role, 'baseline_ce': round(base, 5), 'scored_tokens': ntok,
                      'class_counts': CTRL['counts'][ename], 'total_removal': CTRL['stake'][ename],
                      'joint_ratio': pt, 'joint_ratio_ci95': ci, 'late_attention': late}
        del ev, st
        torch.cuda.empty_cache()

    cf = out['skip1200']
    pa = cf['joint_ratio_ci95']['diff'][0] > 0
    signs = [out[e]['joint_ratio']['attn'] - out[e]['joint_ratio']['mlp'] > 0 for e in out]
    pb = all(signs) or not any(signs)
    pc = (CTRL['classifier']
          and all(abs(CTRL['stake']['skip7000'][k] - v) <= 0.01 for k, v in S1662_STAKE.items()))
    pd = abs(out['skip7000']['joint_ratio']['mlp'] - VOID_S1728_MLP_RATIO) >= 0.02

    print(f'\n  CONFIRMATION skip1200: mlp {cf["joint_ratio"]["mlp"]:.4f} attn '
          f'{cf["joint_ratio"]["attn"]:.4f} | diff CI {cf["joint_ratio_ci95"]["diff"]} '
          f'-> positive and excludes zero {pa}', flush=True)
    print(f'  sign of (attn - mlp) agrees across all three roles -> {pb} {signs}', flush=True)
    print(f'  classifier known-answer + §1662 stakes {CTRL["stake"]["skip7000"]} vs {S1662_STAKE} '
          f'-> control {pc}', flush=True)
    print(f'  corrected skip7000 mlp ratio {out["skip7000"]["joint_ratio"]["mlp"]:.4f} vs void '
          f'{VOID_S1728_MLP_RATIO} -> the bug was material {pd}', flush=True)

    res = {'config': {'eval_sets': [e[0] for e in EVAL_SETS], 'bootstrap_draws': NB,
                      'bootstrap': 'ROW-level clusters over the 192 eval rows; NOT document-clustered '
                                   '(the rowcache carries no document ids) -- §1701',
                      'ratio': 'per-token constant-ablation damage on `induction` targets over the '
                               'same on `novel` targets, JOINT over a whole stack. The PER-SITE '
                               'version of this contrast FAILED its held-out predictions in §1728 '
                               'and is not carried forward.'},
           'VOIDS': 'S1727-S1729 were computed on a future-looking induction mask (S1733). This run uses the shared tested classifier and reproduces NONE of their class-dependent numbers by design; the surviving controls are the partition-invariant baselines plus a known-answer check on the classifier itself.',
           'results': out,
           'predictions': {'pred_a_confirmed_on_skip1200': bool(pa),
                           'pred_b_sign_agrees_across_roles': bool(pb),
                           'pred_c_controls': bool(pc),
                           'pred_d_bug_was_material': bool(pd)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
