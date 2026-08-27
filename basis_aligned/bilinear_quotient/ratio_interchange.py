# ratio_interchange: THE FOURTH AXIS — INTERCHANGE rather than removal. Does the
# slice carry CLASS-SPECIFIC content, or merely content?
#
# Every CE measurement in S1643-S1652 REMOVED the slice (mean- or zero-ablation) or
# removed a bigger slice (rank-8). Removal answers "does this subspace matter?" It
# cannot answer "does it matter FOR THIS CLASS?" -- a subspace carrying generic
# information would also cost CE when removed.
#
# INTERCHANGE separates those. Instead of replacing class C's slice coordinates with
# their global mean, replace them with the coordinates DONATED BY ANOTHER CLASS's
# slice-conditional mean. The subspace is still fully populated -- nothing is deleted --
# but it now carries the wrong class's content. If the slice is class-specific,
# interchange should cost MORE than removal, because a confidently wrong value is worse
# than a neutral one. If the slice carries generic content, interchange should cost
# roughly the SAME as removal or less.
#
# This is the axis S1650 named after its own intended axis turned out not to be one
# (zero- and mean-ablation agreed to within .00067). Applying that lesson, the
# manipulation check is a REGISTERED prediction: pred_b requires interchange to differ
# materially from the removal baseline on these exact classes, so an unvaried axis
# reports itself rather than being read as a finding.
#
# DESIGN: S1648's twelve type-spanning classes at mlp11, rank-2. Donor for each class
# is the NEXT class in a fixed cyclic order, so every class donates once and receives
# once and no donor is chosen by outcome. The donor's slice-conditional mean is
# computed in the RECEIVER's basis -- the subspace is the receiver's; only the values
# placed in it come from the donor.
#
# Registered predictions:
#   pred_a THE SLICE IS CLASS-SPECIFIC: mean relative CE rise under INTERCHANGE exceeds
#          the removal baseline of +.00676 (S1648, same twelve classes, rank-2).
#   pred_b MANIPULATION CHECK -- THE AXIS ACTUALLY VARIED: mean |interchange rise minus
#          removal rise| across the twelve exceeds .002, the threshold at which S1650's
#          failed axis moved only 1 of 12 classes.
#   pred_c THE RATIO STILL PREDICTS UNDER A NON-REMOVAL INTERVENTION:
#          rho(|lam1/lam2|, relative CE rise under interchange) >= +.30.
import json, time, sys, os, re, random, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV
import tiktoken

D = 1152; T = 256
SITE = 11
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ratio_interchange_results.json'
ROWCACHE = PT + '.rowcache/fineweb_n480_skip80.pt'
RECEIPT = PT + '.rowcache/fineweb_oracle_v2_receipt.json'
H = m.transformer.h
ENC = tiktoken.get_encoding('gpt2')
RANK = 2
NROWS = 480
REMOVAL_MEANRISE = 0.00676      # §1648, these exact twelve classes, rank-2 mean-ablation
S1650_UNVARIED_THRESHOLD = 0.002  # S1650's failed axis moved only 1 of 12 classes past this

# twelve TYPE-SPANNING classes; none has ever had its CE rise measured
PATS = {'exclaim': r'^!$|^ !$', 'semicolon': r'^;$|^ ;$', 'colon': r'^:$|^ :$',
        'quote': r'^"$|^ "$', 'dash': r'^-$|^ -$', 'digit': r'^ ?[0-9]+$',
        'cap': r'^ [A-Z][a-z]+$', 'we': r'^ (we|We)$', 'you': r'^ (you|You)$',
        'it': r'^ (it|It)$', 'an': r'^ an$', 'my': r'^ my$'}


def rx(pat):
    v = torch.zeros(50257, dtype=torch.bool)
    for t in range(50257):
        if re.match(pat, ENC.decode([t])):
            v[t] = True
    return v


def slice_and_eigs(mask_v):
    """Top-RANK |lambda| eigenpair of the class-projected quadratic. Weights only."""
    WU = m.lm_head.weight.float().to(DEV)[:50257]
    u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()
    Lw = H[SITE].mlp.Left.weight.float(); Rw = H[SITE].mlp.Right.weight.float()
    Dw = H[SITE].mlp.Down.weight.float()
    S = 0.5 * ((Lw.T @ ((u @ Dw)[:, None] * Rw)) + (Lw.T @ ((u @ Dw)[:, None] * Rw)).T)
    lam, V = torch.linalg.eigh(S)
    o = lam.abs().argsort(descending=True)[:RANK]
    return V[:, o].contiguous(), [float(lam[i]) for i in o]


def mk_pre_hook(V2, mu):
    """Replace the slice coordinates with `mu`. For REMOVAL mu is the global mean; for
    INTERCHANGE mu is the DONOR class's slice-conditional mean, expressed in the
    RECEIVER's basis. The subspace stays fully populated either way."""
    def pre(mod, args):
        f = args[0].float()
        p = f @ V2
        return ((f - (p - mu) @ V2.T).to(args[0].dtype),) + tuple(args[1:])
    return pre


@torch.no_grad()
def class_conditional_mean(rows, mask_v, V2):
    """Mean slice projection over the DONOR class's own positions, in V2's basis."""
    acc = {'s': torch.zeros(V2.shape[1], device=DEV), 'n': 0}
    def h(mod, args):
        f = args[0].float()
        pm = h.pm
        acc['s'] += (f.reshape(-1, D) @ V2)[pm.reshape(-1)].sum(0)
        acc['n'] += int(pm.sum())
        return None
    handle = H[SITE].mlp.register_forward_pre_hook(h)
    try:
        for i in range(0, rows.shape[0], 8):
            bb = rows[i:i + 8]
            idx = bb[:, :-1].to(DEV).contiguous(); tg = bb[:, 1:].to(DEV)
            pm = mask_v.to(DEV)[tg]; pm[:, :64] = False
            h.pm = pm
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            for blk in H:
                x, v1 = blk(x, v1, x0)
    finally:
        handle.remove()
    return acc['s'] / max(acc['n'], 1)


@torch.no_grad()
def ce_pass(rows, mask_v, V2, mu):
    """Mean CE on the class's own positions. mu=None captures the global slice mean."""
    cap = {}
    if mu is None:
        def h(mod, args):
            f = args[0].float().reshape(-1, D)
            cap['s'] = cap.get('s', torch.zeros(RANK, device=DEV)) + (f @ V2).sum(0)
            cap['n'] = cap.get('n', 0) + f.shape[0]
            return None
        handle = H[SITE].mlp.register_forward_pre_hook(h)
    else:
        handle = H[SITE].mlp.register_forward_pre_hook(mk_pre_hook(V2, mu))
    tot, npos = 0.0, 0
    try:
        for i in range(0, rows.shape[0], 8):
            bb = rows[i:i + 8]
            idx = bb[:, :-1].to(DEV).contiguous(); tg = bb[:, 1:].to(DEV)
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            for blk in H:
                x, v1 = blk(x, v1, x0)
            lg = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
            ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                                 reduction='none').reshape(tg.shape)
            pm = mask_v.to(DEV)[tg]; pm[:, :64] = False
            tot += float(ce[pm].sum()); npos += int(pm.sum())
    finally:
        handle.remove()
    return tot / max(npos, 1), npos, ((cap['s'] / max(cap['n'], 1)) if cap else None)


@torch.no_grad()
def main():
    import hashlib
    t0 = time.time()
    raw = torch.load(ROWCACHE, map_location='cpu')
    raw = raw['rows'] if isinstance(raw, dict) else raw
    rows = raw[:NROWS, :T + 1].contiguous()
    rh = hashlib.sha256(open(RECEIPT, 'rb').read()).hexdigest()[:16]
    print(f'INTERCHANGE: {len(PATS)} classes at mlp{SITE}, donor content in receiver basis. {NROWS} canonical '
          f'rows (receipt {rh}). Single registered hypothesis: |lam1/lam2|.', flush=True)

    ks = list(PATS)
    donor = {ks[i]: ks[(i + 1) % len(ks)] for i in range(len(ks))}   # fixed cyclic, outcome-blind
    print(f'donor map (fixed cyclic, chosen before any measurement): '
          f'{ {k: donor[k] for k in ks[:4]} } ...', flush=True)

    basis, masks, eigs_ = {}, {}, {}
    for c, pat in PATS.items():
        masks[c] = rx(pat)
        basis[c], eigs_[c] = slice_and_eigs(masks[c])

    per = {}
    for c in ks:
        d = donor[c]
        V2 = basis[c]                                   # RECEIVER's subspace
        mu_donor = class_conditional_mean(rows, masks[d], V2)   # DONOR's content, receiver's basis
        base, npos, _ = ce_pass(rows, masks[c], V2, None)
        inter, _, _ = ce_pass(rows, masks[c], V2, mu_donor)
        rel = (inter - base) / base if base > 0 else 0.0
        ratio = abs(eigs_[c][0]) / max(abs(eigs_[c][1]), 1e-12)
        per[c] = {'donor': d, 'ratio': round(ratio, 4), 'base_ce': round(base, 5),
                  'interchange_ce': round(inter, 5), 'rel_ce_rise': round(rel, 5),
                  'n_positions': npos}
        print(f'  {c:10s} <- {d:10s} ratio {ratio:6.3f} | n={npos:5d} | '
              f'CE {base:.4f} -> {inter:.4f} | rel rise {rel:+.5f}', flush=True)

    n = len(ks)
    ratio = {k: per[k]['ratio'] for k in ks}
    rel = {k: per[k]['rel_ce_rise'] for k in ks}

    def rk(x):
        o = sorted(x, key=lambda z: -x[z]); return [o.index(z) + 1 for z in ks]

    def rho(a, b):
        return 1 - 6 * sum((a[i] - b[i]) ** 2 for i in range(n)) / (n * (n * n - 1))

    r = rho(rk(ratio), rk(rel))
    mean_rise = sum(rel.values()) / n

    removal = json.load(open(PT + 'ratio_heldout2_types_results.json'))['per_class']
    diffs = {k: rel[k] - removal[k]['rel_ce_rise'] for k in ks}
    mean_abs_diff = sum(abs(v) for v in diffs.values()) / n
    n_moved = sum(1 for v in diffs.values() if abs(v) > S1650_UNVARIED_THRESHOLD)

    pa = mean_rise > REMOVAL_MEANRISE
    pb = mean_abs_diff > S1650_UNVARIED_THRESHOLD
    pc = r >= 0.30

    print(f'\n  INTERCHANGE vs REMOVAL on the same twelve classes:', flush=True)
    print(f'    mean relative rise: interchange {mean_rise:+.5f} vs removal '
          f'{REMOVAL_MEANRISE:+.5f}  ({mean_rise/REMOVAL_MEANRISE:.2f}x)', flush=True)
    print(f'    MANIPULATION CHECK: mean |difference| {mean_abs_diff:.5f} '
          f'(need >{S1650_UNVARIED_THRESHOLD}); classes moved past that: {n_moved}/12', flush=True)
    print(f'    rho(|lam1/lam2|, interchange rise) = {r:+.4f}', flush=True)

    out = {'config': {'site': SITE, 'rank': RANK, 'n_rows': NROWS,
                      'classes': 'TYPE-SPANNING held out -- punctuation, digits, capitalised, function words; no CE rise ever measured for any',
                      'single_registered_hypothesis': 'abs(lam1/lam2) predicts relative CE rise',
                      'currency': 'relative CE rise = ce_rise / base_ce (§1645)',
                      'removal_meanrise_S1648': REMOVAL_MEANRISE, 'intervention': 'INTERCHANGE (donor class-conditional mean)'},
           'per_class': per,
           'predictions': {'pred_a_interchange_costs_more_than_removal': bool(pa),
                           'pred_b_axis_actually_varied': bool(pb),
                           'pred_c_ratio_still_predicts': bool(pc)},
           'mean_relative_rise': round(mean_rise, 5), 'rho': round(r, 4),
           'mean_abs_diff_vs_removal': round(mean_abs_diff, 5),
           'classes_moved': n_moved, 'donor_map': donor,
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
