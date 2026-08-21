"""GROUP SCORECARD -- is the CO-ACTIVATION GROUP the right circuit unit (rescuing
763's failed atom-level convergence)? 761-765 converged on: the atom is the wrong
unit (redundant, unstable, individually mis-named); circuits are co-activation
GROUPS within a moderately-stable subspace. Test whether GROUPS beat ATOMS on the
three properties:
  STABILITY  -- cluster atoms by co-activation into G groups (seed0 and seed1);
                group stability = best subspace-overlap of a seed0 group's decoder
                span with a seed1 group. Compare to atom-match (763: 0.405).
  CAUSAL     -- knock a whole group (all member atoms) vs sum of member singles ->
                superadditivity (redundancy is WITHIN groups); group dCE vs random
                same-size atom set.
  MONOSEM    -- KL selectivity of the group's aggregate top-activating tokens vs
                the mean of its member atoms' KL. Are groups more nameable?

REGISTERED PREDICTIONS:
  (0) SANITY: clustering yields multi-atom groups;
  (a) GROUP RESCUES STABILITY: mean group subspace-overlap across seeds > atom-match
      0.405 by >= 0.15 (groups recur where atoms don't);
  (b) GROUP RESCUES CAUSE+MEANING: group knockout is SUPERADDITIVE (group dCE >
      sum-of-member-singles, ratio > 1.3 -> redundancy is intra-group) and >> a
      random same-size set; group KL >= mean member-atom KL (grouping does not
      lose, ideally gains, monosemanticity);
  NULL: random atom groupings are NOT more stable than atoms and NOT superadditive."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
from scipy.cluster.hierarchy import linkage, fcluster
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; HID = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'group_scorecard_results.json'
NFIT = 48; NEVAL = 24; P = 512; K = 32; G = 32; TOPN = 150; DCE_GROUPS = 24
SUB = {'d0': None}; KNOCK = {'mask': None}


def topk(pre, k):
    val, idx = pre.topk(k, dim=1); z = torch.zeros_like(pre); z.scatter_(1, idx, F.relu(val)); return z


def hook_d0(mo, i_, o_):
    return o_ if SUB['d0'] is None else SUB['d0'](i_[0].reshape(-1, HID)).reshape(o_.shape).to(o_.dtype)


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def ce_on(rows, n):
    s = 0.0; nn = 0
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1)
        s += float(F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1)))*idx.shape[0]; nn += idx.shape[0]
    return s/nn


@torch.no_grad()
def capture(rows, n, want_tokens=False):
    cap = []; toks = []
    h = m.transformer.h[0].mlp.Down.register_forward_hook(lambda mo, i_, o_: cap.append(i_[0].detach().float().reshape(-1, HID)))
    for i in range(0, n, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous()
        if want_tokens: toks.append(idx.reshape(-1).cpu())
        forward_logits(idx)
    h.remove(); g = torch.cat(cap, 0)
    return (g, torch.cat(toks).numpy()) if want_tokens else g


def sub_d0(Dm, Em, b):
    def fn(g):
        z = topk(g @ Em.T, K)
        if KNOCK['mask'] is not None: z = z * KNOCK['mask']
        return z @ Dm.T + b
    return fn


def train_sae(Xin, Ytrue, seed):
    torch.manual_seed(seed)
    Dm = (torch.randn(D, P, device=DEV)/np.sqrt(D)).requires_grad_(True)
    Em = (torch.randn(P, HID, device=DEV)/np.sqrt(HID)).requires_grad_(True)
    b = Ytrue.mean(0).clone().requires_grad_(True); opt = torch.optim.Adam([Dm, Em, b], lr=3e-3)
    for s in range(700):
        z = topk(Xin @ Em.T, K); loss = F.mse_loss(z @ Dm.T + b, Ytrue)
        opt.zero_grad(); loss.backward(); opt.step()
    return Dm.detach(), Em.detach(), b.detach()


def cluster(codes, active, g):
    Z = codes[:, active]                                  # (N, na)
    C = np.corrcoef(Z.T); C = np.nan_to_num(C)
    dist = 1 - C; np.fill_diagonal(dist, 0)
    from scipy.spatial.distance import squareform
    L = linkage(squareform(dist, checks=False), 'average')
    lab = fcluster(L, g, 'maxclust')
    return lab                                            # labels over active atoms


def subspace_ov(Dm, idx_a, Dn, idx_b):
    if len(idx_a) < 1 or len(idx_b) < 1: return 0.0
    Qa = torch.linalg.qr(Dm[:, idx_a])[0]; Qb = torch.linalg.qr(Dn[:, idx_b])[0]
    s = torch.linalg.svdvals(Qa.T @ Qb)
    return float(s.mean())


def kl_sel(code_col, toks, base, topn=TOPN):
    idx = np.argsort(-code_col)[:topn]; idx = idx[code_col[idx] > 0]
    if len(idx) < 10: return 0.0
    c = Counter(toks[idx].tolist()); tot = len(idx); kl = 0.0
    for t, cnt in c.items():
        pt = cnt/tot; kl += pt*np.log(pt/max(base.get(t, 1e-9), 1e-9))
    return float(kl)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NFIT + NEVAL); fit, ev = rows[:NFIT], rows[NFIT:NFIT+NEVAL]
    W0 = m.transformer.h[0].mlp.Down.weight.data.float().to(DEV)
    h0 = m.transformer.h[0].mlp.Down.register_forward_hook(hook_d0)
    g0 = capture(fit, NFIT); Y0 = g0 @ W0.T
    g0_ev, toks = capture(ev, NEVAL, want_tokens=True)
    base = Counter(toks.tolist()); N = len(toks); base = {t: c/N for t, c in base.items()}
    with torch.enable_grad():
        D0, E0, b0 = train_sae(g0, Y0, 0); D1, E1, b1 = train_sae(g0, Y0, 1)

    cod0 = topk(g0_ev @ E0.T, K).cpu().numpy(); cod1 = topk(g0_ev @ E1.T, K).cpu().numpy()
    act0 = np.where((cod0 > 1e-6).mean(0) > 0)[0]; act1 = np.where((cod1 > 1e-6).mean(0) > 0)[0]
    lab0 = cluster(cod0, act0, G); lab1 = cluster(cod1, act1, G)
    groups0 = [act0[lab0 == k] for k in range(1, G+1)]; groups0 = [g for g in groups0 if len(g) >= 2]
    groups1 = [act1[lab1 == k] for k in range(1, G+1)]; groups1 = [g for g in groups1 if len(g) >= 2]
    print(f'seed0 {len(groups0)} groups (mean size {np.mean([len(g) for g in groups0]):.1f})', flush=True)

    # (a) GROUP stability: best seed0->seed1 group subspace overlap
    ov = []
    for ga in groups0:
        ov.append(max(subspace_ov(D0, list(ga), D1, list(gb)) for gb in groups1))
    grp_stab = float(np.mean(ov))
    # random-grouping null: random same-size groups
    gnp = np.random.RandomState(0)
    rand_ov = []
    for ga in groups0:
        ra = gnp.choice(act0, len(ga), replace=False)
        rand_ov.append(max(subspace_ov(D0, list(ra), D1, list(gb)) for gb in groups1))
    print(f'(a) group subspace stability {grp_stab:.3f} (atom-match 0.405, random-group {np.mean(rand_ov):.3f})', flush=True)

    # (b) GROUP causal: knock whole group vs sum of singles (superadditivity), sample groups
    SUB['d0'] = sub_d0(D0, E0, b0); KNOCK['mask'] = None; ce0 = ce_on(ev, NEVAL)
    def knock_dce(atoms):
        mask = torch.ones(P, device=DEV); mask[list(atoms)] = 0.0; KNOCK['mask'] = mask
        c = ce_on(ev, NEVAL); KNOCK['mask'] = None; return c - ce0
    samp = sorted(range(len(groups0)), key=lambda i: -len(groups0[i]))[:DCE_GROUPS]
    supers = []; grp_dces = []; rand_dces = []
    single = {}
    for gi in samp:
        atoms = groups0[gi]; gd = knock_dce(atoms); grp_dces.append(gd)
        ss = 0.0
        for a in atoms.tolist():
            if a not in single: single[a] = knock_dce([a])
            ss += max(single[a], 0)
        supers.append(gd/max(ss, 1e-6))
        ra = gnp.choice(act0, len(atoms), replace=False); rand_dces.append(knock_dce(ra))
    superadd = float(np.mean(supers)); grp_dce_m = float(np.mean(grp_dces)); rand_dce_m = float(np.mean(rand_dces))
    print(f'(b) group dCE {grp_dce_m:.4f} vs random-set {rand_dce_m:.4f}  | superadditivity {superadd:.2f}', flush=True)

    # (c) GROUP monosemanticity: aggregate group code vs mean member-atom KL
    grp_kl = []; atom_kl_all = []
    for ga in groups0:
        agg = cod0[:, ga].sum(1); grp_kl.append(kl_sel(agg, toks, base))
        atom_kl_all.append(np.mean([kl_sel(cod0[:, a], toks, base) for a in ga]))
    grp_kl_m = float(np.mean(grp_kl)); atom_kl_m = float(np.mean(atom_kl_all))
    print(f'(c) group KL {grp_kl_m:.3f} vs mean member-atom KL {atom_kl_m:.3f}', flush=True)
    h0.remove()

    p0 = len(groups0) > 0
    pa = grp_stab - 0.405 >= 0.15 and grp_stab > np.mean(rand_ov) + 0.05
    pb = superadd > 1.3 and grp_dce_m > 1.5*rand_dce_m and grp_kl_m >= atom_kl_m
    null_ok = np.mean(rand_ov) < grp_stab
    out = {'n_groups': len(groups0), 'group_stability': round(grp_stab, 4), 'atom_match_ref': 0.405,
           'random_group_stability': round(float(np.mean(rand_ov)), 4), 'superadditivity': round(superadd, 3),
           'group_dce': round(grp_dce_m, 4), 'random_set_dce': round(rand_dce_m, 4),
           'group_kl': round(grp_kl_m, 4), 'atom_kl': round(atom_kl_m, 4),
           'pred_0': bool(p0), 'pred_a_stability': bool(pa), 'pred_b_cause_meaning': bool(pb),
           'null_ok': bool(null_ok), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) group rescues stability: {pa}; (b) group superadditive+causal+monosem: {pb}; NULL: {null_ok}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
