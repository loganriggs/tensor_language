"""FIX ROUND for section 38 (review section 38b; predictions registered in
qk_dgp_fixround_predictions.json BEFORE this code was written).

Four review-mandated measurements on the planted-modular DGP cells.  The
frozen scoring code in qk_dgp_modular.py is NOT modified; every needed frozen
function is a verbatim port (stated per function), and every deviation forced
by the measurement design is stated in the JSON under 'deviations'.

  F1  Oracle-coordinate gate test: run the frozen section-34 naming gate on
      ORACLE coordinates -- the true planted units' value-path images on the
      identifiable/semi model (no retraining, no CP).  For each head h and
      unit u the oracle coordinate is vdir = W_v[head-h rows] @ e_u with e_u
      the planted unit direction in embedding space (content block =
      UNITS[:, u]); its write direction and exact weight-derived spectrum are
      built exactly as the frozen D3 block builds them from CP vdirs.
      STATED DEVIATION: the frozen gate QRs all of a head's coordinates into
      one write basis; 24 oracle units exceed the 16-dim per-head write
      space, so each oracle coordinate is scored on its own normalized write
      direction (equivalent to being first in the QR order).  The per-unit
      headline takes each unit's best head, mirroring the frozen pipeline's
      pooling of components across heads.

  F2  Symmetric D1/D2 rescoring: rerun the frozen per-head pipeline on the
      existing identifiable checkpoints (components were not persisted), then
      score the semi arm THREE ways -- raw content-slice (frozen D1,
      reproduction), (a) content-slice PROJECTED onto the 24-dim planted-unit
      span, (b) pulled through a ridge map fit exactly as the learned arm's
      (rms-normed FROZEN embedding rows -> planted bundle vectors, ridge =
      1e-3 * trace/Ws, unigram-weighted) -- and the learned arm through its
      map (frozen D2, reproduction).

  F3  D1 seed replication: retrain identifiable/semi at 4 fresh seeds (model
      init seed s in {1,2,3,4}, fresh data seeds 100+s; the original run used
      data seeds 11/12/13), frozen pipeline, D1 raw + F2a-projected scoring
      per seed.  Held/estimation sets stay the original ones (frozen).

  F4  D4 dose-response with standard errors on identifiable/semi: per-token
      and sequence-clustered SEs on mean_dce for every arm (per-position dCE
      persisted to qk_dgp_fixround_d4_dce.npz); selection-contrast top-k dose
      sweep k = 1..4 on the three largest whole-head-dCE heads with
      PCA-top-k and shrink-matched baselines at each k; privilege ratios
      with leave-one-sequence-out jackknife SEs (energies are fixed-seed MC
      constants).  Plus the overlap confirmation number: the fraction of
      total pattern energy carried by the planted class-contrast top-2
      directions in the overlap model's load-bearing head.

Output: qk_dgp_fixround.json (stages merge).  --stage f1,f2,f3,f4 (default
all, in that order).
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
sys.path.insert(0, QK)
import qk_dgp_lang as DL  # noqa: E402
from qk_dgp_lang import V, T, WS, NCLASS, NUNITS  # noqa: E402
import tf_model as TFM  # noqa: E402,F401

torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
HD = 16
NH = WS // HD
TRAIN_SEED, HELD_SEED, EST_SEED = 11, 12, 13         # frozen-run seeds
STEPS, HELD_N, EST_N = 8000, 1500, 1500
SAE_STEPS, SAE_BATCH, SAE_LR = 6000, 4096, 3e-3
RESTARTS, MARGIN, R2_THRESH = 3, 0.15, 0.8
D4_HELD = 256
ENERGY_BATCHES = (8, 4096)
OUT = f'{QK}/qk_dgp_fixround.json'
NPZ = f'{QK}/qk_dgp_fixround_d4_dce.npz'

STAGES = [s.strip() for s in (
    sys.argv[sys.argv.index('--stage') + 1] if '--stage' in sys.argv
    else 'f1,f2,f3,f4').split(',')]

t0 = time.time()


def say(msg):
    print(f'[{time.time() - t0:7.1f}s] {msg}', flush=True)


def save(result):
    prev = {}
    if os.path.exists(OUT):
        try:
            prev = json.load(open(OUT))
        except Exception:
            prev = {}
    prev.update(result)
    json.dump(prev, open(OUT, 'w'), indent=2)


# ----------------------------------------------------------------------------
# Verbatim ports of the frozen machinery from qk_dgp_modular.py (which itself
# ports qk_stage23.py tick 173 / qk_null_repair.py tick 183 via
# qk_iface_ledger.py).  NOT modified.
# ----------------------------------------------------------------------------
def cp_fit(core_raw, R, seed, n_starts=8, iters=60):
    """Symmetric nonneg CP via tensor power iteration + deflation (verbatim)."""
    mdim = core_raw.shape[0]
    gg = torch.Generator().manual_seed(seed)
    scale = core_raw.norm().clamp_min(1e-30)
    res = (core_raw / scale).clone()
    Us, lams = [], []
    for _ in range(R):
        M1 = res.reshape(mdim, mdim * mdim)
        best_u, best_lam = None, -1.0
        for _s in range(n_starts):
            u = torch.rand(mdim, generator=gg).to(core_raw.device)
            u = u / u.norm()
            for _ in range(iters):
                u = (M1 @ (u[:, None] * u[None, :]).reshape(-1)).clamp_min(0)
                n = float(u.norm())
                if n < 1e-20:
                    break
                u = u / n
            lam = float(torch.einsum('abc,a,b,c->', res, u, u, u))
            if lam > best_lam:
                best_lam, best_u = lam, u
        if best_lam <= 0:
            break
        Us.append(best_u)
        lams.append(best_lam)
        res = res - best_lam * torch.einsum('a,b,c->abc', best_u, best_u, best_u)
    if not Us:
        return torch.zeros(mdim, 1, device=core_raw.device)
    return torch.stack(Us, 1)


def eval_on_core(core_n, U, ridge=1e-8, polish=300):
    """Nonneg lambda refit on THIS (unit-norm) core for fixed factors (verbatim)."""
    R = U.shape[1]
    h = torch.einsum('abc,ar,br,cr->r', core_n, U, U, U)
    G = (U.T @ U) ** 3
    lam = torch.clamp(torch.linalg.solve(
        G + ridge * torch.eye(R, device=U.device), h), min=0)
    L = float(torch.linalg.eigvalsh(G)[-1].clamp_min(1e-12))
    for _ in range(polish):
        lam = torch.clamp(lam - (G @ lam - h) / L, min=0)
    res2 = 1.0 - 2.0 * float(lam @ h) + float(lam @ G @ lam)
    return max(res2, 0.0) ** 0.5, lam


def stability(Us):
    """Matched-component |cos| across restarts (verbatim)."""
    vals = []
    for i in range(len(Us)):
        for j in range(i + 1, len(Us)):
            C = (Us[i].T @ Us[j]).abs()
            vals.append(float(C.max(1).values.mean()))
    return sum(vals) / len(vals)


def train_sae(Z, m_atoms, k_code, p_cpu, steps, batch, lr=SAE_LR, seed=0):
    """Nonneg k-sparse coder (verbatim port from qk_dgp_modular.py)."""
    N = Z.shape[0]
    g = torch.Generator(device='cpu').manual_seed(seed)
    Dm = Z[torch.multinomial(p_cpu, m_atoms, replacement=True,
                             generator=g).to(Z.device)].clone()
    Dm = Dm + 1e-4 * torch.randn(Dm.shape, generator=g).to(Z.device)
    Dm = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
    We = Dm.clone()
    b = (p_cpu.to(Z.device)[:, None] * Z).sum(0).clone()
    for t in (Dm, We, b):
        t.requires_grad_(True)
    opt = torch.optim.Adam([Dm, We, b], lr=lr)
    fired = torch.zeros(m_atoms, device=Z.device)
    for step in range(steps):
        kk = max(k_code, int(round(2 * k_code - k_code * min(1.0, 2 * step / steps))))
        bi = torch.multinomial(p_cpu, batch, replacement=True,
                               generator=g).to(Z.device)
        y = Z[bi]
        Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z = torch.relu((y - b) @ We.T)
        vals, idx = z.topk(min(kk, m_atoms), dim=1)
        yhat = b + (vals.unsqueeze(-1) * Dn[idx]).sum(1)
        fired.index_add_(0, idx.reshape(-1), (vals > 1e-8).float().reshape(-1))
        loss = ((yhat - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            dead = (fired == 0).nonzero().squeeze(1)
            if len(dead):
                with torch.no_grad():
                    Dn_ = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    z_ = torch.relu((Z - b) @ We.T)
                    v_, i_ = z_.topk(min(k_code, m_atoms), dim=1)
                    rec = b + (v_.unsqueeze(-1) * Dn_[i_]).sum(1)
                    worst = ((rec - Z) ** 2).sum(1).topk(len(dead)).indices
                    Dm.data[dead] = Z[worst] / Z[worst].norm(
                        dim=1, keepdim=True).clamp(min=1e-8)
                    We.data[dead] = Dm.data[dead]
                    del z_, rec
            fired.zero_()
    with torch.no_grad():
        z = torch.relu((Z - b) @ We.T)
        vals, idx = z.topk(min(k_code, m_atoms), dim=1)
        S = torch.zeros(N, m_atoms, device=Z.device)
        S.scatter_(1, idx, vals)
        Dn = (Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)).detach()
    return S, Dn


def build_core_weighted(S, p):
    return torch.einsum('t,ta,tb,tc->abc', p.to(S.device), S, S, S)


def hungarian_vs_units(C):
    ri, ci = linear_sum_assignment(-C.numpy())
    cos24 = np.zeros(C.shape[1])
    assign = {}
    for r, c in zip(ri, ci):
        cos24[c] = float(C[r, c])
        assign[int(c)] = int(r)
    return cos24, assign


def run_head_pipeline(model, h, p_cpu, dev):
    """Frozen ledger pipeline on one head (verbatim port)."""
    fold = model.fold_layer0_qk(materialize=False, device=dev)
    X = torch.cat([fold['K1'][h], fold['K2'][h], fold['Vv'][h]], 1).to(dev)
    p = p_cpu.to(dev)
    mu = (p[:, None] * X).sum(0)
    Xc = X - mu
    C = (Xc * p[:, None]).T @ Xc
    ev = torch.linalg.eigvalsh(C).clamp_min(0)
    eff = float(ev.sum() ** 2 / (ev ** 2).sum().clamp_min(1e-30))
    r = min(16, max(2, int(round(eff))))
    m_atoms, k_code = 2 * r, 2
    S, Dn = train_sae(X, m_atoms, k_code, p_cpu, SAE_STEPS, SAE_BATCH)
    core = build_core_weighted(S, p)
    core_n = (core / core.norm().clamp_min(1e-30)).cpu()
    Us, rels = [], []
    for sd in range(RESTARTS):
        U = cp_fit(core_n, r, sd)
        rel, _ = eval_on_core(core_n, U)
        Us.append(U)
        rels.append(rel)
    best = int(np.argmin(rels))
    U_best, real_fit = Us[best], rels[best]
    stab = stability(Us)
    gp_ = torch.Generator().manual_seed(7)
    S_null = S.clone()
    for f in range(m_atoms):
        S_null[:, f] = S_null[torch.randperm(V, generator=gp_).to(S.device), f]
    core_null_ = build_core_weighted(S_null, p)
    U_null = cp_fit((core_null_ / core_null_.norm().clamp_min(1e-30)).cpu(), r, 0)
    null_on_real, _ = eval_on_core(core_n, U_null)
    margin = null_on_real - real_fit
    _, lam_best = eval_on_core(core_n, U_best)
    Wv = model.h[0].c_v.weight.detach().float()[h * HD:(h + 1) * HD].to(dev)
    G = Wv @ Wv.T
    comps = []
    order = lam_best.argsort(descending=True)
    for ri in order.tolist():
        if float(lam_best[ri]) <= 0:
            continue
        d48 = (U_best[:, ri].to(Dn.device) @ Dn)
        vdir = d48[2 * HD:]
        if float(vdir.norm()) < 1e-8:
            continue
        e_dir = Wv.T @ torch.linalg.solve(G, vdir.to(dev))
        comps.append({'lambda': round(float(lam_best[ri]), 4),
                      'vdir': vdir.cpu(), 'e_dir': e_dir.cpu()})
    return {'eff': round(eff, 2), 'r': r, 'm_atoms': m_atoms,
            'real_fit': round(real_fit, 4),
            'null_on_real': round(null_on_real, 4),
            'margin': round(margin, 4), 'm1_pass': bool(margin >= MARGIN),
            'restart_stability': round(stab, 3),
            'restart_relerrs': [round(x, 4) for x in rels]}, comps


# ----------------------------------------------------------------------------
# Shared setup (frozen-run data): tables, held/est sets, unigram exposure,
# checkpointed models.
# ----------------------------------------------------------------------------
say(f'=== fix round, stages {STAGES}, device {DEV} ===')
tabs = DL.DGPTables(variant='identifiable')
held_seqs = DL.sample_seqs(tabs, HELD_N, HELD_SEED)
est_seqs = DL.sample_seqs(tabs, EST_N, EST_SEED)
cnt = torch.bincount(est_seqs.flatten(), minlength=V).float() + 0.5
p_est = cnt / cnt.sum()


def load_arm(variant, arm, tabs_v, ck=None):
    ck = ck or f'{QK}/qk_dgp_{variant}_{arm}.pt'
    m = DL.make_arm_model(tabs_v, arm, seed=0, device=DEV)
    st = torch.load(ck, map_location=DEV)
    m.load_state_dict(st['state_dict'])
    return m.eval(), st['log']


def build_cls_lib(tabs_v):
    lib = {}
    for kcl in range(NCLASS):
        lib[f'class_{kcl}'] = (tabs_v.CLS == kcl)
    for u in range(NUNITS):
        lib[f'unit_{u}'] = (tabs_v.CHI[:, u] > 0)
    return lib


def two_level_best(y, w, cls_lib):
    """One coordinate's frequency-weighted two-level class coding -- the exact
    inner loop of the frozen D3 gate (same accumulation, same guards)."""
    mu_ = float((w * y).sum() / w.sum())
    var = float((w * (y - mu_) ** 2).sum())
    best = (None, 0.0)
    r2_by = {}
    for nm, mask_ in cls_lib.items():
        mk = mask_.float()
        wm = float((w * mk).sum())
        if wm < 1e-6:
            continue
        a_in = float((w * mk * y).sum() / wm)
        w0 = float((w * (1 - mk)).sum())
        a_out = float((w * (1 - mk) * y).sum() / w0)
        pred_ = mk * a_in + (1 - mk) * a_out
        r2 = 1 - float((w * (y - pred_) ** 2).sum()) / max(var, 1e-12)
        r2_by[nm] = r2
        if r2 > best[1]:
            best = (nm, r2)
    return best, r2_by


# ============================================================================
# F1 -- oracle-coordinate gate test
# ============================================================================
if 'f1' in STAGES:
    say('F1: oracle-coordinate gate test (identifiable/semi)')
    model, _ = load_arm('identifiable', 'semi', tabs)
    fold = model.fold_layer0_qk(materialize=False, device=DEV)
    cw = model.h[0].c_proj.weight.detach().float().to(DEV)     # (Ws, Dc)
    Wv = model.h[0].c_v.weight.detach().float().to(DEV)        # (Dc, Ws)
    w = p_est.clone().clamp_min(1e-9)
    cls_lib = build_cls_lib(tabs)
    recs = []
    for h in range(NH):
        wtab = fold['Vv'][h].to(DEV) @ cw[:, h * HD:(h + 1) * HD].T  # (V, Ws)
        for u in range(NUNITS):
            e_u = torch.zeros(WS, device=DEV)
            e_u[DL.CONTENT_SLICE] = tabs.UNITS[:, u].to(DEV)
            vd = Wv[h * HD:(h + 1) * HD] @ e_u                 # value-path image
            vnorm = float(vd.norm())
            q = cw[:, h * HD:(h + 1) * HD] @ (vd / max(vnorm, 1e-12))
            q = q / q.norm().clamp_min(1e-12)
            y = (wtab @ q).cpu()
            (bnm, br2), r2_by = two_level_best(y, w, cls_lib)
            recs.append({'head': h, 'unit': u, 'vdir_norm': round(vnorm, 4),
                         'best_class': bnm, 'best_r2': round(br2, 4),
                         'r2_correct_unit': round(
                             r2_by.get(f'unit_{u}', float('nan')), 4)})
    # per-unit best head (the pipeline pools coordinates across heads)
    per_unit = []
    for u in range(NUNITS):
        rs = [r for r in recs if r['unit'] == u]
        b = max(rs, key=lambda r: r['best_r2'])
        per_unit.append({'unit': u, 'best_head': b['head'],
                         'best_r2': b['best_r2'],
                         'best_class': b['best_class'],
                         'named_correctly': b['best_class'] == f'unit_{u}',
                         'r2_correct_unit': b['r2_correct_unit'],
                         'best_r2_correct_over_heads': round(max(
                             r['r2_correct_unit'] for r in rs), 4)})
    all_r2 = np.array([r['best_r2'] for r in recs])
    unit_r2 = np.array([pu['best_r2'] for pu in per_unit])
    cert = int((unit_r2 >= R2_THRESH).sum())
    named = sum(1 for pu in per_unit if pu['named_correctly'])
    per_head_cert = {f'h{h}': int(sum(1 for r in recs if r['head'] == h
                                      and r['best_r2'] >= R2_THRESH))
                     for h in range(NH)}
    F1 = {
        'construction': 'oracle coordinate (h,u): vdir = W_v[head h rows] @ '
                        'e_u, e_u = planted unit direction (content block = '
                        'UNITS[:,u]); write direction and exact weight-derived '
                        'spectrum built exactly as the frozen D3 block; scored '
                        'by the frozen two-level class coder, weights p_est',
        'deviation': 'single-coordinate write basis per oracle coordinate '
                     '(24 units > 16-dim per-head write space, so the frozen '
                     'multi-coordinate QR is impossible); per-unit headline = '
                     'best head, mirroring the pipeline pooling across heads',
        'units_certified_at_R2_0.8_besthead': cert,
        'units_named_correctly_besthead': named,
        'unit_best_r2_mean': round(float(unit_r2.mean()), 4),
        'unit_best_r2_min': round(float(unit_r2.min()), 4),
        'unit_best_r2_quartiles': [round(float(q), 4) for q in
                                   np.percentile(unit_r2, [25, 50, 75])],
        'all_192_r2_quartiles': [round(float(q), 4) for q in
                                 np.percentile(all_r2, [25, 50, 75])],
        'all_192_frac_ge_0.8': round(float((all_r2 >= 0.8).mean()), 4),
        'per_head_certified_of_24': per_head_cert,
        'per_unit': per_unit,
        'all_records': recs,
    }
    # attribution diagnostic: the same two-level coder on the IDENTITY value
    # path -- y_u(t) = <rms_norm(E_t) content part, unit u> on the planted
    # table (no trained weights).  If this is ~1 while the oracle-coordinate
    # R2 is low, the coder is fine and the failure is the per-head
    # write-coordinate construction (24 orthonormal units compressed through
    # a 16-dim per-head value space must leak into each other).
    En = F.rms_norm(tabs.E, (WS,))
    id_r2 = []
    for u in range(NUNITS):
        y = En[:, DL.CONTENT_SLICE] @ tabs.UNITS[:, u]
        mk = (tabs.CHI[:, u] > 0).float()
        mu_ = float((w * y).sum() / w.sum())
        var = float((w * (y - mu_) ** 2).sum())
        wm = float((w * mk).sum())
        a_in = float((w * mk * y).sum() / wm)
        w0 = float((w * (1 - mk)).sum())
        a_out = float((w * (1 - mk) * y).sum() / w0)
        pred_ = mk * a_in + (1 - mk) * a_out
        id_r2.append(1 - float((w * (y - pred_) ** 2).sum()) / max(var, 1e-12))
    id_r2 = np.array(id_r2)
    F1['identity_value_path_diagnostic'] = {
        'r2_mean': round(float(id_r2.mean()), 4),
        'r2_min': round(float(id_r2.min()), 4),
        'frac_ge_0.8': round(float((id_r2 >= 0.8).mean()), 4),
        'note': 'two-level coder on clean unit-projection spectra of the '
                'planted table itself (identity value path): the coder is '
                'near-perfect, so the oracle-coordinate failure localizes to '
                'the per-head write-coordinate spectrum construction, not '
                'the class-coding scheme'}
    save({'F1': F1})
    say(f'F1: {cert}/24 units certify at R2 >= 0.8 (best-head); named '
        f'correctly {named}/24; unit best-R2 mean {unit_r2.mean():.4f} min '
        f'{unit_r2.min():.4f}')


# ============================================================================
# F2 -- symmetric D1/D2 rescoring (rerun pipeline on existing checkpoints)
# ============================================================================
def normalize_rows(rows):
    return torch.stack([r / r.norm().clamp_min(1e-12) for r in rows])


def hung24(rows64):
    Cm = (normalize_rows(rows64) @ tabs.UNITS).abs()
    cos24, _ = hungarian_vs_units(Cm)
    return cos24


def fit_ridge_map(Xl):
    """The frozen D2 map fit, verbatim math: unigram-weighted ridge LS from
    rms-normed embedding rows to planted bundle vectors, ridge=1e-3*trace/Ws."""
    W_ = p_est
    A = Xl.T @ (W_[:, None] * Xl)
    lam_r = 1e-3 * float(torch.diagonal(A).sum()) / WS
    ML = torch.linalg.solve(A + lam_r * torch.eye(WS),
                            Xl.T @ (W_[:, None] * tabs.B)).T   # (64, Ws)
    pred = Xl @ ML.T
    ss_res = float((W_[:, None] * (pred - tabs.B) ** 2).sum())
    mu_b = (W_[:, None] * tabs.B).sum(0)
    ss_tot = float((W_[:, None] * (tabs.B - mu_b) ** 2).sum())
    return ML, 1 - ss_res / max(ss_tot, 1e-12)


def score_variants(dirs, ML_semi=None):
    """Score a list of embedding-space component directions all ways."""
    out = {}
    raw = hung24([e[DL.CONTENT_SLICE] for e in dirs])
    out['raw_content_slice'] = raw
    proj = hung24([tabs.UNITS @ (tabs.UNITS.T @ e[DL.CONTENT_SLICE])
                   for e in dirs])
    out['projected_unit_span'] = proj
    if ML_semi is not None:
        out['ridge_map'] = hung24([ML_semi @ e for e in dirs])
    return out


def pipeline_components(model):
    lst, heads = [], {}
    for h in range(NH):
        hres, cs = run_head_pipeline(model, h, p_est.cpu(), DEV)
        heads[f'h{h}'] = hres
        lst += [c['e_dir'] for c in cs]
    return lst, heads


def summ(cos24):
    return {'frac_ge_0.9': round(float((cos24 >= 0.9).mean()), 4),
            'n_ge_0.9': int((cos24 >= 0.9).sum()),
            'mean': round(float(cos24.mean()), 4),
            'min': round(float(cos24.min()), 4),
            'matched_cosines': [round(float(x), 4) for x in cos24]}


if 'f2' in STAGES or 'f3' in STAGES:
    say('F2: rerunning frozen pipeline on identifiable/semi + learned '
        '(components were not persisted)')
    semi_model, _ = load_arm('identifiable', 'semi', tabs)
    dirs_semi, heads_semi = pipeline_components(semi_model)
    say(f'  semi: {len(dirs_semi)} components pooled')

if 'f2' in STAGES:
    learned_model, _ = load_arm('identifiable', 'learned', tabs)
    dirs_learned, heads_learned = pipeline_components(learned_model)
    say(f'  learned: {len(dirs_learned)} components pooled')
    ML_semi, r2_semi_map = fit_ridge_map(F.rms_norm(tabs.E, (WS,)))
    Xl_learned = F.rms_norm(
        learned_model.wte.weight.detach().float().cpu(), (WS,))
    ML_learned, r2_learned_map = fit_ridge_map(Xl_learned)
    sv = score_variants(dirs_semi, ML_semi)
    lv_map = hung24([ML_learned @ e for e in dirs_learned])
    F2 = {
        'note': 'pipeline rerun on the frozen checkpoints (same seeds/config; '
                'SAE Adam on GPU is the only nondeterminism source, so the '
                'raw reproduction row is the comparability check vs the '
                'original D1 16/24 = 0.6667 / D2 21/24 = 0.875)',
        'semi_n_components': len(dirs_semi),
        'learned_n_components': len(dirs_learned),
        'semi_raw_reproduction': summ(sv['raw_content_slice']),
        'semi_a_projected_unit_span': summ(sv['projected_unit_span']),
        'semi_b_ridge_map': summ(sv['ridge_map']),
        'semi_ridge_map_weighted_R2': round(r2_semi_map, 4),
        'learned_ridge_map_reproduction': summ(lv_map),
        'learned_ridge_map_weighted_R2': round(r2_learned_map, 4),
        'original_D1_frac': 0.6667, 'original_D2_frac': 0.875,
        'per_head_semi': heads_semi, 'per_head_learned': heads_learned,
    }
    F2['reversal_survives_symmetric_treatment'] = bool(
        F2['learned_ridge_map_reproduction']['frac_ge_0.9']
        - max(F2['semi_a_projected_unit_span']['frac_ge_0.9'],
              F2['semi_b_ridge_map']['frac_ge_0.9']) > 0.05)
    save({'F2': F2})
    say(f"F2: semi raw {F2['semi_raw_reproduction']['n_ge_0.9']}/24 | "
        f"projected {F2['semi_a_projected_unit_span']['n_ge_0.9']}/24 | "
        f"ridge {F2['semi_b_ridge_map']['n_ge_0.9']}/24 | learned "
        f"{F2['learned_ridge_map_reproduction']['n_ge_0.9']}/24")


# ============================================================================
# F3 -- D1 seed replication (4 fresh train seeds of identifiable/semi)
# ============================================================================
if 'f3' in STAGES:
    say('F3: seed replication of identifiable/semi (seeds 1..4, fresh data '
        'seeds 101..104)')
    ref = DL.entropy_reference(tabs, held_seqs)
    rows = []
    # seed-0 row = the original checkpoint, scored from the F2 rerun components
    sv0 = score_variants(dirs_semi)
    rows.append({'seed': 0, 'data_seed': TRAIN_SEED,
                 'checkpoint': 'qk_dgp_identifiable_semi.pt (original)',
                 'held_ce': None,
                 'raw': summ(sv0['raw_content_slice']),
                 'projected': summ(sv0['projected_unit_span'])})
    for s in (1, 2, 3, 4):
        ck = f'{QK}/qk_dgp_identifiable_semi_fixseed{s}.pt'
        m = DL.make_arm_model(tabs, 'semi', seed=s, device=DEV)
        if os.path.exists(ck):
            st = torch.load(ck, map_location=DEV)
            m.load_state_dict(st['state_dict'])
            log = st['log']
            say(f'  seed {s}: loaded checkpoint (held CE '
                f'{log["final_held_ce"]:.4f})')
        else:
            say(f'  seed {s}: sampling fresh data (seed {100 + s}) + training')
            tr = DL.sample_seqs(tabs, STEPS * DL.BATCH, 100 + s)
            log = DL.train_arm(m, tr, held_seqs, STEPS, DEV, name=f'semi_s{s}')
            torch.save({'state_dict': m.state_dict(), 'log': log}, ck)
        m.eval()
        gap = log['final_held_ce'] - ref['true_law_logloss_mean']
        dirs_s, heads_s = pipeline_components(m)
        svs = score_variants(dirs_s)
        rows.append({'seed': s, 'data_seed': 100 + s,
                     'held_ce': round(log['final_held_ce'], 5),
                     'task_gap_nats': round(gap, 5),
                     'task_gate_pass': bool(gap <= 0.05),
                     'n_components': len(dirs_s),
                     'raw': summ(svs['raw_content_slice']),
                     'projected': summ(svs['projected_unit_span'])})
        say(f'  seed {s}: gap {gap:+.4f} | raw '
            f'{rows[-1]["raw"]["n_ge_0.9"]}/24 (mean '
            f'{rows[-1]["raw"]["mean"]}) | projected '
            f'{rows[-1]["projected"]["n_ge_0.9"]}/24 (mean '
            f'{rows[-1]["projected"]["mean"]})')
    raw_fracs = [r['raw']['frac_ge_0.9'] for r in rows]
    proj_fracs = [r['projected']['frac_ge_0.9'] for r in rows]
    F3 = {'note': 'seed 0 = original checkpoint rescored from the F2 rerun; '
                  'seeds 1-4 fresh model-init + fresh data seeds 101-104; '
                  'held/est sets frozen (seeds 12/13)',
          'per_seed': rows,
          'raw_frac_ge_0.9_all_seeds': raw_fracs,
          'projected_frac_ge_0.9_all_seeds': proj_fracs,
          'raw_pass_0.8_bar': [f >= 0.8 for f in raw_fracs],
          'projected_pass_0.8_bar': [f >= 0.8 for f in proj_fracs]}
    save({'F3': F3})
    say(f'F3: raw fracs {raw_fracs} | projected fracs {proj_fracs}')


# ============================================================================
# F4 -- D4 dose-response with standard errors
# ============================================================================
if 'f4' in STAGES:
    say('F4: D4 dose-response with SEs (identifiable/semi)')
    model, _ = load_arm('identifiable', 'semi', tabs)
    fold0 = model.fold_layer0_qk(materialize=False, device=DEV)
    mlpT = [model.fold_mlp(0, device=DEV).to(DEV)]
    d4_seqs = held_seqs[:D4_HELD]
    pD = p_est.to(DEV)
    CLS_D = tabs.CLS.to(DEV)

    @torch.no_grad()
    def fold_ce(fd, batch=8):
        outs = []
        for i in range(0, len(d4_seqs), batch):
            b = d4_seqs[i:i + batch].to(DEV)
            lg = model.fold_forward(b[:, :-1], fold=fd, mlp_tensors=mlpT)
            ce = F.cross_entropy(lg.float().reshape(-1, V),
                                 b[:, 1:].reshape(-1), reduction='none')
            outs.append(ce.view(b.shape[0], -1).cpu())
        return torch.cat(outs)

    def with_keys(K1h, K2h, h):
        fd = dict(fold0)
        fd['K1'] = fold0['K1'].clone()
        fd['K2'] = fold0['K2'].clone()
        fd['K1'][h] = K1h
        fd['K2'][h] = K2h
        return fd

    def ses(delta):
        """(per-token SE, sequence-clustered SE) of the mean of delta (n,T-1)."""
        flat = delta.flatten().numpy()
        seqm = delta.mean(1).numpy()
        return (float(flat.std(ddof=1) / np.sqrt(len(flat))),
                float(seqm.std(ddof=1) / np.sqrt(len(seqm))))

    base = fold_ce(fold0)
    with torch.no_grad():
        idc = d4_seqs[:4, :-1].to(DEV)
        fid = float((model.fold_forward(idc, fold=fold0, mlp_tensors=mlpT)
                     - model(idc)).abs().max())
    say(f'  fold identity control: max |logit diff| {fid:.2e}; base CE '
        f'{float(base.mean()):.5f}')

    head_dce = {}
    npz_store = {'base': base.numpy().astype(np.float32)}
    for h in range(NH):
        z = torch.zeros_like(fold0['K1'][h])
        d = fold_ce(with_keys(z, z, h)) - base
        se_t, se_s = ses(d)
        head_dce[h] = {'mean_dce': round(float(d.mean()), 5),
                       'se_token': round(se_t, 6),
                       'se_seq_clustered': round(se_s, 6)}
        npz_store[f'wholehead_h{h}'] = d.numpy().astype(np.float32)
    top3 = sorted(range(NH), key=lambda h: -head_dce[h]['mean_dce'])[:3]
    say('  whole-head dCE: ' + ' '.join(
        f"h{h}={head_dce[h]['mean_dce']:+.4f}(+-{head_dce[h]['se_seq_clustered']:.4f})"
        for h in range(NH)) + f' -> sweep heads {top3}')

    def class_contrast(K, k_top):
        Mn = torch.stack([(pD[:, None] * (CLS_D == kcl)[:, None] * K).sum(0)
                          / (pD * (CLS_D == kcl)).sum()
                          for kcl in range(NCLASS)], 1)      # (hd, 8)
        gm = (pD[:, None] * K).sum(0)
        return torch.linalg.svd(Mn - gm[:, None],
                                full_matrices=False).U[:, :k_top]

    def class_span(K):
        Mn = torch.stack([(pD[:, None] * (CLS_D == kcl)[:, None] * K).sum(0)
                          / (pD * (CLS_D == kcl)).sum()
                          for kcl in range(NCLASS)], 1)
        return torch.linalg.qr(Mn).Q

    def project_out(K, Q):
        return K - (K @ Q) @ Q.T

    @torch.no_grad()
    def pattern_energy(Hs, K1a, K2a, K1ref, K2ref, seed=0):
        nb, n = ENERGY_BATCHES
        g = torch.Generator().manual_seed(seed)
        Q1f, Q2f = fold0['Q1'][Hs], fold0['Q2'][Hs]
        tot = 0.0
        for _ in range(nb):
            si = torch.multinomial(p_est, n, replacement=True,
                                   generator=g).to(DEV)
            ti = torch.multinomial(p_est, n, replacement=True,
                                   generator=g).to(DEV)
            s1 = Q1f[si] @ K1ref[ti].T / HD
            s2 = Q2f[si] @ K2ref[ti].T / HD
            a1 = Q1f[si] @ K1a[ti].T / HD
            a2 = Q2f[si] @ K2a[ti].T / HD
            tot += float(((s1 * s2 - a1 * a2) ** 2).mean())
        return tot / nb

    def run_arm(name, Hs, K1a, K2a, K1ref, K2ref):
        E = pattern_energy(Hs, K1a, K2a, K1ref, K2ref)
        d = (fold_ce(with_keys(K1a, K2a, Hs)) - base)
        se_t, se_s = ses(d)
        npz_store[name] = d.numpy().astype(np.float32)
        m = float(d.mean())
        return {'energy_removed': E, 'mean_dce': round(m, 5),
                'se_token': round(se_t, 6), 'se_seq_clustered': round(se_s, 6),
                'dce_per_energy': round(m / max(E, 1e-12), 3),
                'seq_means': d.mean(1).numpy()}

    def ratio_jack(arm_s, arm_b):
        """Privilege ratio (sel dce/energy over baseline dce/energy) with
        leave-one-sequence-out jackknife SE; energies fixed constants."""
        ds, db = arm_s['seq_means'], arm_b['seq_means']
        Es, Eb = arm_s['energy_removed'], arm_b['energy_removed']
        n = len(ds)
        efac = Eb / max(Es, 1e-30)
        tot_s, tot_b = ds.sum(), db.sum()
        r_full = (tot_s / max(tot_b, 1e-30)) * efac
        r_i = ((tot_s - ds) / np.clip(tot_b - db, 1e-30, None)) * efac
        se = float(np.sqrt((n - 1) / n * ((r_i - r_i.mean()) ** 2).sum()))
        return round(float(r_full), 4), round(se, 4)

    sweep = {}
    for Hs in top3:
        K1 = fold0['K1'][Hs].clone()
        K2 = fold0['K2'][Hs].clone()
        EP2 = pattern_energy(Hs, torch.zeros_like(K1), torch.zeros_like(K2),
                             K1, K2)
        hrow = {'E_P2_total': EP2, 'k': {}}
        for k in (1, 2, 3, 4):
            C1, C2 = class_contrast(K1, k), class_contrast(K2, k)
            pca1 = torch.linalg.svd(pD.sqrt()[:, None] * K1,
                                    full_matrices=False).Vh[:k].T
            pca2 = torch.linalg.svd(pD.sqrt()[:, None] * K2,
                                    full_matrices=False).Vh[:k].T
            arm_sel = run_arm(f'h{Hs}_sel_top{k}', Hs,
                              project_out(K1, C1), project_out(K2, C2), K1, K2)
            arm_pca = run_arm(f'h{Hs}_pca_top{k}', Hs,
                              project_out(K1, pca1), project_out(K2, pca2),
                              K1, K2)
            frac = min(1.0, (arm_sel['energy_removed'] / max(EP2, 1e-30)) ** 0.5)
            beta = (1 - frac) ** 0.5
            arm_shr = run_arm(f'h{Hs}_shrink_top{k}', Hs, K1 * beta, K2 * beta,
                              K1, K2)
            rp, rp_se = ratio_jack(arm_sel, arm_pca)
            rs, rs_se = ratio_jack(arm_sel, arm_shr)
            krow = {'sel': {kk: vv for kk, vv in arm_sel.items()
                            if kk != 'seq_means'},
                    'pca': {kk: vv for kk, vv in arm_pca.items()
                            if kk != 'seq_means'},
                    'shrink': {kk: vv for kk, vv in arm_shr.items()
                               if kk != 'seq_means'},
                    'shrink_beta': round(beta, 4),
                    'privilege_sel_vs_pca': rp,
                    'privilege_sel_vs_pca_se_jack': rp_se,
                    'privilege_sel_vs_shrink': rs,
                    'privilege_sel_vs_shrink_se_jack': rs_se,
                    'pca_resolvable_2se': bool(abs(rp - 1.0) > 2 * rp_se),
                    'shrink_resolvable_2se': bool(abs(rs - 1.0) > 2 * rs_se)}
            hrow['k'][k] = krow
            say(f'  h{Hs} k={k}: sel/pca {rp}+-{rp_se} | sel/shrink '
                f'{rs}+-{rs_se} | sel dCE {arm_sel["mean_dce"]:+.4f} '
                f'(seq SE {arm_sel["se_seq_clustered"]:.4f})')
        sweep[f'h{Hs}'] = hrow

    # the original 8-dim span arms on the load-bearing head, now with SEs
    Hb = top3[0]
    K1 = fold0['K1'][Hb].clone()
    K2 = fold0['K2'][Hb].clone()
    Q1s, Q2s = class_span(K1), class_span(K2)
    pca1f = torch.linalg.svd(pD.sqrt()[:, None] * K1,
                             full_matrices=False).Vh[:NCLASS].T
    pca2f = torch.linalg.svd(pD.sqrt()[:, None] * K2,
                             full_matrices=False).Vh[:NCLASS].T
    arm_s8 = run_arm(f'h{Hb}_sel_span8', Hb, project_out(K1, Q1s),
                     project_out(K2, Q2s), K1, K2)
    arm_p8 = run_arm(f'h{Hb}_pca_matched8', Hb, project_out(K1, pca1f),
                     project_out(K2, pca2f), K1, K2)
    EP2b = sweep[f'h{Hb}']['E_P2_total']
    frac8 = min(1.0, (arm_s8['energy_removed'] / max(EP2b, 1e-30)) ** 0.5)
    beta8 = (1 - frac8) ** 0.5
    arm_h8 = run_arm(f'h{Hb}_shrink_matched8', Hb, K1 * beta8, K2 * beta8,
                     K1, K2)
    rp8, rp8_se = ratio_jack(arm_s8, arm_p8)
    rs8, rs8_se = ratio_jack(arm_s8, arm_h8)
    span8 = {'head': Hb,
             'sel': {kk: vv for kk, vv in arm_s8.items() if kk != 'seq_means'},
             'pca': {kk: vv for kk, vv in arm_p8.items() if kk != 'seq_means'},
             'shrink': {kk: vv for kk, vv in arm_h8.items()
                        if kk != 'seq_means'},
             'privilege_sel_vs_pca': rp8,
             'privilege_sel_vs_pca_se_jack': rp8_se,
             'privilege_sel_vs_shrink': rs8,
             'privilege_sel_vs_shrink_se_jack': rs8_se}

    # overlap confirmation number: fraction of total pattern energy carried by
    # the planted class-contrast top-2 directions in the load-bearing head
    say('  overlap confirmation: class-contrast top-2 energy fraction')
    tabs_ov = DL.DGPTables(variant='overlap')
    est_ov = DL.sample_seqs(tabs_ov, EST_N, EST_SEED)
    cnt_ov = torch.bincount(est_ov.flatten(), minlength=V).float() + 0.5
    p_ov = cnt_ov / cnt_ov.sum()
    model_ov, _ = load_arm('overlap', 'semi', tabs_ov)
    fold_ov = model_ov.fold_layer0_qk(materialize=False, device=DEV)
    pDo = p_ov.to(DEV)
    CLSo = tabs_ov.CLS.to(DEV)
    Hov = 5   # load-bearing head per qk_dgp_modular_overlap.json D4

    def cc_ov(K, k_top):
        Mn = torch.stack([(pDo[:, None] * (CLSo == kcl)[:, None] * K).sum(0)
                          / (pDo * (CLSo == kcl)).sum()
                          for kcl in range(NCLASS)], 1)
        gm = (pDo[:, None] * K).sum(0)
        return torch.linalg.svd(Mn - gm[:, None],
                                full_matrices=False).U[:, :k_top]

    @torch.no_grad()
    def pe_ov(K1a, K2a, K1ref, K2ref, seed=0):
        nb, n = ENERGY_BATCHES
        g = torch.Generator().manual_seed(seed)
        Q1f, Q2f = fold_ov['Q1'][Hov], fold_ov['Q2'][Hov]
        tot = 0.0
        for _ in range(nb):
            si = torch.multinomial(p_ov, n, replacement=True,
                                   generator=g).to(DEV)
            ti = torch.multinomial(p_ov, n, replacement=True,
                                   generator=g).to(DEV)
            tot += float((((Q1f[si] @ K1ref[ti].T / HD)
                           * (Q2f[si] @ K2ref[ti].T / HD)
                           - (Q1f[si] @ K1a[ti].T / HD)
                           * (Q2f[si] @ K2a[ti].T / HD)) ** 2).mean())
        return tot / nb

    K1o = fold_ov['K1'][Hov].clone()
    K2o = fold_ov['K2'][Hov].clone()
    C1o, C2o = cc_ov(K1o, 2), cc_ov(K2o, 2)
    E_sel2_ov = pe_ov(project_out(K1o, C1o), project_out(K2o, C2o), K1o, K2o)
    E_tot_ov = pe_ov(torch.zeros_like(K1o), torch.zeros_like(K2o), K1o, K2o)
    ov_frac = E_sel2_ov / max(E_tot_ov, 1e-30)

    F4 = {
        'design': 'per-position dCE persisted to qk_dgp_fixround_d4_dce.npz '
                  '(arrays (256 seqs, 127 positions)); SEs per token and '
                  'sequence-clustered; privilege-ratio SEs by leave-one-'
                  'sequence-out jackknife (energies fixed-seed MC constants); '
                  'dose sweep k=1..4 on the three largest whole-head-dCE '
                  'heads',
        'fold_identity_control_maxdiff': fid,
        'base_ce': round(float(base.mean()), 5),
        'whole_head_dce': {f'h{h}': head_dce[h] for h in range(NH)},
        'sweep_heads': top3,
        'dose_sweep': sweep,
        'span8_with_ses': span8,
        'overlap_confirmation': {
            'head': Hov,
            'E_class_contrast_top2': E_sel2_ov,
            'E_P2_total': E_tot_ov,
            'fraction_of_pattern_energy': ov_frac,
            'note': 'planted class-contrast top-2 directions in the overlap '
                    'model load-bearing head: fraction of total pattern '
                    'energy they carry'},
    }
    np.savez_compressed(NPZ, **npz_store)
    save({'F4': F4})
    say(f'F4 done; overlap top-2 energy fraction {ov_frac:.2e}')

say(f'fix round stages {STAGES} DONE in {time.time() - t0:.1f}s -> {OUT}')
