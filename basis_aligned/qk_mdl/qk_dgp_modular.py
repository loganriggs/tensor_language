"""PLANTED-MODULAR DGP EXPERIMENT (aggregate list #4; predictions registered in
qk_dgp_modular_predictions.json BEFORE this code was written).

Train the standard foldable bilinear architecture (tf_model.TinyBilin, vanilla,
depth 1, width 128) on the planted-modular-content DGP of qk_dgp_lang.py in two
arms (semi-planted frozen embeddings / fully-learned), then run the UNCHANGED
bilin18 ledger pipeline per attention head and score recovery against planted
truth (measurements D1-D4 of the registered predictions).

FROZEN PIPELINE RULES (no per-DGP tuning; sources in parentheses):
  * moment object: per head, the exposure(unigram)-weighted third moment of the
    source-side triple rows [k1 | k2 | v] from the exact layer-0 fold
    (BILIN18_LAYERS_0_1.md section 3; token-ledger form).
  * rank rule: eff = (sum ev)^2 / sum(ev^2) of the p-weighted mean-removed row
    covariance; r = min(16, max(2, round(eff))) (qk_iface_ledger.py).
  * dictionary: m = 2r atoms, k = 2 active (annealed 4 -> 2), SAE 6000 steps,
    batch 4096, Adam lr 3e-3, dead-atom revival every 500 (qk_iface_ledger.py,
    porting train_sae of qk_null_repair.py tick 183); sampling is BY UNIGRAM
    over tokens -- the original token-ledger form (qk_iface_ledger adapted it
    to uniform positions because its rows were positions; ours are tokens, so
    unigram sampling IS exposure).
  * CP: symmetric nonneg tensor power iteration + deflation, rank R = r,
    3 restarts (seeds 0,1,2), 8 starts, 60 iters (cp_fit of qk_stage23.py
    tick 173 via qk_iface_ledger.py).
  * null: CORRECTED transplant -- factors fit on the column-permuted-code core,
    transplanted onto the SAME real core, nonneg lambdas refit only; pass
    margin 0.15 (qk_null_repair.py tick 183 via qk_iface_ledger.py).
  * codability gate (D3): frequency-weighted two-level class coding of exact
    weight-derived spectra on the archetype write coordinates, codable at
    R^2 >= 0.8, substitution meaning gate exact-vs-coded
    (qk_coord_semantics_arch.py, BILIN18 section 34 recipe).  Class library =
    the PLANTED classes: 8 selection classes + 24 unit-membership classes.
  * equal-ablation (D4): arms scored by pattern energy removed
    E_{pxp}[(P_abl - P)^2], damage per unit energy + concentration
    (qk_equal_ablation.py, BILIN18 section 5n).
  * trainer: tiny-models defaults, stated in qk_dgp_lang.py.
  * task-learned gate: held CE minus the true law's log-loss on the SAME held
    tokens <= 0.05 nats (paired form of the registered entropy-margin gate;
    the MC entropy itself is also reported).  An arm that fails is stopped --
    no recovery verdict.

POSITIVE CONTROLS (gate, run first; 2026-08-10 amendment): the exact
third-moment tensor implied by each variant's DGP tables (uniform token
weights, no training, no sampling) in the 24-unit code space, through the same
CP + corrected-null machinery.  TWO VARIANTS (--variant, see qk_dgp_lang):
  * identifiable (D1-D4 verdict arm): gate = Hungarian-matched cosine vs the
    planted units >= 0.99 mean AND null margin >= 0.15 (the original bar).
  * overlap (realism arm, calibration finding F0): its exact tensor is not a
    pure rank-24 object, so the gate is that the frozen machinery REPRODUCES
    THE ANALYTIC-TENSOR OPTIMUM (matched cosine vs the heavy-opt solution
    >= 0.99) AND margin >= 0.15; recovery is scored against the noiseless
    ceiling (23/24 units at >= 0.9), reported alongside the vs-units numbers.
Full-mode gate failure for the SELECTED variant writes the JSON and exits.
--arm {semi,learned,both} selects arms; per-arm invocations of one variant
merge into the same qk_dgp_modular_<variant>.json.  --controls-only runs both
variants' controls, writes qk_dgp_controls.json, and exits (chain pre-gate).

--smoke (or QK_SMOKE=1): the analytic control at FULL fidelity, a few hundred
training steps of the semi-planted arm, one head's pipeline on a small sample;
writes qk_dgp_modular_smoke.json only.  In smoke a failed gate is reported but
does not stop the later stages, which exist to exercise the machinery (no
verdicts).  (Measured: even 400 steps pass the semi-arm task gate -- the law
is very learnable with frozen planted embeddings.)
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
import tf_model as TFM  # noqa: E402  (path added by qk_dgp_lang)

SMOKE = ('--smoke' in sys.argv) or (os.environ.get('QK_SMOKE') == '1')
# build-verification hook: QK_DGP_TESTALL=1 with --smoke additionally runs the
# learned arm and the D3/D4 sections at smoke scale (all code paths exercised
# once before a full run).  Never changes full-mode behavior.
TESTALL = os.environ.get('QK_DGP_TESTALL') == '1'


def _argval(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


# 2026-08-10 amendment (see qk_dgp_modular_predictions.json amendment block):
# two DGP variants.  'identifiable' carries the D1-D4 verdicts at the original
# bar; 'overlap' is the realism arm, scored against its noiseless ceiling.
VARIANT = os.environ.get('QK_DGP_VARIANT') or _argval('--variant', 'identifiable')
assert VARIANT in ('identifiable', 'overlap'), VARIANT
ARM_SEL = _argval('--arm', 'both')
assert ARM_SEL in ('semi', 'learned', 'both'), ARM_SEL
CONTROLS_ONLY = '--controls-only' in sys.argv
torch.manual_seed(0)
OUT = (f'{QK}/qk_dgp_modular_smoke.json' if SMOKE
       else f'{QK}/qk_dgp_modular_{VARIANT}.json')

# device rule: smoke must be runnable beside a busy GPU -- take CUDA only if
# >= 4000 MiB are free, else CPU (the model is tiny; CPU smoke is fine).
if torch.cuda.is_available():
    free_mib = torch.cuda.mem_get_info()[0] // 2**20
    DEV = 'cuda' if free_mib >= 4000 else 'cpu'
else:
    free_mib, DEV = 0, 'cpu'

STEPS = 400 if SMOKE else 8000
HELD_N = 300 if SMOKE else 1500
EST_N = 300 if SMOKE else 1500
SAE_STEPS = 600 if SMOKE else 6000
SAE_BATCH = 2048 if SMOKE else 4096
SAE_LR = 3e-3
RESTARTS = 3
MARGIN = 0.15
GATE_COS = 0.99
TASK_GATE_NATS = 0.05
R2_THRESH = 0.8
D3_HELD = 32 if SMOKE else 256
D4_HELD = 32 if SMOKE else 256
ENERGY_BATCHES = (2, 1024) if SMOKE else (8, 4096)
HD = 16
NH = WS // HD
TRAIN_SEED, HELD_SEED, EST_SEED = 11, 12, 13

t0 = time.time()


def say(msg):
    print(f'[{time.time() - t0:7.1f}s] {msg}', flush=True)


# ----------------------------------------------------------------------------
# CP machinery -- verbatim ports from qk_iface_ledger.py (itself porting
# qk_stage23.py tick 173 cp_fit and qk_null_repair.py tick 183 eval_on_core).
# ----------------------------------------------------------------------------
def cp_fit(core_raw, R, seed, n_starts=8, iters=60):
    """Symmetric nonneg CP via tensor power iteration + deflation."""
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
    """Nonneg lambda refit on THIS (unit-norm) core for fixed factors."""
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
    """Matched-component |cos| across restarts (qk_stage23.py port)."""
    vals = []
    for i in range(len(Us)):
        for j in range(i + 1, len(Us)):
            C = (Us[i].T @ Us[j]).abs()
            vals.append(float(C.max(1).values.mean()))
    return sum(vals) / len(vals)


def train_sae(Z, m_atoms, k_code, p_cpu, steps, batch, lr=SAE_LR, seed=0):
    """Nonneg k-sparse coder with k-annealing + dead-atom revival -- port of
    train_sae in qk_iface_ledger.py (from qk_null_repair.py tick 183), with
    two STATED modifications: (1) sampling is by UNIGRAM over token rows
    (multinomial on p) -- the original token-ledger form, since here rows are
    tokens and p is exposure; (2) the normalized dictionary is returned along
    with the codes (needed to pull components back to embedding space)."""
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
    """Exposure-weighted third moment in code space: sum_t p_t s_t x3."""
    return torch.einsum('t,ta,tb,tc->abc', p.to(S.device), S, S, S)


def hungarian_vs_units(C):
    """C: (ncomp, 24) |cos| matrix.  Hungarian max matching; returns the
    24-vector of matched cosines (0 for unmatched units) + assignment."""
    ri, ci = linear_sum_assignment(-C.numpy())
    cos24 = np.zeros(C.shape[1])
    assign = {}
    for r, c in zip(ri, ci):
        cos24[c] = float(C[r, c])
        assign[int(c)] = int(r)
    return cos24, assign


# ----------------------------------------------------------------------------
# A. ANALYTIC POSITIVE CONTROLS (GATE) -- the exact third moment implied by
# each variant's DGP tables, in the 24-unit code space, uniform token weights,
# no sampling.  Per-variant gate rules (2026-08-10 amendment):
#   identifiable: frozen CP matches the PLANTED UNITS at mean cos >= 0.99
#                 (the original bar) AND null margin >= 0.15.
#   overlap:      frozen CP REPRODUCES THE ANALYTIC-TENSOR OPTIMUM (matched
#                 cos vs the heavy-opt solution >= 0.99) AND margin >= 0.15;
#                 the units comparison is reported as the noiseless CEILING
#                 the trained-model recovery is scored against (F0).
# ----------------------------------------------------------------------------
say(f'=== planted-modular DGP experiment ({"SMOKE" if SMOKE else "FULL"}) '
    f'variant={VARIANT} arms={ARM_SEL} on {DEV} (free {free_mib} MiB) ===')
tabs = DL.DGPTables(variant=VARIANT)


def analytic_control(tabs_v):
    chi = tabs_v.CHI                                    # (V, 24)
    core_a = torch.einsum('ta,tb,tc->abc', chi, chi, chi) / V
    core_an = core_a / core_a.norm()
    Us_a, rels_a = [], []
    for sd in range(RESTARTS):
        U = cp_fit(core_an, NUNITS, sd)
        rel, _ = eval_on_core(core_an, U)
        Us_a.append(U)
        rels_a.append(rel)
    best_a = int(np.argmin(rels_a))
    U_a = Us_a[best_a]
    rel_a = rels_a[best_a]
    stab_a = stability(Us_a)
    # components live in the code space where the planted units ARE the axes
    cos24_a, _ = hungarian_vs_units(U_a.T.abs())
    # corrected transplant null: permute each unit column of chi over tokens
    gp = torch.Generator().manual_seed(7)
    chi_null = chi.clone()
    for u in range(NUNITS):
        chi_null[:, u] = chi_null[torch.randperm(V, generator=gp), u]
    core_null = torch.einsum('ta,tb,tc->abc',
                             chi_null, chi_null, chi_null) / V
    U_null_a = cp_fit(core_null / core_null.norm(), NUNITS, 0)
    null_on_real_a, _ = eval_on_core(core_an, U_null_a)
    margin_a = null_on_real_a - rel_a
    # diagnostics + the analytic-optimum reference: oracle pure-unit factors
    # with refit lambdas, and a heavy-optimization solution (32 starts, 400
    # iters -- the best available stand-in for the tensor's true optimum; the
    # pipeline itself stays at the frozen 8/60)
    rel_oracle, _ = eval_on_core(core_an, torch.eye(NUNITS))
    U_heavy = cp_fit(core_an, NUNITS, 0, n_starts=32, iters=400)
    cos24_h, _ = hungarian_vs_units(U_heavy.T.abs())
    # does the frozen machinery reproduce the analytic optimum?
    Cfh = (U_a.T @ U_heavy).abs()
    ri_, ci_ = linear_sum_assignment(-Cfh.numpy())
    repro_mean = float(Cfh.numpy()[ri_, ci_].mean())
    ctrl = {
        'variant': tabs_v.variant,
        'construction': 'core = (1/V) sum_t chi_t^{x3} in the 24-unit code '
                        'space; uniform token weights (no sampling, no '
                        'training)',
        'matched_cos_mean': round(float(cos24_a.mean()), 4),
        'matched_cos_min': round(float(cos24_a.min()), 4),
        'matched_cosines': [round(float(x), 4) for x in cos24_a],
        'frac_ge_0.99': round(float((cos24_a >= 0.99).mean()), 4),
        'frac_ge_0.9': round(float((cos24_a >= 0.9).mean()), 4),
        'real_relerr': round(rel_a, 4),
        'null_on_real': round(null_on_real_a, 4),
        'margin': round(margin_a, 4), 'restart_stability': round(stab_a, 4),
        'restart_relerrs': [round(x, 4) for x in rels_a],
        'frozen_reproduces_analytic_optimum_cos': round(repro_mean, 4),
        'diagnostics': {
            'oracle_pure_units_relerr': round(rel_oracle, 4),
            'heavy_opt_probe_matched_cos_mean': round(float(cos24_h.mean()), 4),
            'heavy_opt_probe_matched_cos_min': round(float(cos24_h.min()), 4),
            'note': 'if oracle_pure_units_relerr > real_relerr, the best '
                    'rank-24 nonneg CP of the DGP-implied tensor is NOT the '
                    'planted units (co-occurrence cross-moments of '
                    'overlapping bundles) -- an identifiability property of '
                    'the object, not a solver failure (calibration finding '
                    'F0); the heavy-opt solution (32 starts, 400 iters) is '
                    'the analytic-optimum reference.'},
    }
    if tabs_v.variant == 'identifiable':
        ok = bool(cos24_a.mean() >= GATE_COS and margin_a >= MARGIN)
        ctrl['gate_rule'] = ('mean matched cos vs planted units >= '
                             f'{GATE_COS} and margin >= {MARGIN}')
    else:
        ok = bool(repro_mean >= GATE_COS and margin_a >= MARGIN)
        ctrl['gate_rule'] = ('frozen solution reproduces the analytic-tensor '
                             f'optimum at matched cos >= {GATE_COS} and '
                             f'margin >= {MARGIN} (units comparison = '
                             'noiseless ceiling, not the gate; F0)')
    ctrl['pass'] = ok
    say(f'  {tabs_v.variant}: vs-units mean {cos24_a.mean():.4f} min '
        f'{cos24_a.min():.4f} | reproduces-optimum {repro_mean:.4f} | real '
        f'{rel_a:.4f} null-on-real {null_on_real_a:.4f} margin {margin_a:+.4f}'
        f' | stability {stab_a:.4f} -> {"PASS" if ok else "FAIL"}')
    return ctrl, U_a, cos24_a


say('A. analytic positive controls (both variants; per-variant gate rules)')
CTRLS, U_AN, COS24_AN = {}, {}, {}
for vn in ('identifiable', 'overlap'):
    tv = tabs if vn == VARIANT else DL.DGPTables(variant=vn)
    CTRLS[vn], U_AN[vn], COS24_AN[vn] = analytic_control(tv)
gate_ctrl = CTRLS[VARIANT]['pass']

if CONTROLS_ONLY:
    json.dump({vn: CTRLS[vn] for vn in CTRLS},
              open(f'{QK}/qk_dgp_controls.json', 'w'), indent=2)
    say(f'controls-only mode: wrote qk_dgp_controls.json '
        f'(identifiable {"PASS" if CTRLS["identifiable"]["pass"] else "FAIL"}'
        f', overlap {"PASS" if CTRLS["overlap"]["pass"] else "FAIL"})')
    sys.exit(0)

result = {
    'mode': 'smoke' if SMOKE else 'full',
    'variant': VARIANT,
    'testall': TESTALL,
    'device': DEV,
    'config': {
        'dgp': {'V': V, 'T': T, 'n_classes': NCLASS, 'n_units': NUNITS,
                'variant': VARIANT,
                'gamma': DL.GAMMA, 'bw_std': DL.BW_STD,
                'sigma': 'derangement k -> (k+3) mod 8',
                'table_seed': DL.TABLE_SEED,
                'train_seed': TRAIN_SEED, 'held_seed': HELD_SEED,
                'est_seed': EST_SEED,
                'law': 'p(next=w) = softmax(gamma * <A_i, chi(w)> + b_w); '
                       'A_i = per-unit counts over ALL earlier tokens of class '
                       'sigma(c(t_i)); see qk_dgp_lang.py docstring incl. '
                       'stated deviations (sum- not most-recent-selection; '
                       'in-softmax unigram floor; always-on coordinate)'},
        'model': 'tf_model.TinyBilin vanilla depth 1 width 128 (8 heads x '
                 'hd 16, hidden 512), T 128, vocab 512; reused, not '
                 'reimplemented (fold machinery + identity gates for free)',
        'trainer': f'tiny-models defaults: Muon 0.02 (nesterov 0.95, 5-step '
                   f'NS), AdamW 0.004 b(0.9,0.95) wd 0.1 on embedding, warmup '
                   f'250, cosine, clip 1.0, batch 16, {STEPS} steps, fresh '
                   f'single-epoch iid data, fp32 held eval',
        'pipeline_rules': {
            'moment': 'per head, unigram-weighted third moment of source-side '
                      'triple rows [k1|k2|v] from the exact layer-0 fold',
            'rank_rule': 'r = min(16, max(2, round(eff))), eff = participation '
                         'ratio of p-weighted mean-removed row covariance',
            'dict_rule': f'm = 2r, k = 2 (annealed 4->2), SAE {SAE_STEPS} '
                         f'steps batch {SAE_BATCH} lr {SAE_LR}, unigram token '
                         f'sampling, revival every 500',
            'cp': f'TPI+deflation, R = r, {RESTARTS} restarts, 8 starts, 60 iters',
            'null': 'corrected transplant (fit on permuted-code core, '
                    'transplant onto SAME real core, nonneg lambdas refit '
                    f'only), margin {MARGIN}',
            'task_gate': f'held CE - true-law log-loss on same tokens <= '
                         f'{TASK_GATE_NATS} (paired form; MC entropy also '
                         'reported)',
            'codability': f'R2 >= {R2_THRESH}, frequency-weighted; class '
                          'library = 8 planted selection classes + 24 unit-'
                          'membership classes; substitution meaning gate',
            'equal_ablation': 'pattern-energy-matched arms, damage per unit '
                              'energy + concentration, one load-bearing head',
        },
    },
    'analytic_controls': CTRLS,
    'gates': {f'G0_analytic_{vn}': ('PASS' if CTRLS[vn]['pass'] else 'FAIL')
              for vn in CTRLS},
}
# per-arm invocations of the same variant share one JSON: seed from the
# existing file so semi/learned cells merge instead of clobbering (data,
# tables and controls are deterministic, so overlapping keys are identical)
if (not SMOKE) and os.path.exists(OUT):
    try:
        prev = json.load(open(OUT))
        result['gates'] = {**prev.get('gates', {}), **result['gates']}
        for k in ('arms', 'per_head', 'D1', 'D2', 'D3', 'D4',
                  'dgp_reference'):
            if k in prev:
                result[k] = prev[k]
    except Exception as e:
        say(f'  (could not merge existing {OUT}: {e} -- starting fresh)')
json.dump(result, open(OUT, 'w'), indent=2)
if not gate_ctrl:
    result['verdict'] = (f'GATE FAILED for variant {VARIANT} under its rule '
                         f'({CTRLS[VARIANT]["gate_rule"]}). The pipeline '
                         'cannot certify anything downstream for this '
                         'variant. STOP. See analytic_controls.diagnostics.')
    json.dump(result, open(OUT, 'w'), indent=2)
    if not SMOKE:
        say(f'G0 analytic control FAILED for {VARIANT} -- stopping.')
        sys.exit(1)
    say('G0 analytic control FAILED -- SMOKE continues past the gate to '
        'exercise the training + pipeline machinery (no verdicts).')
else:
    say(f'G0 gate OPEN for variant {VARIANT}')

# ----------------------------------------------------------------------------
# B. Data + entropy references
# ----------------------------------------------------------------------------
say(f'B. sampling corpus: {STEPS * DL.BATCH} train / {HELD_N} held / '
    f'{EST_N} est sequences (T={T})')
train_seqs = DL.sample_seqs(tabs, STEPS * DL.BATCH, TRAIN_SEED)
held_seqs = DL.sample_seqs(tabs, HELD_N, HELD_SEED)
est_seqs = DL.sample_seqs(tabs, EST_N, EST_SEED)
ref = DL.entropy_reference(tabs, held_seqs)
say(f'  DGP entropy (MC over held prefixes): {ref["entropy_mc_mean"]:.4f} '
    f'nats/token; true-law log-loss on held tokens: '
    f'{ref["true_law_logloss_mean"]:.4f}')
result['dgp_reference'] = ref
cnt = torch.bincount(est_seqs.flatten(), minlength=V).float() + 0.5
p_est = (cnt / cnt.sum())                              # unigram exposure (V,)
json.dump(result, open(OUT, 'w'), indent=2)

# ----------------------------------------------------------------------------
# C. Train arms + task-learned gate
# ----------------------------------------------------------------------------
ARMS = ['semi'] if (SMOKE and not TESTALL) else ['semi', 'learned']
if ARM_SEL != 'both':
    ARMS = [a for a in ARMS if a == ARM_SEL]
models, arm_logs, task_pass = {}, {}, {}
for arm in ARMS:
    ck = f'{QK}/qk_dgp_{VARIANT}_{arm}.pt'
    model = DL.make_arm_model(tabs, arm, seed=0, device=DEV)
    if (not SMOKE) and os.path.exists(ck):
        st = torch.load(ck, map_location=DEV)
        if st['log']['steps'] == STEPS:
            model.load_state_dict(st['state_dict'])
            arm_logs[arm] = st['log']
            say(f'C. {arm}: loaded checkpoint {ck} '
                f'(held CE {st["log"]["final_held_ce"]:.4f})')
        else:
            arm_logs[arm] = None
    if arm not in arm_logs or arm_logs[arm] is None:
        say(f'C. training {arm} arm ({STEPS} steps)')
        arm_logs[arm] = DL.train_arm(model, train_seqs, held_seqs, STEPS, DEV,
                                     name=arm)
        if not SMOKE:
            torch.save({'state_dict': model.state_dict(),
                        'log': arm_logs[arm]}, ck)
    models[arm] = model.eval()
    ce = arm_logs[arm]['final_held_ce']
    gap_paired = ce - ref['true_law_logloss_mean']
    gap_entropy = ce - ref['entropy_mc_mean']
    ok = bool(gap_paired <= TASK_GATE_NATS)
    task_pass[arm] = ok
    arm_logs[arm]['task_gate'] = {
        'held_ce': ce, 'gap_vs_true_law_paired': round(gap_paired, 5),
        'gap_vs_entropy_mc': round(gap_entropy, 5),
        'threshold': TASK_GATE_NATS, 'pass': ok}
    result['gates'][f'G_task_{arm}'] = 'PASS' if ok else 'FAIL'
    say(f'  {arm}: held CE {ce:.4f} | paired gap {gap_paired:+.4f} '
        f'(vs entropy {gap_entropy:+.4f}) -> task gate '
        f'{"PASS" if ok else "FAIL"}')
result['arms'] = {**result.get('arms', {}),
                  **{a: arm_logs[a] for a in ARMS}}
json.dump(result, open(OUT, 'w'), indent=2)
if SMOKE and not task_pass['semi']:
    say('  (smoke: task-gate failure at 400 steps is expected; pipeline '
        'stage still runs to exercise the machinery)')

# ----------------------------------------------------------------------------
# D. Per-head ledger pipeline + D1/D2 scoring
# ----------------------------------------------------------------------------
def run_head_pipeline(model, h, p_cpu, dev):
    """Frozen ledger pipeline on one head; returns per-head dict + components
    as minimal-norm value-path preimages in embedding space."""
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
    # components -> value-slice -> minimal-norm preimage in embedding space
    Wv = model.h[0].c_v.weight.detach().float()[h * HD:(h + 1) * HD].to(dev)
    G = Wv @ Wv.T
    comps = []
    order = lam_best.argsort(descending=True)
    for ri in order.tolist():
        if float(lam_best[ri]) <= 0:
            continue
        d48 = (U_best[:, ri].to(Dn.device) @ Dn)          # (48,)
        vdir = d48[2 * HD:]                               # value slice (16,)
        if float(vdir.norm()) < 1e-8:
            continue
        e_dir = Wv.T @ torch.linalg.solve(G, vdir.to(dev))  # min-norm preimage
        comps.append({'lambda': round(float(lam_best[ri]), 4),
                      'vdir': vdir.cpu(), 'e_dir': e_dir.cpu()})
    return {'eff': round(eff, 2), 'r': r, 'm_atoms': m_atoms,
            'real_fit': round(real_fit, 4),
            'null_on_real': round(null_on_real, 4),
            'margin': round(margin, 4), 'm1_pass': bool(margin >= MARGIN),
            'restart_stability': round(stab, 3),
            'restart_relerrs': [round(x, 4) for x in rels]}, comps


def score_components_vs_units(comp_dirs, content_map=None):
    """comp_dirs: list of embedding-space directions (Ws,).  Semi arm:
    restrict to the content block.  Learned arm: apply the fitted linear map
    `content_map` (64, Ws) instead.  Hungarian-match |cos| vs the 24 units."""
    rows, fracs = [], []
    for e in comp_dirs:
        if content_map is None:
            ec = e[DL.CONTENT_SLICE]
        else:
            ec = content_map @ e
        fr = float(ec.norm() ** 2 / e.norm().clamp_min(1e-12) ** 2) \
            if content_map is None else None
        n = float(ec.norm())
        rows.append(ec / max(n, 1e-12))
        fracs.append(fr)
    if not rows:
        return np.zeros(NUNITS), {}, fracs
    Cm = (torch.stack(rows) @ tabs.UNITS).abs()           # (ncomp, 24)
    cos24, assign = hungarian_vs_units(Cm)
    return cos24, assign, fracs


HEADS = [0] if SMOKE else list(range(NH))
head_results, arm_comps = {}, {}
for arm in ARMS:
    if not SMOKE and not task_pass[arm]:
        say(f'D. {arm}: task gate FAILED -- arm stopped, no recovery verdict')
        head_results[arm] = {'skipped': 'task gate failed'}
        continue
    say(f'D. ledger pipeline on {arm} arm, heads {HEADS}')
    head_results[arm] = {}
    arm_comps[arm] = []
    for h in HEADS:
        hres, comps = run_head_pipeline(models[arm], h, p_est.cpu(), DEV)
        head_results[arm][f'h{h}'] = hres
        for c in comps:
            arm_comps[arm].append((h, c))
        say(f'  {arm} h{h}: eff {hres["eff"]} r={hres["r"]} | real '
            f'{hres["real_fit"]} null-on-real {hres["null_on_real"]} margin '
            f'{hres["margin"]:+.4f} {"PASS" if hres["m1_pass"] else "fail"} | '
            f'stability {hres["restart_stability"]} | {len(comps)} components')
result['per_head'] = {**result.get('per_head', {}), **head_results}
json.dump(result, open(OUT, 'w'), indent=2)

# D1: semi arm, components pooled across heads, cosine in the content block.
# Per the 2026-08-10 amendment: the identifiable variant carries the verdict
# at the original bar; the overlap variant reports the same numbers AGAINST
# ITS NOISELESS CEILING (the analytic-tensor optimum, finding F0), plus the
# match against the analytic-optimum directions themselves.
if 'semi' in arm_comps:
    dirs = [c['e_dir'] for _, c in arm_comps['semi']]
    cos24, assign, fracs = score_components_vs_units(dirs)
    ceil24 = COS24_AN[VARIANT]
    ceiling = round(float((ceil24 >= 0.9).mean()), 4)
    frac09 = round(float((cos24 >= 0.9).mean()), 4)
    result['D1'] = {
        'arm': 'semi', 'variant': VARIANT, 'n_components_pooled': len(dirs),
        'matched_cosines': [round(float(x), 4) for x in cos24],
        'frac_units_matched_cos_ge_0.9': frac09,
        'matched_cos_mean': round(float(cos24.mean()), 4),
        'noiseless_ceiling_frac_ge_0.9': ceiling,
        'frac_relative_to_ceiling': (round(frac09 / ceiling, 4)
                                     if ceiling > 0 else None),
        'content_energy_frac_per_component': [
            round(f, 3) for f in fracs if f is not None],
        'note': 'Hungarian match of pooled per-head CP components (value-'
                'slice minimal-norm preimage, content block) vs 24 planted '
                'units' + (' -- SMOKE: reduced heads/steps; not a D1 verdict'
                           if SMOKE else ''),
    }
    # match vs the analytic-optimum directions in content space (primary
    # reference for the overlap variant; reported for both)
    A64 = tabs.UNITS @ U_AN[VARIANT]                     # (64, n_analytic)
    A64 = A64 / A64.norm(dim=0, keepdim=True).clamp_min(1e-12)
    rows = torch.stack([(e[DL.CONTENT_SLICE]
                         / e[DL.CONTENT_SLICE].norm().clamp_min(1e-12))
                        for e in dirs])
    Ca = (rows @ A64).abs()
    cosA, _ = hungarian_vs_units(Ca)
    result['D1']['vs_analytic_optimum'] = {
        'matched_cosines': [round(float(x), 4) for x in cosA],
        'frac_ge_0.9': round(float((cosA >= 0.9).mean()), 4),
        'matched_cos_mean': round(float(cosA.mean()), 4)}
    say(f'D1 (semi, {VARIANT}): frac units matched at cos>=0.9: {frac09} '
        f'(ceiling {ceiling}, vs-analytic-optimum '
        f'{result["D1"]["vs_analytic_optimum"]["frac_ge_0.9"]}) over '
        f'{len(dirs)} components')

# D2: learned arm -- fit linear map learned-embedding -> planted content space
if 'learned' in arm_comps:
    Xl = F.rms_norm(models['learned'].wte.weight.detach().float().cpu(), (WS,))
    W_ = p_est
    A = Xl.T @ (W_[:, None] * Xl)
    lam_r = 1e-3 * float(torch.diagonal(A).sum()) / WS
    ML = torch.linalg.solve(A + lam_r * torch.eye(WS),
                            Xl.T @ (W_[:, None] * tabs.B)).T   # (64, Ws)
    pred = Xl @ ML.T
    ss_res = float((W_[:, None] * (pred - tabs.B) ** 2).sum())
    mu_b = (W_[:, None] * tabs.B).sum(0)
    ss_tot = float((W_[:, None] * (tabs.B - mu_b) ** 2).sum())
    map_r2 = 1 - ss_res / max(ss_tot, 1e-12)
    dirs = [c['e_dir'] for _, c in arm_comps['learned']]
    cos24_l, _, _ = score_components_vs_units(dirs, content_map=ML)
    d1f = result.get('D1', {}).get('frac_units_matched_cos_ge_0.9')
    d2f = round(float((cos24_l >= 0.9).mean()), 4)
    result['D2'] = {
        'arm': 'learned', 'n_components_pooled': len(dirs),
        'map_fit': 'unigram-weighted ridge LS from rms-normed learned '
                   f'embedding rows to planted bundle vectors, ridge = 1e-3 * '
                   f'trace/Ws (fit on estimation split); weighted R2 = '
                   f'{round(map_r2, 4)}',
        'matched_cosines': [round(float(x), 4) for x in cos24_l],
        'frac_units_matched_cos_ge_0.9': d2f,
        'matched_cos_mean': round(float(cos24_l.mean()), 4),
        'points_fewer_than_D1': (round(100 * (d1f - d2f), 1)
                                 if d1f is not None else None),
    }
    say(f'D2 (learned): frac units matched at cos>=0.9: {d2f} '
        f'(map R2 {map_r2:.3f})')
json.dump(result, open(OUT, 'w'), indent=2)

# ----------------------------------------------------------------------------
# E. D3 -- section-34 codability gate on the semi arm
# ----------------------------------------------------------------------------
def forward_subst(model, idx, spectra=None, basis=None):
    """Vanilla depth-1 TinyBilin forward with layer-0 attention writes
    optionally replaced by pattern-weighted per-key SPECTRA on per-head write
    BASES (the section-34 substitution meaning gate, qk_coord_semantics_arch
    recipe).  spectra=None reproduces the model exactly (identity control)."""
    cfg = model.cfg
    Ws, Dc, H, hd = model.Ws, model.Dc, cfg.n_heads, cfg.head_dim
    B, Tq = idx.shape
    cos = model.cos[None, :Tq, None, :]
    sin = model.sin[None, :Tq, None, :]
    mask = model.mask[:Tq, :Tq]
    blk = model.h[0]
    e = F.rms_norm(model.wte(idx), (Ws,))
    x = e
    hn = F.rms_norm(x, (Ws,))

    def qk(lin):
        z = lin(hn).view(B, Tq, H, hd)
        if cfg.qk_norm:
            z = F.rms_norm(z, (hd,))
        return TFM.apply_rot(z, cos, sin)

    q, k = qk(blk.c_q), qk(blk.c_k)
    q2, k2 = qk(blk.c_q2), qk(blk.c_k2)
    v = blk.c_v(hn).view(B, Tq, H, hd)
    s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / hd
    s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / hd
    pat = (s1 * s2).masked_fill(~mask, 0.0)
    if spectra is None:
        y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, Tq, Dc)
        aw = blk.c_proj(y)
    else:
        aw = torch.zeros(B, Tq, Ws, device=idx.device)
        for h in range(H):
            sp = spectra[h][idx.reshape(-1)].view(B, Tq, -1)
            Z = torch.einsum('bqk,bkc->bqc', pat[:, h], sp)
            aw = aw + Z @ basis[h].T
    x = x + aw
    xn = F.rms_norm(x, (Ws,))
    x = x + blk.Down(blk.Left(xn) * blk.Right(xn)) + blk.Down_bias
    xf = F.rms_norm(x, (Ws,))
    return 30 * torch.tanh(xf @ model.wte.weight.t() / 30)


@torch.no_grad()
def percetok(model, spectra, basis, seqs, batch=8):
    ces = []
    for i in range(0, len(seqs), batch):
        b = seqs[i:i + batch].to(DEV)
        lg = forward_subst(model, b[:, :-1], spectra, basis)
        ce = F.cross_entropy(lg.float().reshape(-1, V), b[:, 1:].reshape(-1),
                             reduction='none')
        ces.append(ce.cpu())
    return torch.cat(ces)


if (TESTALL or not SMOKE) and task_pass.get('semi') and 'semi' in arm_comps:
    say('E. D3 codability gate (section-34 recipe) on semi arm')
    model = models['semi']
    with torch.no_grad():
        # substitution identity control: spectra=None path == model forward
        idc = held_seqs[:4, :-1].to(DEV)
        dmax = float((forward_subst(model, idc) - model(idc)).abs().max())
    say(f'  substitution identity control: max |diff| {dmax:.2e}')
    fold = model.fold_layer0_qk(materialize=False, device=DEV)
    cw = model.h[0].c_proj.weight.detach().float().to(DEV)   # (Ws, Dc)
    # per-head archetype write bases from the CP components
    BASIS, SPEC = [], []
    nc_per_head = []
    for h in range(NH):
        vdirs = [c['vdir'].to(DEV) for hh, c in arm_comps['semi'] if hh == h]
        wtab = fold['Vv'][h].to(DEV) @ cw[:, h * HD:(h + 1) * HD].T  # (V, Ws)
        if vdirs:
            Wdirs = torch.stack([cw[:, h * HD:(h + 1) * HD] @ (vd / vd.norm())
                                 for vd in vdirs], 1)      # (Ws, NC)
            Qh, _ = torch.linalg.qr(Wdirs)
        else:  # fallback: top-1 PCA of the p-weighted write table (stated)
            Qh = torch.linalg.svd(
                (p_est.to(DEV).sqrt()[:, None] * wtab),
                full_matrices=False).Vh[:1].T
        BASIS.append(Qh.contiguous())
        SPEC.append(wtab @ Qh)                              # (V, NC)
        nc_per_head.append(Qh.shape[1])
    # full-rank exact-spectra control: basis = full write column space ->
    # substitution must reproduce the model exactly
    BF = [torch.linalg.qr(cw[:, h * HD:(h + 1) * HD]).Q for h in range(NH)]
    SF = [fold['Vv'][h].to(DEV) @ cw[:, h * HD:(h + 1) * HD].T @ BF[h]
          for h in range(NH)]
    ce_real = percetok(model, None, None, held_seqs[:D3_HELD])
    ce_fullrank = percetok(model, SF, BF, held_seqs[:D3_HELD])
    full_ctrl = float((ce_fullrank - ce_real).mean())
    say(f'  full-rank spectra control dCE {full_ctrl:+.6f} (should be ~0)')
    # class library: the planted classes
    CLS_LIB = {}
    for kcl in range(NCLASS):
        CLS_LIB[f'class_{kcl}'] = (tabs.CLS == kcl)
    for u in range(NUNITS):
        CLS_LIB[f'unit_{u}'] = (tabs.CHI[:, u] > 0)
    w = p_est.clone().clamp_min(1e-9)
    rows_tab, codable, r2s = [], 0, []
    SHAT = [s.clone() for s in SPEC]
    for h in range(NH):
        Sh = SPEC[h].cpu()
        for c in range(nc_per_head[h]):
            y = Sh[:, c]
            mu_ = float((w * y).sum() / w.sum())
            var = float((w * (y - mu_) ** 2).sum())
            best = (None, 0.0, 0.0, mu_)
            for nm, mask_ in CLS_LIB.items():
                mk = mask_.float()
                wm = float((w * mk).sum())
                if wm < 1e-6:
                    continue
                a_in = float((w * mk * y).sum() / wm)
                w0 = float((w * (1 - mk)).sum())
                a_out = float((w * (1 - mk) * y).sum() / w0)
                pred_ = mk * a_in + (1 - mk) * a_out
                r2 = 1 - float((w * (y - pred_) ** 2).sum()) / max(var, 1e-12)
                if r2 > best[1]:
                    best = (nm, r2, a_in, a_out)
            rows_tab.append({'head': h, 'coord': c, 'class': best[0],
                             'r2': round(best[1], 3)})
            r2s.append(best[1])
            if best[1] >= R2_THRESH:
                codable += 1
                mk = CLS_LIB[best[0]].float().to(DEV)
                SHAT[h][:, c] = mk * best[2] + (1 - mk) * best[3]
    ce_exact = percetok(model, SPEC, BASIS, held_seqs[:D3_HELD])
    ce_coded = percetok(model, SHAT, BASIS, held_seqs[:D3_HELD])
    total = sum(nc_per_head)
    from collections import Counter
    top_cls = Counter(r['class'] for r in rows_tab
                      if r['r2'] >= R2_THRESH).most_common(12)
    sel_codable = sum(1 for r in rows_tab if r['r2'] >= R2_THRESH
                      and r['class'].startswith('class_'))
    unit_codable = sum(1 for r in rows_tab if r['r2'] >= R2_THRESH
                       and r['class'].startswith('unit_'))
    d_exact = float((ce_exact - ce_real).mean())
    d_coded = float((ce_coded - ce_real).mean())
    d_gate = float((ce_coded - ce_exact).mean())
    se_gate = float((ce_coded - ce_exact).std()
                    / np.sqrt(len(ce_exact)))
    result['D3'] = {
        'arm': 'semi', 'total_coords': total, 'codable_n': codable,
        'codable_frac': round(codable / max(total, 1), 3),
        'selection_class_codable': sel_codable,
        'unit_class_codable': unit_codable,
        'median_class_r2': round(float(np.median(r2s)), 3),
        'codable_classes_top': top_cls,
        'substitution_identity_control_maxdiff': dmax,
        'fullrank_spectra_control_dce': round(full_ctrl, 6),
        'exact_spectra_dce': round(d_exact, 5),
        'coded_spectra_dce': round(d_coded, 5),
        'meaning_gate_coded_minus_exact': {'dCE': round(d_gate, 5),
                                           'SE': round(se_gate, 6)},
        'coord_table': rows_tab,
    }
    say(f'D3: codable {codable}/{total} (selection {sel_codable}, unit '
        f'{unit_codable}); median R2 {np.median(r2s):.3f}; meaning gate '
        f'coded-exact dCE {d_gate:+.5f} (SE {se_gate:.6f})')
    json.dump(result, open(OUT, 'w'), indent=2)

# ----------------------------------------------------------------------------
# F. D4 -- section-5n equal-ablation on the planted selection directions
# ----------------------------------------------------------------------------
if (TESTALL or not SMOKE) and task_pass.get('semi'):
    say('F. D4 equal-ablation (section-5n design) on semi arm')
    model = models['semi']
    fold0 = model.fold_layer0_qk(materialize=False, device=DEV)
    mlpT = [model.fold_mlp(0, device=DEV).to(DEV)]
    d4_seqs = held_seqs[:D4_HELD]

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

    base = fold_ce(fold0)
    with torch.no_grad():
        idc = d4_seqs[:4, :-1].to(DEV)
        fid = float((model.fold_forward(idc, fold=fold0, mlp_tensors=mlpT)
                     - model(idc)).abs().max())
    say(f'  fold identity control: max |logit diff| {fid:.2e}; base CE '
        f'{float(base.mean()):.5f}')

    def with_keys(K1h, K2h, h):
        fd = dict(fold0)
        fd['K1'] = fold0['K1'].clone()
        fd['K2'] = fold0['K2'].clone()
        fd['K1'][h] = K1h
        fd['K2'][h] = K2h
        return fd

    # load-bearing head: whole-head pattern zeroing
    head_dce = {}
    for h in range(NH):
        z = torch.zeros_like(fold0['K1'][h])
        head_dce[h] = float((fold_ce(with_keys(z, z, h)) - base).mean())
    Hstar = max(head_dce, key=head_dce.get)
    say('  whole-head dCE: ' + ' '.join(f'h{h}={head_dce[h]:+.4f}'
                                        for h in range(NH))
        + f' -> load-bearing head h{Hstar}')

    K1, K2 = fold0['K1'][Hstar].clone(), fold0['K2'][Hstar].clone()
    pD = p_est.to(DEV)

    def class_span(K):
        Mn = torch.stack([(pD[:, None] * (tabs.CLS.to(DEV) == kcl)[:, None]
                           * K).sum(0)
                          / (pD * (tabs.CLS.to(DEV) == kcl)).sum()
                          for kcl in range(NCLASS)], 1)     # (hd, 8)
        return torch.linalg.qr(Mn).Q

    Q1s, Q2s = class_span(K1), class_span(K2)
    pca1 = torch.linalg.svd(pD.sqrt()[:, None] * K1,
                            full_matrices=False).Vh[:NCLASS].T
    pca2 = torch.linalg.svd(pD.sqrt()[:, None] * K2,
                            full_matrices=False).Vh[:NCLASS].T

    def class_contrast(K, k_top=2):
        """Top-k left singular directions of the CENTERED class-mean matrix:
        the leading selection-CONTRAST directions.  The 8-dim full span is
        half of a 16-dim key space and can remove ~all pattern energy (a
        scale degeneracy bilin18's 10-of-128 spans never hit), so the
        dose-response is completed with low-rank arms, exactly as section 5n
        ran arch1 alongside arch10."""
        Mn = torch.stack([(pD[:, None] * (tabs.CLS.to(DEV) == kcl)[:, None]
                           * K).sum(0)
                          / (pD * (tabs.CLS.to(DEV) == kcl)).sum()
                          for kcl in range(NCLASS)], 1)     # (hd, 8)
        gm = (pD[:, None] * K).sum(0)
        return torch.linalg.svd(Mn - gm[:, None],
                                full_matrices=False).U[:, :k_top]

    def project_out(K, Q):
        return K - (K @ Q) @ Q.T

    @torch.no_grad()
    def pattern_energy(K1a, K2a, seed=0):
        """E_{i,t ~ p x p}[(P_abl - P)^2], static (non-rotary) scores --
        the section-5n energy meter (qk_equal_ablation.py port)."""
        nb, n = ENERGY_BATCHES
        g = torch.Generator().manual_seed(seed)
        Q1f, Q2f = fold0['Q1'][Hstar], fold0['Q2'][Hstar]
        tot = 0.0
        for _ in range(nb):
            si = torch.multinomial(p_est, n, replacement=True,
                                   generator=g).to(DEV)
            ti = torch.multinomial(p_est, n, replacement=True,
                                   generator=g).to(DEV)
            s1 = Q1f[si] @ K1[ti].T / HD
            s2 = Q2f[si] @ K2[ti].T / HD
            a1 = Q1f[si] @ K1a[ti].T / HD
            a2 = Q2f[si] @ K2a[ti].T / HD
            tot += float(((s1 * s2 - a1 * a2) ** 2).mean())
        return tot / nb

    C1s, C2s = class_contrast(K1), class_contrast(K2)
    arms_k = {'sel_span8': (project_out(K1, Q1s), project_out(K2, Q2s)),
              'pca_matched8': (project_out(K1, pca1), project_out(K2, pca2)),
              'sel_top2': (project_out(K1, C1s), project_out(K2, C2s)),
              'pca_top2': (project_out(K1, pca1[:, :2]),
                           project_out(K2, pca2[:, :2]))}
    E = {nm: pattern_energy(*kk) for nm, kk in arms_k.items()}
    EP2 = pattern_energy(torch.zeros_like(K1), torch.zeros_like(K2))
    betas = {}
    for tgt, nm in (('sel_span8', 'shrink_matched8'),
                    ('sel_top2', 'shrink_top2')):
        frac = min(1.0, (E[tgt] / max(EP2, 1e-30)) ** 0.5)
        beta = (1 - frac) ** 0.5
        arms_k[nm] = (K1 * beta, K2 * beta)
        E[nm] = pattern_energy(*arms_k[nm])
        betas[nm] = round(beta, 4)
    d4 = {'head': Hstar, 'whole_head_dce': {f'h{h}': round(v, 5)
                                            for h, v in head_dce.items()},
          'fold_identity_control_maxdiff': fid,
          'base_ce': round(float(base.mean()), 5),
          'E_P2_total': EP2, 'shrink_betas': betas,
          'span_dims': {'sel_span8': NCLASS, 'sel_top2': 2}, 'arms': {}}
    for nm, (K1a, K2a) in arms_k.items():
        la = fold_ce(with_keys(K1a, K2a, Hstar))
        delta = (la - base).flatten()
        mean_dce = float(delta.mean())
        pos = delta.clamp_min(0)
        totp = float(pos.sum()) or 1e-9
        srt = pos.sort(descending=True).values
        n = len(srt)
        d4['arms'][nm] = {
            'energy_removed': E[nm], 'mean_dce': round(mean_dce, 5),
            'dce_per_energy': round(mean_dce / max(E[nm], 1e-12), 2),
            'top0.1pct_share': round(float(srt[:max(1, n // 1000)].sum())
                                     / totp, 3),
            'top1pct_share': round(float(srt[:max(1, n // 100)].sum())
                                   / totp, 3),
            'frac_positions_hit': round(float((delta.abs() > 0.01)
                                              .float().mean()), 4)}
        say(f'  {nm}: E {E[nm]:.3e} | mean dCE {mean_dce:+.5f} | per-energy '
            f'{d4["arms"][nm]["dce_per_energy"]}')
    for lvl, sel, pca, shr in (('span8', 'sel_span8', 'pca_matched8',
                                'shrink_matched8'),
                               ('top2', 'sel_top2', 'pca_top2',
                                'shrink_top2')):
        sel_pe = d4['arms'][sel]['dce_per_energy']
        d4[f'privilege_ratio_sel_vs_pca_{lvl}'] = round(
            sel_pe / max(d4['arms'][pca]['dce_per_energy'], 1e-9), 3)
        d4[f'privilege_ratio_sel_vs_shrink_{lvl}'] = round(
            sel_pe / max(d4['arms'][shr]['dce_per_energy'], 1e-9), 3)
    result['D4'] = d4
    json.dump(result, open(OUT, 'w'), indent=2)

# ----------------------------------------------------------------------------
say('writing final JSON')
result['wall_seconds'] = round(time.time() - t0, 1)
json.dump(result, open(OUT, 'w'), indent=2)
say(f'PLANTED-MODULAR DGP {"SMOKE" if SMOKE else "RUN"} DONE -> {OUT}')
