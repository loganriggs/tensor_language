"""ARCHETYPE EXTRACTION INTO A STANDALONE MACHINE (aggregate list #5;
predictions E1-E4 registered in qk_dgp_extract_predictions.json BEFORE this
code was written; E4 is the gating positive control).

THE MACHINE TEMPLATE (the sum-selection form the planted language actually
uses -- qk_dgp_lang.py law):
  a machine is a set of COMPONENTS, each carrying three slots, plus a bias:
    * attention table  SelTab_c : (8, 8) over token classes -- the weight a
      query of class a puts on every earlier position of class b (sum-
      selection, j <= i, exactly the law's accumulation form);
    * content dictionary entry  enc_c : (V,) -- the content the component
      reads off an attended token;
    * readout entry  dec_c : (V,) -- the logit direction the component writes;
  plus a unigram bias b : (V,), and TWO GLOBAL CALIBRATION SCALARS (g on the
  interaction, beta on the centered bias).  Logits at position i:
      logit(w) = g * sum_c [ sum_{j<=i} SelTab_c[cls(t_i), cls(t_j)]
                             * enc_c(t_j) ] * dec_c(w)  +  beta * b(w).
  No gradient training anywhere; the machine is explicit code + small tables.

ORACLE ASSEMBLY (E4, the gate): built directly from the true DGP tables --
24 components, SelTab_c[a,b] = 1[b == sigma(a)], enc_c = dec_c = CHI[:, c],
b = tabs.b, g = beta = 1 (NO calibration).  This machine IS the law if the
template's accumulation is right, so its held CE must sit within 0.05 nats of
the true law's log-loss on the same fresh held tokens.  It validates the
assembly template before any recovered-component verdict counts.

RECOVERED ASSEMBLY (E1/E3): run the FROZEN ledger pipeline (verbatim ports
from qk_dgp_modular.py: exposure-weighted third moment of [k1|k2|v] fold rows
per head, rank rule, 2r-atom k=2 sparse coder, tensor-power-iteration CP) on
a trained checkpoint, then map each recovered component into the template:
  * content entry: per head, the component value-directions {vdir_c} form a
    dictionary D; enc coefficients are the ridge-LS DUAL coefficients
    a(t) = (D'D + eps I)^{-1} D' Vv[h][t] of the head's folded value table --
    reconstruction semantics, so non-orthogonal components are not double
    counted;
  * readout entry: dec_c = wte @ (c_proj[:, head cols] @ vdir_c) -- the
    component write direction pushed through the tied readout;
  * attention table: the model's actual STATIC QK pattern restricted to the
    recovered key directions -- per head, project K1/K2 onto the span of the
    components' k1/k2 slices (relative norm >= 0.1), form
    P(q,k) = (Q1 q . K1r k / hd)(Q2 q . K2r k / hd), and take exposure-
    weighted class-block means -> an (8, 8) table per head (shared by the
    head's components).  Block R^2 (weighted ANOVA) is reported per head;
  * bias: exposure-weighted mean of the MODEL's position-0 logits on single-
    token sequences (the model's own no-context table; the law's position-0
    distribution is exactly softmax(b) because sigma is a derangement);
  * calibration: (g, beta) by coordinate ternary search (CE is convex in
    both) on a FRESH calibration split -- two scalars, no gradients.
The SAME recipe applied to oracle component directions (the planted units'
value-path images on the semi model, pinv dual since 24 dirs > 16 dims,
unrestricted keys) is the consistency check separating mapping-recipe failure
from recovery failure.

An honest structural note (stated up front, discovered at design time): with
dual-coefficient content entries, the aggregate machine depends on the
recovered components only through their per-head SPANS (value side) and key
spans (pattern side) -- if the components span the head's 16-dim value space,
sum_c enc_c(t) dec_c(w) telescopes to the span-projected value table.  E1
therefore tests the assembly TEMPLATE + spans; per-component individuation is
tested by E2's ablations, where zeroing a subset genuinely changes the
reconstruction.

SUB-ADDITIVITY (E2): on the assembled (identifiable) machines, classify
components as SHARED (serving >= 2 planted units at exposure-weighted
|correlation| >= 0.3 of enc vs the unit indicator, OR >= 2 selection classes
at >= 0.25 row-energy share of its SelTab) vs SINGLE-ROLE (exactly one unit,
<= 1 class); zero each set and score CE damage on fresh held.  Energy meter
(stated): m_c = g^2 * (sum_{a,b} pcls_a pcls_b SelTab_c[a,b]^2)
* (sum_t p_t enc_c(t)^2) * (Var_{w~p} dec_c(w)) -- the exposure-weighted
mean-square logit contribution of the component; the single-role comparison
set is greedily energy-matched to the shared set's total.  The oracle
machine is the reference; there the class axis is DEGENERATE (every oracle
component serves all 8 classes by construction of the law), so the oracle
split is unit-only and the reference number is the damage of removing a
random energy-fraction-matched subset.

FRESH DATA (stated seeds, never used before): held2 = sampler seed 21
(scoring), cal = seed 22 (calibration only).  Frozen-run seeds 11/12/13 and
101-104 are not reused for scoring; the pipeline's exposure p_est stays the
frozen est split (seed 13), as the frozen pipeline defines it.

--stage oracle,identifiable,overlap,subadd (default all, in order).  --smoke
(or QK_SMOKE=1): oracle assembly at full fidelity on 200 held sequences plus
one tiny recovered assembly (identifiable/semi, heads 0-1, SAE 600 steps,
100 held) -> qk_dgp_extract_smoke.json only.  Full mode: E4 failure writes
the JSON and hard-stops before any recovered stage.  Machines persist to
qk_dgp_extract_machines.pt so stages can run in separate invocations.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
sys.path.insert(0, QK)
import qk_dgp_lang as DL  # noqa: E402
from qk_dgp_lang import V, T, WS, NCLASS, NUNITS  # noqa: E402

SMOKE = ('--smoke' in sys.argv) or (os.environ.get('QK_SMOKE') == '1')
STAGES = [s.strip() for s in (
    sys.argv[sys.argv.index('--stage') + 1] if '--stage' in sys.argv
    else 'oracle,identifiable,overlap,subadd').split(',')]

torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
HD = 16
NH = WS // HD
EST_SEED = 13                      # frozen pipeline exposure split
HELD2_SEED, CAL_SEED = 21, 22      # FRESH: scoring + calibration
HELD2_N = 200 if SMOKE else 1500
CAL_N = 60 if SMOKE else 400
SAE_STEPS = 600 if SMOKE else 6000
SAE_BATCH = 2048 if SMOKE else 4096
SAE_LR = 3e-3
RESTARTS, MARGIN = 3, 0.15
EST_N = 300 if SMOKE else 1500
KEY_REL_THRESH = 0.1
UNIT_LOAD_THRESH = 0.3
CLASS_SHARE_THRESH = 0.25
E4_GATE_NATS = 0.05
HEADS = ([0, 1] if SMOKE else list(range(NH)))
OUT = f'{QK}/qk_dgp_extract{"_smoke" if SMOKE else ""}.json'
MACH = f'{QK}/qk_dgp_extract_machines{"_smoke" if SMOKE else ""}.pt'

t0 = time.time()


def say(msg):
    print(f'[{time.time() - t0:7.1f}s] {msg}', flush=True)


def save(update):
    prev = {}
    if os.path.exists(OUT):
        try:
            prev = json.load(open(OUT))
        except Exception:
            prev = {}
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(prev.get(k), dict):
            prev[k].update(v)
        else:
            prev[k] = v
    json.dump(prev, open(OUT, 'w'), indent=2)


DEVIATIONS = [
    'class index: the machine indexes its selection tables by the PLANTED '
    'token->class map (taken as known vocabulary structure, like scoring '
    'recovery against planted units); the model\'s conformance to the class '
    'block structure is measured, not assumed (per-head pattern block R^2).',
    'calibration: two global scalars per machine (g on the interaction, beta '
    'on the centered bias), fit by coordinate ternary search (CE is convex '
    'in both) on the fresh calibration split (seed 22). No gradient '
    'training. The E4 gate itself is UNCALIBRATED (g = beta = 1).',
    'bias slot: exposure-weighted mean of the model\'s position-0 logits on '
    'single-token sequences (the model\'s no-context table; the law at '
    'position 0 is exactly softmax(b) since sigma is a derangement).',
    'content entries are ridge-LS dual coefficients of the folded value '
    'table on the component dictionary (ridge 1e-6 * trace(Gram)/nc): '
    'reconstruction semantics, no double counting of non-orthogonal '
    'components. The oracle-component consistency machine uses pinv '
    '(rtol 1e-6) because 24 unit images exceed the 16-dim head value space.',
    'attention tables use STATIC (non-rotary) fold scores -- the frozen '
    'pipeline\'s own energy-meter convention; rotary drift is part of what '
    'the assembly may lose.',
    'key restriction: per head, K1/K2 are projected onto the span of the '
    'recovered components\' k1/k2 slices with relative norm >= 0.1 of the '
    'full 48-dim atom combination; a head with no passing slice contributes '
    'no selection.',
    'E2 operationalization: shared = (>= 2 planted units at exposure-'
    'weighted |corr(enc, unit indicator)| >= 0.3) OR (>= 2 selection '
    'classes at >= 0.25 row-energy share of the head SelTab); single-role = '
    'exactly 1 unit and <= 1 class; components with no unit above threshold '
    'are "unclassified" and belong to neither set. On the ORACLE machine '
    'the class axis is degenerate (every component serves all 8 classes by '
    'construction), so its split is unit-only and its reference arm is a '
    'random energy-fraction-matched subset.',
    'structural note: with dual-coefficient content entries the aggregate '
    'machine depends on components only through per-head value/key SPANS; '
    'E1 tests template+spans, E2 tests per-component individuation.',
]


# ----------------------------------------------------------------------------
# Frozen ledger machinery -- verbatim ports from qk_dgp_modular.py (itself
# porting qk_stage23.py tick 173 / qk_null_repair.py tick 183 via
# qk_iface_ledger.py).  run_head_pipeline is the same computation; the ONLY
# extension is that each returned component additionally exposes its k1/k2
# slices of d48 (already computed there, just not returned).
# ----------------------------------------------------------------------------
def cp_fit(core_raw, R, seed, n_starts=8, iters=60):
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
    vals = []
    for i in range(len(Us)):
        for j in range(i + 1, len(Us)):
            C = (Us[i].T @ Us[j]).abs()
            vals.append(float(C.max(1).values.mean()))
    return sum(vals) / len(vals)


def train_sae(Z, m_atoms, k_code, p_cpu, steps, batch, lr=SAE_LR, seed=0):
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


def run_head_pipeline(model, h, p_cpu, dev):
    """Frozen ledger pipeline on one head (verbatim port; components
    additionally expose the k1/k2 slices of d48)."""
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
                      'vdir': vdir.cpu(), 'e_dir': e_dir.cpu(),
                      'k1dir': d48[:HD].cpu(), 'k2dir': d48[HD:2 * HD].cpu(),
                      'd48norm': float(d48.norm())})
    return {'eff': round(eff, 2), 'r': r, 'm_atoms': m_atoms,
            'real_fit': round(real_fit, 4),
            'null_on_real': round(null_on_real, 4),
            'margin': round(margin, 4), 'm1_pass': bool(margin >= MARGIN),
            'restart_stability': round(stab, 3),
            'restart_relerrs': [round(x, 4) for x in rels]}, comps


def load_arm(variant, arm, tabs_v):
    ck = f'{QK}/qk_dgp_{variant}_{arm}.pt'
    assert os.path.exists(ck), f'missing checkpoint {ck}'
    m = DL.make_arm_model(tabs_v, arm, seed=0, device=DEV)
    st = torch.load(ck, map_location=DEV)
    m.load_state_dict(st['state_dict'])
    return m.eval(), st['log']


# ----------------------------------------------------------------------------
# The machine: representation, evaluation, calibration
# ----------------------------------------------------------------------------
@torch.no_grad()
def machine_interaction(M, toks, dev, keep=None):
    """Interaction logits X (B, T-1, V): position i predicts token i+1 from
    the prefix j <= i (the law's accumulation).  No g, no bias."""
    SEL = M['SEL'].to(dev)
    ENC = M['ENC'].to(dev)
    DEC = M['DEC'].to(dev)
    cls = M['cls'].to(dev)
    if keep is not None:
        SEL, ENC, DEC = SEL[keep], ENC[keep], DEC[keep]
    NC = ENC.shape[0]
    B, Tt = toks.shape
    encv = ENC[:, toks]                                    # (NC, B, T)
    clsq = cls[toks]                                       # (B, T)
    S = torch.zeros(B, NCLASS, NC, device=dev)
    X = torch.empty(B, Tt - 1, ENC.shape[1], device=dev)
    for i in range(Tt - 1):
        oh = F.one_hot(clsq[:, i], NCLASS).float()         # (B, 8)
        S = S + oh[:, :, None] * encv[:, :, i].T[:, None, :]
        rows = SEL[:, clsq[:, i], :]                       # (NC, B, 8)
        inner = torch.einsum('cbk,bkc->bc', rows, S)       # (B, NC)
        X[:, i] = inner @ DEC
    return X


@torch.no_grad()
def machine_interaction_fulltable(P, comp_head, ENC, DEC, toks, dev):
    """Full-token-table variant: the same machine but with the (V, V) static
    pattern table per head instead of its class-block approximation."""
    NC = ENC.shape[0]
    B, Tt = toks.shape
    heads = sorted(set(int(h) for h in comp_head))
    idx_h = {h: torch.tensor([i for i in range(NC) if int(comp_head[i]) == h],
                             device=dev) for h in heads}
    cnt = torch.zeros(B, V, device=dev)
    ar = torch.arange(B, device=dev)
    X = torch.empty(B, Tt - 1, V, device=dev)
    for i in range(Tt - 1):
        cnt[ar, toks[:, i]] += 1.0
        inner = torch.zeros(B, NC, device=dev)
        for h in heads:
            w = P[h][toks[:, i]] * cnt                     # (B, V)
            inner[:, idx_h[h]] = w @ ENC[idx_h[h]].T
        X[:, i] = inner @ DEC
    return X


@torch.no_grad()
def machine_ce(M, seqs, dev, keep=None, per_seq=False, batch=256,
               interaction=None):
    """Mean held CE of the machine; per_seq returns per-sequence means too."""
    bias = M['bias'].to(dev)
    g, beta = M['g'], M['beta']
    means = []
    tot, n = 0.0, 0
    for i in range(0, len(seqs), batch):
        b = seqs[i:i + batch].to(dev)
        if interaction is None:
            X = machine_interaction(M, b, dev, keep=keep)
        else:
            X = interaction(b)
        logits = g * X + beta * bias
        ce = F.cross_entropy(logits.reshape(-1, V), b[:, 1:].reshape(-1),
                             reduction='none').view(b.shape[0], -1)
        tot += float(ce.sum())
        n += ce.numel()
        if per_seq:
            means.append(ce.mean(1).cpu())
    if per_seq:
        return tot / n, torch.cat(means)
    return tot / n


def ternary_min(f, lo, hi, iters=70):
    for _ in range(iters):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if f(m1) <= f(m2):
            hi = m2
        else:
            lo = m1
    return (lo + hi) / 2


@torch.no_grad()
def calibrate(X, bias, targets, dev):
    """(g, beta) by coordinate ternary search; CE is convex in each (logits
    are affine in (g, beta)).  X (N, V) interaction logits, normalized
    internally for a well-scaled search interval."""
    sx = float(X.std()) or 1.0
    sb = float(bias.std()) or 1.0
    Xn, bn = X / sx, bias / sb

    def ce(gn, betan):
        return float(F.cross_entropy(gn * Xn + betan * bn, targets))

    gn, betan = 0.0, 1.0
    for _ in range(6):
        gn = ternary_min(lambda x: ce(x, betan), -64.0, 64.0)
        betan = ternary_min(lambda x: ce(gn, x), -16.0, 16.0)
    return gn / sx, betan / sb, ce(gn, betan)


def apply_calibration(M, cal_seqs, dev, interaction=None):
    Xs, tg = [], []
    for i in range(0, len(cal_seqs), 256):
        b = cal_seqs[i:i + 256].to(dev)
        X = (machine_interaction(M, b, dev) if interaction is None
             else interaction(b))
        Xs.append(X.reshape(-1, V))
        tg.append(b[:, 1:].reshape(-1))
    X = torch.cat(Xs)
    tg = torch.cat(tg)
    g, beta, cal_ce = calibrate(X, M['bias'].to(dev), tg, dev)
    M['g'], M['beta'] = float(g), float(beta)
    return round(cal_ce, 5)


# ----------------------------------------------------------------------------
# Machine builders
# ----------------------------------------------------------------------------
def oracle_machine(tabs_v):
    """E4: directly from the true tables.  g = beta = 1, no calibration."""
    SEL = torch.zeros(NUNITS, NCLASS, NCLASS)
    for a in range(NCLASS):
        SEL[:, a, int(tabs_v.SIGMA[a])] = 1.0
    return {'SEL': SEL, 'ENC': tabs_v.CHI.T.clone(),
            'DEC': tabs_v.CHI.T.clone(), 'bias': tabs_v.b.clone(),
            'cls': tabs_v.CLS.clone(),
            'comp_head': torch.full((NUNITS,), -1, dtype=torch.long),
            'g': 1.0, 'beta': 1.0,
            'meta': {'kind': 'oracle_tables', 'variant': tabs_v.variant}}


@torch.no_grad()
def model_bias_table(model, p_est, dev):
    """Exposure-weighted mean of position-0 logits over single-token
    sequences: the model's own no-context table (centered)."""
    toks = torch.arange(V, device=dev)[:, None]           # (V, 1)
    lg = torch.cat([model(toks[i:i + 128]).float()[:, 0]
                    for i in range(0, V, 128)])           # (V, V)
    b = (p_est.to(dev)[:, None] * lg).sum(0)
    return (b - b.mean()).cpu()


@torch.no_grad()
def class_block(Pt, cls_d, p_d):
    """Exposure-weighted class-block means of a (V, V) score table + the
    weighted ANOVA R^2 of the block approximation."""
    onehot = F.one_hot(cls_d, NCLASS).float()             # (V, 8)
    wq = onehot * p_d[:, None]                            # (V, 8)
    mass = wq.sum(0)                                      # (8,)
    Sel = (wq.T @ Pt @ wq) / (mass[:, None] * mass[None, :]).clamp_min(1e-30)
    Phat = onehot @ Sel @ onehot.T
    W = p_d[:, None] * p_d[None, :]
    mu = float((W * Pt).sum() / W.sum())
    ss_tot = float((W * (Pt - mu) ** 2).sum())
    ss_res = float((W * (Pt - Phat) ** 2).sum())
    r2 = 1 - ss_res / max(ss_tot, 1e-30)
    return Sel, r2


@torch.no_grad()
def recovered_machine(model, tabs_v, comps_by_head, p_est, dev,
                      oracle_dirs=False):
    """Map recovered (or oracle-direction) components into the template.
    Returns (machine, fulltable_P, per_head_meta)."""
    fold = model.fold_layer0_qk(materialize=False, device=dev)
    cw = model.h[0].c_proj.weight.detach().float().to(dev)     # (Ws, Dc)
    wte = model.wte.weight.detach().float().to(dev)            # (V, Ws)
    p_d = p_est.to(dev)
    cls_d = tabs_v.CLS.to(dev)
    SELs, ENCs, DECs, comp_head, comp_meta = [], [], [], [], []
    P_full = torch.zeros(NH, V, V, device=dev)
    per_head = {}
    for h in sorted(comps_by_head):
        comps = comps_by_head[h]
        if not comps:
            continue
        Dm = torch.stack([c['vdir'].to(dev) for c in comps], 1)  # (hd, nc)
        Dm = Dm / Dm.norm(dim=0, keepdim=True).clamp_min(1e-12)
        nc = Dm.shape[1]
        Vv = fold['Vv'][h].to(dev)                               # (V, hd)
        if oracle_dirs:
            A = torch.linalg.pinv(Dm, rtol=1e-6) @ Vv.T          # (nc, V)
        else:
            G = Dm.T @ Dm
            ridge = 1e-6 * float(torch.diagonal(G).sum()) / nc
            A = torch.linalg.solve(G + ridge * torch.eye(nc, device=dev),
                                   Dm.T @ Vv.T)                  # (nc, V)
        dec = (wte @ (cw[:, h * HD:(h + 1) * HD] @ Dm)).T        # (nc, V)
        # content coverage: exposure-weighted centered variance of Vv
        # captured by the component span
        Vc = Vv - (p_d[:, None] * Vv).sum(0)
        proj = (Dm @ torch.linalg.pinv(Dm, rtol=1e-6))
        cov_frac = float((p_d[:, None] * (Vc @ proj.T) ** 2).sum()
                         / (p_d[:, None] * Vc ** 2).sum().clamp_min(1e-30))
        # key restriction -> static pattern -> class block table
        if oracle_dirs:
            K1r, K2r = fold['K1'][h].to(dev), fold['K2'][h].to(dev)
            kdims = (HD, HD)
        else:
            spans = []
            for which in ('k1dir', 'k2dir'):
                ds = [c[which].to(dev) for c in comps
                      if float(c[which].norm()) >= KEY_REL_THRESH * c['d48norm']]
                if ds:
                    Q, R = torch.linalg.qr(torch.stack(ds, 1))
                    # drop numerically-null columns from rank-deficient stacks
                    keep = R.abs().diagonal() > 1e-6 * R.abs().max().clamp_min(1e-30)
                    spans.append(Q[:, keep])
                else:
                    spans.append(torch.zeros(HD, 0, device=dev))
            Qk1, Qk2 = spans
            K1r = (fold['K1'][h].to(dev) @ Qk1) @ Qk1.T
            K2r = (fold['K2'][h].to(dev) @ Qk2) @ Qk2.T
            kdims = (Qk1.shape[1], Qk2.shape[1])
        s1 = fold['Q1'][h].to(dev) @ K1r.T / HD
        s2 = fold['Q2'][h].to(dev) @ K2r.T / HD
        Pt = s1 * s2                                             # (V, V)
        P_full[h] = Pt
        Sel, block_r2 = class_block(Pt, cls_d, p_d)
        # diagnostic: exposure-weighted energy overlap of the restricted
        # pattern with the head's FULL static pattern
        Ptf = (fold['Q1'][h].to(dev) @ fold['K1'][h].to(dev).T / HD) \
            * (fold['Q2'][h].to(dev) @ fold['K2'][h].to(dev).T / HD)
        Wp = p_d[:, None] * p_d[None, :]
        pat_overlap = float((Wp * Pt * Ptf).sum()
                            / (Wp * Ptf ** 2).sum().clamp_min(1e-30))
        per_head[f'h{h}'] = {'n_components': nc,
                             'key_span_dims': list(kdims),
                             'pattern_block_r2': round(block_r2, 4),
                             'pattern_overlap_with_full': round(pat_overlap, 4),
                             'value_span_coverage': round(cov_frac, 4)}
        for ci in range(nc):
            SELs.append(Sel.cpu())
            ENCs.append(A[ci].cpu())
            DECs.append(dec[ci].cpu())
            comp_head.append(h)
            comp_meta.append({'head': h, 'lambda': comps[ci].get('lambda'),
                              'k1_rel': round(float(comps[ci]['k1dir'].norm())
                                              / comps[ci]['d48norm'], 4)
                              if 'k1dir' in comps[ci] else None,
                              'k2_rel': round(float(comps[ci]['k2dir'].norm())
                                              / comps[ci]['d48norm'], 4)
                              if 'k2dir' in comps[ci] else None})
    M = {'SEL': torch.stack(SELs), 'ENC': torch.stack(ENCs),
         'DEC': torch.stack(DECs), 'bias': model_bias_table(model, p_est, dev),
         'cls': tabs_v.CLS.clone(),
         'comp_head': torch.tensor(comp_head, dtype=torch.long),
         'g': 1.0, 'beta': 1.0,
         'meta': {'kind': 'oracle_dirs' if oracle_dirs else 'recovered',
                  'variant': tabs_v.variant, 'per_head': per_head,
                  'components': comp_meta}}
    return M, P_full, per_head


# ----------------------------------------------------------------------------
# Scoring references
# ----------------------------------------------------------------------------
def references(tabs_v, held, dev):
    ll, en = DL.true_scores(tabs_v, held)
    logb = torch.log_softmax(tabs_v.b, 0)
    floor = float(-logb[held[:, 1:]].mean())
    return {'true_law_logloss': round(float(ll.mean()), 5),
            'entropy_mc': round(float(en.mean()), 5),
            'unigram_floor_ce': round(floor, 5),
            'n_seqs': int(held.shape[0])}, ll


def score_machine(M, held, dev, refs, model_ce=None, P_full=None,
                  comp_head=None):
    ce_b = machine_ce(M, held, dev)
    bias_only = machine_ce({**M, 'g': 0.0}, held, dev,
                           interaction=lambda b: torch.zeros(
                               b.shape[0], b.shape[1] - 1, V, device=dev))
    row = {'machine_ce_blocked': round(ce_b, 5),
           'machine_minus_law': round(ce_b - refs['true_law_logloss'], 5),
           'bias_only_ce': round(bias_only, 5),
           'calibration': {'g': M['g'], 'beta': M['beta']}}
    if model_ce is not None:
        row['model_ce'] = round(model_ce, 5)
        row['machine_minus_model'] = round(ce_b - model_ce, 5)
    if P_full is not None:
        Mft = dict(M)
        ce_f = None
        # calibrate the full-table variant separately (same recipe)
        inter = lambda b: machine_interaction_fulltable(  # noqa: E731
            P_full, comp_head, M['ENC'].to(dev), M['DEC'].to(dev), b, dev)
        cal_ce = apply_calibration(Mft, CAL_CACHE[M['meta']['variant']], dev,
                                   interaction=inter)
        ce_f = machine_ce(Mft, held, dev, interaction=inter)
        row['machine_ce_fulltable'] = round(ce_f, 5)
        row['fulltable_calibration'] = {'g': Mft['g'], 'beta': Mft['beta'],
                                        'cal_ce': cal_ce}
        row['blocked_minus_fulltable'] = round(ce_b - ce_f, 5)
    return row


# ----------------------------------------------------------------------------
say(f'=== archetype extraction ({"SMOKE" if SMOKE else "FULL"}), stages '
    f'{STAGES}, device {DEV} ===')
save({'experiment': 'archetype extraction into a standalone machine '
                    '(aggregate #5)',
      'registered': 'qk_dgp_extract_predictions.json (E1-E4)',
      'mode': 'smoke' if SMOKE else 'full',
      'seeds': {'held2_scoring': HELD2_SEED, 'calibration': CAL_SEED,
                'pipeline_exposure_est': EST_SEED,
                'note': 'held2/cal are FRESH sampler seeds, never used in '
                        'the frozen runs (11/12/13, 101-104)'},
      'config': {'held2_n': HELD2_N, 'cal_n': CAL_N, 'sae_steps': SAE_STEPS,
                 'heads': HEADS, 'key_rel_thresh': KEY_REL_THRESH,
                 'unit_load_thresh': UNIT_LOAD_THRESH,
                 'class_share_thresh': CLASS_SHARE_THRESH,
                 'e4_gate_nats': E4_GATE_NATS},
      'deviations': DEVIATIONS})

TABS = {vn: DL.DGPTables(variant=vn) for vn in ('identifiable', 'overlap')}
HELD_CACHE, CAL_CACHE, REF_CACHE = {}, {}, {}


def get_data(vn):
    if vn not in HELD_CACHE:
        HELD_CACHE[vn] = DL.sample_seqs(TABS[vn], HELD2_N, HELD2_SEED)
        CAL_CACHE[vn] = DL.sample_seqs(TABS[vn], CAL_N, CAL_SEED)
        REF_CACHE[vn], _ = references(TABS[vn], HELD_CACHE[vn], DEV)
        say(f'  {vn}: sampled held2 {HELD2_N} (seed {HELD2_SEED}) + cal '
            f'{CAL_N} (seed {CAL_SEED}); law log-loss '
            f'{REF_CACHE[vn]["true_law_logloss"]:.4f}, unigram floor '
            f'{REF_CACHE[vn]["unigram_floor_ce"]:.4f}')
    return HELD_CACHE[vn], CAL_CACHE[vn], REF_CACHE[vn]


def machines_store():
    return torch.load(MACH) if os.path.exists(MACH) else {}


# ============================================================================
# Stage ORACLE (E4, the gate)
# ============================================================================
if 'oracle' in STAGES:
    say('E4: oracle assembly from the true tables (uncalibrated, g=beta=1)')
    E4 = {}
    gate_ok = True
    store = machines_store()
    for vn in ('identifiable', 'overlap'):
        held, cal, refs = get_data(vn)
        M = oracle_machine(TABS[vn])
        ce = machine_ce(M, held, DEV)
        gap = ce - refs['true_law_logloss']
        ok = bool(abs(gap) <= E4_GATE_NATS)
        gate_ok = gate_ok and ok
        # recipe consistency: the calibration step applied to the oracle
        # machine must return (g, beta) ~ (1, 1)
        Mc = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in M.items()}
        cal_ce = apply_calibration(Mc, cal, DEV)
        ce_c = machine_ce(Mc, held, DEV)
        E4[vn] = {'references': refs,
                  'oracle_ce': round(ce, 5),
                  'gap_vs_law_paired': round(gap, 6),
                  'gate_threshold': E4_GATE_NATS, 'pass': ok,
                  'calibration_consistency': {
                      'g': round(Mc['g'], 5), 'beta': round(Mc['beta'], 5),
                      'cal_ce': cal_ce, 'held_ce_calibrated': round(ce_c, 5),
                      'note': 'the recipe\'s calibration applied to the '
                              'oracle machine should return ~ (1, 1)'}}
        store[f'oracle_{vn}'] = M
        say(f'  {vn}: oracle CE {ce:.5f} vs law '
            f'{refs["true_law_logloss"]:.5f} -> gap {gap:+.6f} '
            f'({"PASS" if ok else "FAIL"}); calibrated (g, beta) = '
            f'({Mc["g"]:.4f}, {Mc["beta"]:.4f})')
    torch.save(store, MACH)
    E4['gate'] = 'PASS' if gate_ok else 'FAIL'
    save({'E4': E4})
    if not gate_ok and not SMOKE:
        say('E4 GATE FAILED -- the assembly template is unsound; E1-E3 are '
            'uninterpretable.  STOP (fix the recipe before proceeding).')
        sys.exit(1)
elif not SMOKE:
    prev = json.load(open(OUT)) if os.path.exists(OUT) else {}
    assert prev.get('E4', {}).get('gate') == 'PASS', \
        'E4 gate has not passed; run --stage oracle first'


# ============================================================================
# Stage RECOVERED ASSEMBLY per variant (E1: identifiable; E3: overlap)
# ============================================================================
def run_variant(vn):
    tabs_v = TABS[vn]
    held, cal, refs = get_data(vn)
    est_seqs = DL.sample_seqs(tabs_v, EST_N, EST_SEED)
    cnt = torch.bincount(est_seqs.flatten(), minlength=V).float() + 0.5
    p_est = cnt / cnt.sum()
    block = {'references': refs}
    store = machines_store()
    arms = ['semi'] if SMOKE else ['learned', 'semi']
    for arm in arms:
        say(f'{vn}/{arm}: loading checkpoint + frozen ledger pipeline '
            f'(heads {HEADS})')
        model, log = load_arm(vn, arm, tabs_v)
        model_ce = DL.eval_ce(model, held, DEV)
        comps_by_head, heads_res = {}, {}
        for h in HEADS:
            hres, comps = run_head_pipeline(model, h, p_est, DEV)
            comps_by_head[h] = comps
            heads_res[f'h{h}'] = hres
            say(f'  h{h}: r={hres["r"]} margin {hres["margin"]:+.3f} '
                f'{len(comps)} comps')
        M, P_full, per_head = recovered_machine(model, tabs_v, comps_by_head,
                                                p_est, DEV)
        cal_ce = apply_calibration(M, cal, DEV)
        row = score_machine(M, held, DEV, refs, model_ce=model_ce,
                            P_full=P_full, comp_head=M['comp_head'])
        row['n_components'] = int(M['ENC'].shape[0])
        row['cal_ce'] = cal_ce
        row['per_head'] = per_head
        row['pipeline_heads'] = heads_res
        block[arm] = row
        store[f'{vn}_{arm}'] = M
        torch.save(store, MACH)
        say(f'{vn}/{arm}: machine CE {row["machine_ce_blocked"]:.4f} '
            f'(fulltable {row.get("machine_ce_fulltable")}) vs model '
            f'{model_ce:.4f} vs law {refs["true_law_logloss"]:.4f} vs floor '
            f'{refs["unigram_floor_ce"]:.4f} | machine-model gap '
            f'{row["machine_minus_model"]:+.4f} | g {M["g"]:.4f} beta '
            f'{M["beta"]:.4f}')
        # consistency check: same recipe on ORACLE component directions
        # (semi arm only -- the planted embedding is its input table)
        if arm == 'semi':
            say(f'{vn}/semi: consistency check -- recipe on oracle unit '
                f'value-path images (24 dirs x {len(HEADS)} heads, '
                f'unrestricted keys)')
            oc_by_head = {}
            Wv = model.h[0].c_v.weight.detach().float().to(DEV)
            for h in HEADS:
                cs = []
                for u in range(NUNITS):
                    e_u = torch.zeros(WS, device=DEV)
                    e_u[DL.CONTENT_SLICE] = tabs_v.UNITS[:, u].to(DEV)
                    vd = Wv[h * HD:(h + 1) * HD] @ e_u
                    cs.append({'vdir': vd.cpu(), 'lambda': None})
                oc_by_head[h] = cs
            Mo, Po, per_head_o = recovered_machine(
                model, tabs_v, oc_by_head, p_est, DEV, oracle_dirs=True)
            cal_o = apply_calibration(Mo, cal, DEV)
            row_o = score_machine(Mo, held, DEV, refs, model_ce=model_ce)
            row_o['per_head'] = per_head_o
            row_o['cal_ce'] = cal_o
            block['consistency_oracle_components_semi'] = row_o
            store[f'{vn}_oracle_dirs_semi'] = Mo
            torch.save(store, MACH)
            say(f'{vn}/semi oracle-dirs machine CE '
                f'{row_o["machine_ce_blocked"]:.4f} (machine-model gap '
                f'{row_o["machine_minus_model"]:+.4f})')
    key = 'E1' if vn == 'identifiable' else 'E3'
    save({key: {vn: block}})
    return block


for vn in ('identifiable', 'overlap'):
    if vn in STAGES:
        run_variant(vn)


# ============================================================================
# Stage SUBADD (E2): shared vs single-role ablations on assembled machines
# ============================================================================
def classify_components(M, tabs_v, p_est):
    """Unit loadings (exposure-weighted |corr| of enc vs unit indicator) and
    served selection classes (row-energy share of the SelTab)."""
    p = p_est
    pcls = torch.tensor([float(p[tabs_v.CLS == a].sum())
                         for a in range(NCLASS)])
    ENC, SEL = M['ENC'], M['SEL']
    NC = ENC.shape[0]
    chi = tabs_v.CHI                                       # (V, 24)
    mu_c = (p[None, :] * ENC).sum(1, keepdim=True)
    Ec = ENC - mu_c
    sd_e = ((p[None, :] * Ec ** 2).sum(1)).sqrt().clamp_min(1e-12)
    mu_u = (p[:, None] * chi).sum(0, keepdim=True)
    Uc = chi - mu_u
    sd_u = ((p[:, None] * Uc ** 2).sum(0)).sqrt().clamp_min(1e-12)
    corr = ((Ec * p[None, :]) @ Uc) / (sd_e[:, None] * sd_u[None, :])
    rows = []
    for c in range(NC):
        loads = corr[c].abs()
        units = (loads >= UNIT_LOAD_THRESH).nonzero().squeeze(1).tolist()
        row_energy = pcls * (SEL[c] ** 2 @ pcls)           # (8,)
        share = row_energy / row_energy.sum().clamp_min(1e-30)
        classes = (share >= CLASS_SHARE_THRESH).nonzero().squeeze(1).tolist()
        rows.append({'units': units, 'classes': classes,
                     'max_load': round(float(loads.max()), 4),
                     'n_units': len(units), 'n_classes': len(classes)})
    return rows


def component_energy(M, tabs_v, p_est):
    p = p_est
    pcls = torch.tensor([float(p[tabs_v.CLS == a].sum())
                         for a in range(NCLASS)])
    ENC, DEC, SEL = M['ENC'], M['DEC'], M['SEL']
    e_sel = torch.einsum('cab,a,b->c', SEL ** 2, pcls, pcls)
    e_enc = (p[None, :] * ENC ** 2).sum(1)
    mu_d = (p[None, :] * DEC).sum(1, keepdim=True)
    e_dec = (p[None, :] * (DEC - mu_d) ** 2).sum(1)
    return (M['g'] ** 2) * e_sel * e_enc * e_dec


def ablate(M, held, dev, drop_idx, base_ce, base_seq):
    NC = M['ENC'].shape[0]
    keep = torch.tensor([i for i in range(NC) if i not in set(drop_idx)],
                        dtype=torch.long)
    ce, seq = machine_ce(M, held, dev, keep=keep, per_seq=True)
    d = seq - base_seq
    se = float(d.std(unbiased=True) / np.sqrt(len(d)))
    return {'n_dropped': len(drop_idx), 'ce': round(ce, 5),
            'dce': round(ce - base_ce, 5), 'dce_se_seq_clustered': round(se, 6)}


def greedy_match(cands, energies, target):
    """Greedy energy match: largest-first while staying under 1.05*target."""
    order = sorted(cands, key=lambda i: -float(energies[i]))
    pick, tot = [], 0.0
    for i in order:
        e = float(energies[i])
        if tot + e <= 1.05 * target:
            pick.append(i)
            tot += e
    return pick, tot


def subadd_on(M, tabs_v, held, dev, p_est, name):
    rows = classify_components(M, tabs_v, p_est)
    en = component_energy(M, tabs_v, p_est)
    NC = len(rows)
    shared = [c for c in range(NC)
              if rows[c]['n_units'] >= 2
              or (rows[c]['n_units'] >= 1 and rows[c]['n_classes'] >= 2)]
    single = [c for c in range(NC)
              if rows[c]['n_units'] == 1 and rows[c]['n_classes'] <= 1]
    unclassified = [c for c in range(NC) if rows[c]['n_units'] == 0]
    base_ce, base_seq = machine_ce(M, held, dev, per_seq=True)
    E_sh = float(en[shared].sum()) if shared else 0.0
    E_si_all = float(en[single].sum()) if single else 0.0
    out = {'n_components': NC, 'n_shared': len(shared),
           'n_single_role': len(single), 'n_unclassified': len(unclassified),
           'shared_energy': E_sh, 'single_all_energy': E_si_all,
           'total_energy': float(en.sum()),
           'base_machine_ce': round(base_ce, 5),
           'component_table': [
               {'i': c, 'head': int(M['comp_head'][c]), **rows[c],
                'energy': float(en[c]),
                'set': ('shared' if c in shared else
                        'single' if c in single else 'unclassified')}
               for c in range(NC)]}
    if not shared or not single:
        out['verdict'] = ('degenerate split: shared or single-role set is '
                          'empty; see component_table')
        return out
    arm_sh = ablate(M, held, dev, shared, base_ce, base_seq)
    pick, E_pick = greedy_match(single, en, E_sh)
    arm_si = ablate(M, held, dev, pick, base_ce, base_seq)
    arm_si_all = ablate(M, held, dev, single, base_ce, base_seq)
    out['zero_shared'] = {**arm_sh, 'energy_removed': E_sh}
    out['zero_single_energy_matched'] = {
        **arm_si, 'energy_removed': E_pick,
        'energy_match_achieved_frac': round(E_pick / max(E_sh, 1e-30), 4),
        'picked': pick}
    out['zero_single_all'] = {**arm_si_all, 'energy_removed': E_si_all}
    d_sh, d_si = arm_sh['dce'], arm_si['dce']
    out['damage_ratio_shared_over_single_matched'] = (
        round(d_sh / d_si, 3) if abs(d_si) > 1e-9 else None)
    per_sh = d_sh / max(E_sh, 1e-30)
    per_si = arm_si_all['dce'] / max(E_si_all, 1e-30)
    out['damage_per_energy_ratio_shared_over_single_all'] = (
        round(per_sh / per_si, 3) if abs(per_si) > 1e-12 else None)
    say(f'  {name}: shared {len(shared)} comps (E {E_sh:.3g}) dCE '
        f'{d_sh:+.4f} | single matched {len(pick)} comps (E {E_pick:.3g}) '
        f'dCE {d_si:+.4f} | ratio '
        f'{out["damage_ratio_shared_over_single_matched"]}')
    return out


if 'subadd' in STAGES:
    say('E2: sub-additivity ablations on the assembled machines')
    store = machines_store()
    E2 = {}
    vn = 'identifiable'
    tabs_v = TABS[vn]
    held, cal, refs = get_data(vn)
    est_seqs = DL.sample_seqs(tabs_v, EST_N, EST_SEED)
    cnt = torch.bincount(est_seqs.flatten(), minlength=V).float() + 0.5
    p_est = cnt / cnt.sum()
    arms = [a for a in ('learned', 'semi') if f'{vn}_{a}' in store]
    for arm in arms:
        E2[f'{vn}_{arm}'] = subadd_on(store[f'{vn}_{arm}'], tabs_v, held,
                                      DEV, p_est, f'{vn}/{arm}')
    # oracle reference: unit-only split (class axis degenerate); remove a
    # random subset matched to the primary machine's shared ENERGY FRACTION
    if f'oracle_{vn}' in store and arms:
        Mo = store[f'oracle_{vn}']
        en_o = component_energy(Mo, tabs_v, p_est)
        prim = E2[f'{vn}_{arms[0]}']
        base_o, base_o_seq = machine_ce(Mo, held, DEV, per_seq=True)
        ref = {'note': 'oracle machine: class axis degenerate, all 24 '
                       'components single-unit; reference = random subsets '
                       'matched to the primary machine\'s shared energy '
                       'fraction', 'base_ce': round(base_o, 5),
               'draws': []}
        if 'shared_energy' in prim and prim.get('total_energy'):
            f_sh = prim['shared_energy'] / prim['total_energy']
            tgt = f_sh * float(en_o.sum())
            for sd in range(3):
                gr = torch.Generator().manual_seed(sd)
                perm = torch.randperm(NUNITS, generator=gr).tolist()
                pick, tot = [], 0.0
                for i in perm:
                    if tot + float(en_o[i]) <= 1.05 * tgt or not pick:
                        pick.append(i)
                        tot += float(en_o[i])
                    if tot >= tgt:
                        break
                arm_r = ablate(Mo, held, DEV, pick, base_o, base_o_seq)
                ref['draws'].append({**arm_r, 'energy_removed': tot,
                                     'picked': pick})
            ref['target_energy_fraction'] = round(f_sh, 4)
            ref['mean_dce'] = round(float(np.mean(
                [d['dce'] for d in ref['draws']])), 5)
        E2['oracle_reference'] = ref
        say(f'  oracle reference mean dCE {ref.get("mean_dce")}')
    save({'E2': E2})

say(f'DONE stages {STAGES} in {time.time() - t0:.1f}s -> {OUT}')
