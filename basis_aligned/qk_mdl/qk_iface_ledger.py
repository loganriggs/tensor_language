"""REALISED-INTERFACE LEDGER (aggregate list #2; predictions registered in
qk_iface_ledger_predictions.json BEFORE this code was written).

Today the mechanism ledger's third moment lives either in pure weight space
(rank-4608 composed tensor, corrected null TIED 0.483 vs 0.485 — BILIN18 sec 5,
qk_composed_cp_uw.py tick 208) or on raw activations. Here instead: for each of
the 36 realised MLP->layer-1-reader channels (tick 203, qk_mlp_datarank.py:
4 maps q1/k1/q2/k2 x 9 heads of layer-1 attention), restrict the block-0 MLP's
output to that channel's REALISED subspace, build the exposure-weighted third
moment of the projected object, sparse-code + symmetric nonnegative CP exactly
as the validated ledger pipeline does, and score against the CORRECTED
transplant null.

EXACT SUBSPACE RULE (uniform across all channels):
  eff = (sum ev)^2 / sum(ev^2) of the channel's held-out activation covariance
  (mean-removed, PCA split of the held-out docs);  r = min(16, max(2, round(eff))).
  The channel output is centered with the PCA-split mean and projected onto the
  top-r principal directions. Dictionary m = 2r atoms, k = 2 active per position
  (annealed 4 -> 2), CP rank R = r. Held-out docs are split in half: first half
  estimates the subspace (PCA split), second half provides codes/moment/tokens
  (moment split) — no position is used for both.

Ported pieces (paths + ticks):
  - activation capture + 36-channel definition : qk_mlp_datarank.py (tick 203)
  - sparse coder (nonneg + k-annealing + dead-atom revival): train_sae in
    qk_null_repair.py (tick 183), adapted from unigram token sampling to uniform
    position sampling (positions are corpus-drawn, so uniform IS exposure).
  - symmetric nonneg CP via tensor power iteration + deflation: cp_fit in
    qk_stage23.py (tick 173; the only solver that passed planted recovery).
  - corrected transplant null + nonneg-lambda refit (eval_on_core):
    qk_null_repair.py (tick 183) — null factors fit on the permuted core, then
    transplanted onto the SAME real core, only lambdas refit. Never two targets.
  - planted known-answer control: qk_cp_planted.py (tick 174), re-shaped to the
    realised-interface scale (core dim 16-32, spiky-sparse supports 2-3).
  - token classes: constructed the same way as build_classes in
    qk_coord_semantics_arch.py (no stored masks found for the four selection
    classes, so they are rebuilt here with the identical recipe and stated).

Deviation from the token-ledger null: the null core is rebuilt from the FULL
column-permuted dense code matrix (no top-k re-sparsification cap). This is
exact rather than capped, makes the null slightly stronger, i.e. conservative
for M1.

M3 gate runs FIRST: if tensor power iteration + deflation fails planted
recovery (mean matched cos < 0.99 on any shape), the script writes the failure
and exits — no real-data verdict counts.

--smoke (or QK_SMOKE=1): planted control at full fidelity + 3 channels
(q1_h3 top-causal reader, k1_h1 subword giant, q1_h7 determiner reader) on a
reduced sample; writes qk_iface_ledger_smoke.json, never the real output.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot

SMOKE = ('--smoke' in sys.argv) or (os.environ.get('QK_SMOKE') == '1')
torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_iface_ledger_smoke.json' if SMOKE else f'{QK}/qk_iface_ledger.json'

N_DOCS = 16 if SMOKE else 256          # halved into PCA split / moment split
SAE_STEPS = 600 if SMOKE else 6000
SAE_BATCH = 2048 if SMOKE else 4096
SAE_LR = 3e-3
MIN_TOK_COUNT = 2 if SMOKE else 5
RESTARTS = 3
MARGIN = 0.15
TOPK_TOKENS = 20
GATE_COS = 0.99

t0 = time.time()


def say(msg):
    print(f'[{time.time() - t0:7.1f}s] {msg}', flush=True)


# ----------------------------------------------------------------------------
# CP machinery (ports; see module docstring for provenance)
# ----------------------------------------------------------------------------
def cp_fit(core_raw, R, seed, n_starts=8, iters=60):
    """Symmetric nonneg CP via tensor power iteration + deflation — verbatim port
    of cp_fit in qk_stage23.py (the only solver that passed planted recovery)."""
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
    """Nonneg lambda refit on THIS (unit-norm) core for fixed factors; rel-err.
    Port of eval_on_core in qk_null_repair.py (dense-core form)."""
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
    """Matched-component |cos| across restarts — port of stability in qk_stage23.py."""
    vals = []
    for i in range(len(Us)):
        for j in range(i + 1, len(Us)):
            C = (Us[i].T @ Us[j]).abs()
            vals.append(float(C.max(1).values.mean()))
    return sum(vals) / len(vals)


# ----------------------------------------------------------------------------
# M3 — planted known-answer control (GATE, runs first).
# Shapes matched to the realised interface: core dim 16-32 (= 2r for r 8-16),
# spiky-sparse components (supports 2-3 of m, like the real archetypes' 6/512
# density scaled down), 1% dense noise — planting recipe ported from
# qk_cp_planted.py (tick 174) with two STATED deviations, both needed to keep
# the planted problem identifiable at interface scale (they tune the PLANT, not
# the solver): (1) lambdas log-spread over ONE decade, not two — at m=16-32 the
# 1% dense noise concentrates in 4k-33k cells instead of 134M, so a 100x-spread
# weakest component sits AT the noise floor and no solver could recover it;
# (2) supports drawn DISJOINT — random 6-of-512 supports in the validated
# planted test were effectively disjoint, and at 16-32 dims unconstrained (or
# even overlap<=1) support-2/3 draws produce component cosines up to ~0.7,
# breaking CP uniqueness (Kruskal), i.e. not identifiable by ANY solver.
# ----------------------------------------------------------------------------
def plant_core(mdim, R0, support, seed):
    g = torch.Generator().manual_seed(seed)
    MU = torch.zeros(mdim, R0)
    used = set()
    for r in range(R0):
        while True:
            sup = torch.randperm(mdim, generator=g)[:support]
            if not (set(sup.tolist()) & used):
                break
        used |= set(sup.tolist())
        MU[sup, r] = 0.3 + torch.rand(support, generator=g)
    MU = MU / MU.norm(dim=0, keepdim=True)
    LAM = 10 ** torch.rand(R0, generator=g)
    W = MU * LAM ** (1 / 3)
    core = torch.einsum('ar,br,cr->abc', W, W, W)
    core = core + 0.01 * core.norm() / mdim ** 1.5 * torch.randn(
        mdim, mdim, mdim, generator=g)
    return core / core.norm(), MU


def matched_cos(U, MU):
    C = (U.T @ MU).abs()
    return float(C.max(0).values.mean())


def fit_als_sym(core, R, seed, iters=200):
    """Symmetric projected ALS (normal-equations solve + clamp>=0) — the solver
    family that failed planted recovery at layer 0; run alongside for M3."""
    mdim = core.shape[0]
    g = torch.Generator().manual_seed(seed)
    A = (0.01 + torch.rand(mdim, R, generator=g)).to(core.device) * 0.05
    M1 = core.reshape(mdim, mdim * mdim)
    I = torch.eye(R, device=core.device)
    for _ in range(iters):
        KR = (A[:, None, :] * A[None, :, :]).reshape(mdim * mdim, R)
        G = (A.T @ A) ** 2
        A = torch.linalg.solve(G + 1e-9 * I, (M1 @ KR).T).T.clamp_min(0)
    lam = A.norm(dim=0).clamp_min(1e-12)
    return A / lam[None, :]


def fit_mult(core, R, seed, iters=400):
    """Lee-Seung multiplicative nonneg symmetric CP — second comparison variant,
    port of fit_mult in qk_cp_planted.py."""
    mdim = core.shape[0]
    g = torch.Generator().manual_seed(seed)
    A = (0.01 + torch.rand(mdim, R, generator=g)).to(core.device) * 0.05
    M1 = core.reshape(mdim, mdim * mdim)
    for _ in range(iters):
        KR = (A[:, None, :] * A[None, :, :]).reshape(mdim * mdim, R)
        num = (M1 @ KR).clamp_min(0)
        den = A @ (A.T @ A) ** 2 + 1e-12
        A = A * (num / den) ** (1 / 3)
    lam = A.norm(dim=0).clamp_min(1e-12)
    return A / lam[None, :]


say(f'=== realised-interface ledger ({"SMOKE" if SMOKE else "FULL"}) on {DEV} ===')
say('M3 planted known-answer control (gate) — runs before any real data')
PLANT_SHAPES = [(16, 6, 2), (24, 6, 3), (32, 8, 3)]   # (core dim m, R0, support)
planted = {}
gate_primary = True
als_failed_somewhere = False
for mdim, R0, support in PLANT_SHAPES:
    core_p, MU = plant_core(mdim, R0, support, seed=5)
    U_tpi = cp_fit(core_p, R0, seed=0)
    rel_tpi, _ = eval_on_core(core_p, U_tpi)
    c_tpi = matched_cos(U_tpi, MU)
    U_als = fit_als_sym(core_p, R0, seed=0)
    rel_als, _ = eval_on_core(core_p, U_als)
    c_als = matched_cos(U_als, MU)
    U_mul = fit_mult(core_p, R0, seed=0)
    rel_mul, _ = eval_on_core(core_p, U_mul)
    c_mul = matched_cos(U_mul, MU)
    row = {'m': mdim, 'R0': R0, 'support': support,
           'tpi_matched_cos': round(c_tpi, 4), 'tpi_relerr': round(rel_tpi, 4),
           'als_matched_cos': round(c_als, 4), 'als_relerr': round(rel_als, 4),
           'mult_matched_cos': round(c_mul, 4), 'mult_relerr': round(rel_mul, 4)}
    planted[f'm{mdim}_R{R0}_s{support}'] = row
    if c_tpi < GATE_COS:
        gate_primary = False
    if c_als < GATE_COS or c_mul < GATE_COS:
        als_failed_somewhere = True
    say(f'  planted m={mdim} R0={R0} sup={support}: TPI cos {c_tpi:.4f} rel {rel_tpi:.3f}'
        f' | ALS cos {c_als:.4f} rel {rel_als:.3f} | mult cos {c_mul:.4f} rel {rel_mul:.3f}'
        f' -> primary {"PASS" if c_tpi >= GATE_COS else "FAIL"}')

result = {
    'mode': 'smoke' if SMOKE else 'full',
    'config': {
        'rank_rule': 'r = min(16, max(2, round(eff))), eff = (sum ev)^2/sum ev^2 of '
                     'mean-removed held-out channel covariance (PCA split)',
        'dict_rule': 'm_atoms = 2r, k = 2 (annealed 4->2), CP rank R = r, '
                     'restarts = 3 (seeds 0,1,2), n_starts = 8, power iters = 60',
        'null': 'corrected transplant: factors fit on column-permuted-code core, '
                'transplanted onto the SAME real core, nonneg lambdas refit only',
        'n_docs': N_DOCS, 'sae_steps': SAE_STEPS, 'sae_batch': SAE_BATCH,
        'margin_threshold': MARGIN, 'gate_cos': GATE_COS,
        'min_token_count': MIN_TOK_COUNT, 'top_k_tokens': TOPK_TOKENS,
    },
    'planted_control': {'detail': planted,
                        'primary_solver_pass': gate_primary,
                        'an_als_variant_failed': als_failed_somewhere},
    'gates': {'M3_primary_solver_planted_recovery': 'PASS' if gate_primary else 'FAIL'},
}
if not gate_primary:
    result['verdict'] = ('GATE FAILED: tensor power iteration with deflation did not '
                         'recover planted components at matched cosine >= 0.99 on '
                         'interface-shaped cores. No real-data verdict counts. STOP.')
    json.dump(result, open(OUT, 'w'), indent=2)
    say('M3 GATE FAILED — stopping before any real-data fitting.')
    sys.exit(1)
say('M3 gate OPEN (primary solver passed planted recovery on all shapes)')

# ----------------------------------------------------------------------------
# Real pipeline — activation capture (port of qk_mlp_datarank.py, tick 203),
# model loaded on CPU with only wte + block 0 moved to DEV (keeps GPU footprint
# small; layer-1 reader weights used as matrices only).
# ----------------------------------------------------------------------------
say('loading bilin18 (cpu) and moving wte + block0 to device...')
model, cfg = load_elriggs('bilin18', device='cpu')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
model.transformer.wte.to(DEV)
blk0 = model.transformer.h[0].to(DEV)
mlp = blk0.mlp
a1 = model.transformer.h[1].attn
W_MAPS = {name: lin.weight.detach().float().to(DEV)
          for name, lin in (('q1', a1.c_q), ('k1', a1.c_k),
                            ('q2', a1.c_q2), ('k2', a1.c_k2))}
COOC = torch.from_numpy(np.load(
    '/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))


@torch.no_grad()
def mlp_out_batch(idx):
    dt = model.transformer.wte.weight.dtype
    x = model.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    B, T = idx.shape
    x = blk0.lambdas[0] * x + blk0.lambdas[1] * x0
    a = blk0.attn
    hcur = F.rms_norm(x, (x.size(-1),))
    cos, sin = rope_tables(T, HD, idx.device, dt, 'bf16')
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]

    def qk(lin):
        z = lin(hcur).view(B, T, NH, HD)
        return apply_rot(F.rms_norm(z, (HD,)), cos, sin)

    v = a.c_v(hcur).view(B, T, NH, HD)
    mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
    q, k = qk(a.c_q), qk(a.c_k)
    q2, k2 = qk(a.c_q2), qk(a.c_k2)
    s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
    s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
    pat = (s1 * s2).masked_fill(~mask, 0.0)
    y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1)
    x = x + a.c_proj(y)
    return mlp(F.rms_norm(x, (x.size(-1),)))          # (B, T, D) MLP output


say(f'capturing MLP outputs on {N_DOCS} held-out docs...')
OUTS, IDS = [], []
with torch.no_grad():
    for i in range(0, N_DOCS, 4):
        b = COOC[i:i + 4].to(DEV)
        idx = b[:, :-1]
        OUTS.append(mlp_out_batch(idx).float().reshape(-1, D).cpu())
        IDS.append(idx.reshape(-1).cpu())
MO = torch.cat(OUTS)
IDS_ALL = torch.cat(IDS)
N_TOT = MO.shape[0]
HALF = N_TOT // 2                       # PCA split = first half, moment split = second
IDS_B = IDS_ALL[HALF:].to(DEV)
N_B = N_TOT - HALF
say(f'{N_TOT} positions ({HALF} PCA split / {N_B} moment split)')

# token classes — same recipe as build_classes in qk_coord_semantics_arch.py;
# the four selection classes named at the two ledgers (BILIN18 sec 3/6):
# boundary (newline/document boundary), punctuation, function words
# (determiner/preposition/conjunction/pronoun/be/modal — covers {the},{a/an},
# {of},{and}), subword fragments (no leading space, lowercase alpha start).
say('building token classes...')
from transformers import AutoTokenizer
import string as _string
tok = AutoTokenizer.from_pretrained('gpt2')
_P = set(_string.punctuation)
FUNC_WORDS = set().union(
    {'and', 'or', 'but', 'because', 'while', 'as', 'though', 'although', 'whereas',
     'nor', 'so', 'yet', 'since', 'if', 'when', 'unless', 'whether'},
    {'to', 'in', 'on', 'for', 'with', 'at', 'by', 'from', 'of', 'into', 'over',
     'under', 'about', 'through', 'between', 'during', 'against', 'among'},
    {'you', 'your', 'they', 'we', 'my', 'he', 'i', 'she', 'it', 'us', 'them',
     'our', 'his', 'her', 'me', 'him', 'their', 'its'},
    {'a', 'an', 'the', 'this', 'that', 'these', 'those', 'some', 'any', 'no',
     'each', 'every'},
    {'is', 'are', 'was', 'were', 'be', 'been', 'being', 'am'},
    {'will', 'would', 'can', 'could', 'may', 'might', 'must', 'shall', 'should'})
CLASS_MASKS = {c: torch.zeros(V, dtype=torch.bool) for c in
               ['boundary', 'punctuation', 'function_word', 'subword_fragment']}
for i in range(50257):
    s = tok.convert_ids_to_tokens(i)
    if s is None:
        continue
    core_s = s.replace('Ġ', '')
    lead = s.startswith('Ġ')
    if 'Ċ' in s or s == '<|endoftext|>':
        CLASS_MASKS['boundary'][i] = True
    if len(core_s) and all(c in _P for c in core_s):
        CLASS_MASKS['punctuation'][i] = True
    if core_s.lower() in FUNC_WORDS:
        CLASS_MASKS['function_word'][i] = True
    if (not lead) and len(core_s) and core_s[0].isalpha() and core_s[0].islower():
        CLASS_MASKS['subword_fragment'][i] = True
CLASS_EXPOSURE = {c: float(mask[IDS_B.cpu()].float().mean())
                  for c, mask in CLASS_MASKS.items()}
say('class exposure mass on moment split: ' +
    ' '.join(f'{c}={v:.4f}' for c, v in CLASS_EXPOSURE.items()))


def train_sae(Z, m_atoms, k_code, seed=0):
    """Nonneg k-sparse coder with k-annealing + dead-atom revival — port of
    train_sae in qk_null_repair.py (tick 183), unigram token sampling replaced
    by uniform position sampling (= exposure). Returns dense codes (N, m)."""
    N = Z.shape[0]
    g = torch.Generator(device='cpu').manual_seed(seed)
    Dm = Z[torch.randperm(N, generator=g)[:m_atoms].to(Z.device)].clone()
    Dm = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
    We = Dm.clone()
    b = Z.mean(0).clone()
    for t in (Dm, We, b):
        t.requires_grad_(True)
    opt = torch.optim.Adam([Dm, We, b], lr=SAE_LR)
    fired = torch.zeros(m_atoms, device=Z.device)
    for step in range(SAE_STEPS):
        kk = max(k_code, int(round(2 * k_code - k_code * min(1.0, 2 * step / SAE_STEPS))))
        bi = torch.randint(0, N, (SAE_BATCH,), generator=g).to(Z.device)
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
                    Dm.data[dead] = Z[worst] / Z[worst].norm(dim=1, keepdim=True).clamp(min=1e-8)
                    We.data[dead] = Dm.data[dead]
                    del z_, rec
            fired.zero_()
    with torch.no_grad():
        z = torch.relu((Z - b) @ We.T)
        vals, idx = z.topk(min(k_code, m_atoms), dim=1)
        S = torch.zeros(N, m_atoms, device=Z.device)
        S.scatter_(1, idx, vals)
    return S


def build_core(S):
    """Exposure-weighted third moment in code space: (1/N) sum_n s x s x s."""
    N, mdim = S.shape
    core = torch.zeros(mdim, mdim, mdim, device=S.device)
    for i in range(0, N, 8192):
        s = S[i:i + 8192]
        core += torch.einsum('na,nb,nc->abc', s, s, s)
    return core / N


CHANNELS = ([('q1', 3), ('k1', 1), ('q1', 7)] if SMOKE else
            [(mn, h) for mn in ('q1', 'k1', 'q2', 'k2') for h in range(NH)])
chan_results = {}
n_pass = 0
margins = []
for mapname, h in CHANNELS:
    name = f'{mapname}_h{h}'
    W = W_MAPS[mapname][h * HD:(h + 1) * HD]           # (128, D)
    # channel outputs (chunked CPU->GPU matmul)
    Ys = []
    for i in range(0, N_TOT, 16384):
        Ys.append(MO[i:i + 16384].to(DEV) @ W.T)
    Y = torch.cat(Ys)                                  # (N_TOT, 128) on DEV
    Y_A, Y_B = Y[:HALF], Y[HALF:]
    mean_A = Y_A.mean(0)
    Yc = Y_A - mean_A
    C = Yc.T @ Yc / HALF
    ev, evec = torch.linalg.eigh(C)
    ev = ev.clamp_min(0)
    eff = float(ev.sum() ** 2 / (ev ** 2).sum().clamp_min(1e-30))
    r = min(16, max(2, int(round(eff))))
    P = evec[:, -r:]                                   # top-r principal directions
    Z = (Y_B - mean_A) @ P                             # (N_B, r) projected object
    del Y, Y_A, Y_B, Yc, C, Ys
    m_atoms, k_code = 2 * r, 2
    S = train_sae(Z, m_atoms, k_code, seed=0)
    core = build_core(S)
    # CP fitting runs on CPU: the cores are tiny (m <= 32) and GPU execution is
    # kernel-launch-bound at this size (measured ~20s/shape on the planted test)
    core_n = (core / core.norm().clamp_min(1e-30)).cpu()
    # real fit: 3 restarts, nonneg-lambda refit on the real core, keep the best
    Us, rels = [], []
    for seed in range(RESTARTS):
        U = cp_fit(core_n, r, seed)
        rel, _ = eval_on_core(core_n, U)
        Us.append(U)
        rels.append(rel)
    best = int(torch.tensor(rels).argmin())
    U_best = Us[best]
    real_fit = rels[best]
    stab = stability(Us)
    # corrected transplant null: column-permute codes over positions (destroys
    # within-position co-occurrence, preserves marginals), fit on the null core,
    # transplant onto the SAME real core, refit nonneg lambdas only
    gp = torch.Generator().manual_seed(7)
    S_null = S.clone()
    for f in range(m_atoms):
        S_null[:, f] = S_null[torch.randperm(S.shape[0], generator=gp).to(S.device), f]
    core_null = build_core(S_null)
    U_null = cp_fit((core_null / core_null.norm().clamp_min(1e-30)).cpu(), r, 0)
    null_on_real, _ = eval_on_core(core_n, U_null)
    margin = null_on_real - real_fit
    m1_pass = bool(margin >= MARGIN)
    n_pass += int(m1_pass)
    margins.append(margin)
    # M2: per-token mean code (exposure attribution), top tokens per component
    _, lam_best = eval_on_core(core_n, U_best)
    cnt = torch.bincount(IDS_B, minlength=V).float()
    sums = torch.zeros(V, m_atoms, device=DEV)
    sums.index_add_(0, IDS_B, S)
    mean_code = sums / cnt[:, None].clamp_min(1)
    okmask = cnt >= MIN_TOK_COUNT
    comps = []
    order = lam_best.argsort(descending=True)
    for ri in order.tolist():
        if float(lam_best[ri]) <= 0:
            continue
        load = mean_code @ U_best[:, ri].to(DEV)
        load[~okmask] = -1e30
        top = load.argsort(descending=True)[:TOPK_TOKENS]
        top_ids = top.tolist()
        top_str = [tok.decode([t]).replace('\n', '\\n') for t in top_ids[:12]]
        enrich = {}
        for cname, mask in CLASS_MASKS.items():
            frac = float(mask[torch.tensor(top_ids)].float().mean())
            base = CLASS_EXPOSURE[cname]
            enrich[cname] = {'top_frac': round(frac, 3),
                             'exposure_base': round(base, 5),
                             'ratio': (round(frac / base, 2) if base > 1e-6 else None)}
        comps.append({'lambda': round(float(lam_best[ri]), 4),
                      'top_tokens': top_str, 'enrichment': enrich})
    chan_results[name] = {
        'eff_rank_measured': round(eff, 2), 'r_used': r,
        'm_atoms': m_atoms, 'k_code': k_code, 'cp_rank': r,
        'real_fit': round(real_fit, 4), 'null_on_real': round(null_on_real, 4),
        'margin': round(margin, 4), 'm1_pass': m1_pass,
        'restart_stability': round(stab, 3), 'restart_relerrs': [round(x, 4) for x in rels],
        'components': comps,
    }
    say(f'{name}: eff {eff:.1f} r={r} | real {real_fit:.4f} null-on-real '
        f'{null_on_real:.4f} margin {margin:+.4f} {"PASS" if m1_pass else "fail"} '
        f'| stability {stab:.3f}')
    json.dump(result | {'channels': chan_results}, open(OUT, 'w'), indent=2)
    del S, S_null, core, core_n, core_null, Z, sums, mean_code
    if DEV == 'cuda':
        torch.cuda.empty_cache()

result['channels'] = chan_results
result['m1_summary'] = {
    'n_channels': len(CHANNELS),
    'n_pass_margin_0.15': n_pass,
    'median_margin': round(float(np.median(margins)), 4) if margins else None,
    'majority_pass': bool(n_pass > len(CHANNELS) / 2),
}
result['gates']['M1_majority_beats_null'] = ('PASS' if n_pass > len(CHANNELS) / 2
                                             else 'FAIL')
result['gates']['M2_computed_on_validated_channels_only'] = (
    'enrichments stored for every channel; only m1_pass channels count for M2')
json.dump(result, open(OUT, 'w'), indent=2)
say(f'M1: {n_pass}/{len(CHANNELS)} channels beat the corrected null by >= {MARGIN}')
say(f'IFACE LEDGER DONE -> {OUT}')
