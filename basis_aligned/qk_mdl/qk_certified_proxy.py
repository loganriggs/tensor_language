"""CERTIFIED RESTRICTED-CORE PROXY (fold-audit top upgrade).

The recurring lesson (§56/§67): the LINEAR direct-to-logits proxy -- map a path's
output movement straight through the unembedding, ignoring downstream layers -- is
wrong in magnitude, sign, and case, because it linearizes through NONLINEAR
downstream computation (bilinear MLPs + quadratic attention patterns). §85
(qk_hub_maprestrict) showed each layer's MLP MAP restricts to a compact core
(input top-K_in=288 x output top-K_out=144 train-gram directions) at small cost.

THIS SCRIPT tests whether PROPAGATING a candidate perturbation through the actual
downstream computation with every downstream MLP replaced by its restricted core
(attention exact) predicts TRUE causal effects (real mean-ablation delta
cross-entropy) far better than the linear proxy.

For ~24 census candidates spanning the causal range (high / mid / null / negative,
incl. the §67 misranked h.L16.2):
  (1) TRUE effect  = full-model mean-ablation dCE, global, on a 32-seq subsample of
      held FW[448:600] (+ census full-slice values as reference; 4 recomputed on the
      FULL held slice to verify the harness).
  (2) LINEAR proxy = direct-to-logits: the exact per-position residual perturbation
      the ablation injects at the path's layer, added STRAIGHT to the final residual
      (downstream layers = identity), then the true readout (final rms_norm +
      lm_head + 30*tanh cap) -> dCE. This is the strongest fair version of the
      existing top_boost / class-summed direct-to-logits construction (same direct
      path, but per-position and through the actual readout).
  (3) RESTRICTED-CORE proxy = mean-ablate at the path's layer, propagate through
      remaining layers with downstream MLPs replaced by FOLDED restricted cores
      (K_in=288 x K_out=144, train-gram bases, per-position held means -- map
      restriction VERBATIM in math from qk_hub_maprestrict.py, folded into small
      matrices for genuine FLOP reduction), attention exact; paired against a
      same-restriction no-ablation base -> dCE.
Report Spearman/sign agreement vs truth for (2) and (3), the §67 failure cases at
their census trigger positions on the FULL held slice, and per-candidate cost.

FORWARD + mean-ablation + means + CE conventions VERBATIM from
qk_census_difficulty.py / qk_unsup_classpush.py; map-restriction math VERBATIM from
qk_hub_maprestrict.py; paired standard errors as in qk_unsup_verify.py
(std(ddof=1)/sqrt(n) over per-position paired diffs).
"""
import json, sys, math, time, subprocess
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'

# ---------------- GPU GUARD (verbatim from census) ----------------
def gpu_guard(min_free=4500, tries=45, sleep=20):
    for _ in range(tries):
        free = int(subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits']
        ).decode().split('\n')[0].strip())
        if free >= min_free:
            print(f"GPU guard: {free} MiB free -- proceeding.", flush=True); return
        print(f"GPU guard: only {free} MiB free (<{min_free}); sleeping {sleep}s ...", flush=True)
        time.sleep(sleep)
    raise RuntimeError("GPU guard timed out waiting for free memory")
gpu_guard()

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h); N_SVD = 4
DFF = cfg['expansion_factor'] * D
assert not cfg.get('gated', False), "folded core assumes ungated bilinear MLP"
tok = AutoTokenizer.from_pretrained('gpt2')
print(f"bilin18 NL={NL} NH={NH} HD={HD} D={D} DFF={DFF} V={V}", flush=True)

FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQL = 128
TRAIN = FINEWEB[0:256, :SEQL].to(DEV)        # discovery slice -- grams (path defn + restriction bases)
HELD = FINEWEB[448:600, :SEQL].to(DEV)       # held-back verification slice
NHELD = HELD.shape[0]
BATCH = 6
KCAUSAL = 200                                # census trigger top-K (verbatim)
K_IN, K_OUT = 288, 144                       # §85 restricted-core ranks

# 32-sequence evenly spaced subsample of the held slice (proxy evaluation set)
SUBIDX = np.unique(np.round(np.linspace(0, NHELD - 1, 32)).astype(int))
assert len(SUBIDX) == 32
SUB = HELD[torch.from_numpy(SUBIDX).to(DEV)]
NSUB = SUB.shape[0]
print(f"subsample: {NSUB} held sequences (evenly spaced rows of FW[448:600])", flush=True)

# special tokens (verbatim census/verify)
_special = {tok.eos_token_id}
for _t in range(min(V, 50257)):
    if '�' in tok.decode([_t]): _special.add(_t)
SPECIAL = np.array(sorted(_special))

# =====================================================================================
# CANDIDATES: 24 census paths spanning the causal range (global mean-ablation dCE).
# =====================================================================================
CAND_NAMES = [
    # high-causal (census top-8 by global dCE)
    'mlp.L16.d0', 'mlp.L17.d0', 'h.L0.3', 'mlp.L17.d1', 'mlp.L17.d2', 'mlp.L16.d2',
    'h.L1.1', 'h.L6.3',
    # clean winners (early/mid, real effects)
    'h.L3.3', 'h.L9.6', 'h.L2.1',
    # §67 misranked clean-but-null (h.L16.2 = the cleanest path, NEGATIVE true effect)
    'h.L16.2', 'h.L8.2', 'mlp.L1.d3', 'h.L14.4',
    # negative global dCE
    'h.L14.2', 'h.L2.4', 'h.L12.7',
    # mid-range
    'h.L9.7', 'h.L4.5', 'h.L2.6',
    # near-null
    'h.L8.0', 'h.L15.5', 'mlp.L7.d3',
]
def parse(comp):
    if comp.startswith('h.'):
        _, l, h = comp.split('.'); return ('head', int(l[1:]), int(h))
    _, l, d = comp.split('.'); return ('mlp', int(l[1:]), int(d[1:]))
CANDS = [(c,) + parse(c) for c in CAND_NAMES]           # (comp, kind, li, ix)
# restriction key: first downstream MLP layer to restrict (head at li -> li; mlp at li -> li+1)
def rkey(kind, li): return li if kind == 'head' else li + 1
KEYS = sorted({rkey(k, l) for _, k, l, _ in CANDS if rkey(k, l) < NL})
print(f"{len(CANDS)} candidates; {len(KEYS)} distinct restriction keys: {KEYS}", flush=True)

census = json.load(open(f'{QK}/qk_census_difficulty.json'))
cen = {r['comp']: r for r in census['records']}
for c in CAND_NAMES: assert c in cen, c

# =====================================================================================
# TRAIN gram pass: per-layer MLP-output gram (VERBATIM census -> mlp_dirs; also the §85
# OUTPUT restriction gram) + per-layer MLP-INPUT gram (post-attn residual x, VERBATIM
# qk_hub_maprestrict gin).
# =====================================================================================
gram_in = [torch.zeros(D, D, device=DEV) for _ in range(NL)]
gram_out = [torch.zeros(D, D, device=DEV) for _ in range(NL)]

@torch.no_grad()
def fwd_gram(idx):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        gram_in[li] += torch.einsum('btd,bte->de', x, x)
        mo = blk.mlp(F.rms_norm(x, (D,)))
        gram_out[li] += torch.einsum('btd,bte->de', mo, mo)
        x = x + mo

print("TRAIN gram pass (input + output grams, all layers) ...", flush=True)
for i in range(0, TRAIN.shape[0], BATCH): fwd_gram(TRAIN[i:i+BATCH])
mlp_dirs = torch.zeros(NL, N_SVD, D, device=DEV)
PIN = {}; POUT = {}
for li in range(NL):
    _e, evecs = torch.linalg.eigh(gram_out[li])
    mlp_dirs[li] = evecs[:, -N_SVD:].T.flip(0)              # verbatim census path defn
    POUT[li] = evecs.flip(1)[:, :K_OUT].contiguous()        # §85 output basis
    _e, evecs = torch.linalg.eigh(gram_in[li])
    PIN[li] = evecs.flip(1)[:, :K_IN].contiguous()          # §85 input basis
del gram_in, gram_out
torch.cuda.empty_cache()
print("bases + MLP directions ready.", flush=True)

# =====================================================================================
# HELD PASS A: per-position means (yh, mlp-proj -- VERBATIM census; plus per-layer MLP
# input/output means for the restriction, as in qk_hub_maprestrict but per layer) and
# activation magnitudes for the CANDIDATE paths only (census trigger selection).
# =====================================================================================
YH_SUM = {li: torch.zeros(SEQL, NH, HD, device=DEV) for li in range(NL)}
PROJ_SUM = {li: torch.zeros(SEQL, N_SVD, device=DEV) for li in range(NL)}
X_SUM = {li: torch.zeros(SEQL, D, device=DEV) for li in range(NL)}
MO_SUM = {li: torch.zeros(SEQL, D, device=DEV) for li in range(NL)}
cand_heads = sorted({(l, i) for _, k, l, i in CANDS if k == 'head'})
cand_mlps = sorted({(l, i) for _, k, l, i in CANDS if k == 'mlp'})
head_act = {hl: np.zeros((NHELD, SEQL), np.float32) for hl in cand_heads}
mlp_act = {ml: np.zeros((NHELD, SEQL), np.float32) for ml in cand_mlps}

@torch.no_grad()
def fwd_passA(idx, i0):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        YH_SUM[li] += yh4.sum(0)
        Wr = a.c_proj.weight.view(D, NH, HD)
        for (l, h) in cand_heads:
            if l == li:
                comp = torch.einsum('btc,dc->btd', yh4[:, :, h], Wr[:, h]).norm(dim=-1)
                head_act[(l, h)][i0:i0+B] = comp.cpu().numpy()
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        X_SUM[li] += x.sum(0)
        mo = blk.mlp(F.rms_norm(x, (D,)))
        MO_SUM[li] += mo.sum(0)
        pr = torch.einsum('btd,nd->btn', mo, mlp_dirs[li])
        PROJ_SUM[li] += pr.sum(0)
        for (l, kk) in cand_mlps:
            if l == li:
                mlp_act[(l, kk)][i0:i0+B] = pr[:, :, kk].abs().cpu().numpy()
        x = x + mo

print("HELD PASS A: per-position means + candidate activation magnitudes ...", flush=True)
for i in range(0, NHELD, BATCH): fwd_passA(HELD[i:i+BATCH], i)
YHMEAN = {li: YH_SUM[li] / NHELD for li in range(NL)}
PROJMEAN = {li: PROJ_SUM[li] / NHELD for li in range(NL)}
MX = {li: X_SUM[li] / NHELD for li in range(NL)}
MOMEAN = {li: MO_SUM[li] / NHELD for li in range(NL)}
del YH_SUM, PROJ_SUM, X_SUM, MO_SUM
print("PASS A done.", flush=True)

# census trigger masks (VERBATIM selection: top-KCAUSAL activation, bad positions excluded)
held_np = HELD.cpu().numpy()
pos_t = np.tile(np.arange(SEQL), NHELD).reshape(NHELD, SEQL)
is_special = np.isin(held_np, SPECIAL)
valid_next = pos_t < (SEQL - 1)
bad_trigger = (pos_t == 0) | is_special | ~valid_next
trig_mask = {}
for (comp, kind, li, ix) in CANDS:
    act = head_act[(li, ix)] if kind == 'head' else mlp_act[(li, ix)]
    a = act.copy().reshape(-1); a[bad_trigger.reshape(-1)] = -1e30
    tk = np.argpartition(a, -KCAUSAL)[-KCAUSAL:]
    mk = np.zeros(NHELD*SEQL, bool); mk[tk] = True
    trig_mask[comp] = mk.reshape(NHELD, SEQL)
del head_act, mlp_act

# =====================================================================================
# FOLDED RESTRICTED CORE per layer (math VERBATIM qk_hub_maprestrict:
#   xr = mx + Pin Pin^T (x - mx);  mo' = mo_mean + Pout Pout^T (mlp(rms_norm(xr)) - mo_mean)
# folded: cc = x @ Pin (K_in); rms scale from ||mx_perp||^2 + ||cc||^2;
#   hidden = (uL + cc AL^T)(uR + cc AR^T)/s^2;  mo' = mo_mean + (hidden Dc^T + bc) Pout^T)
# =====================================================================================
CORE = {}
for li in range(NL):
    blk = m.transformer.h[li]
    Pin, Pout = PIN[li], POUT[li]
    mx = MX[li]                                              # (T,D) per-position held mean
    mx_par = mx @ Pin                                        # (T,K_in)
    mx_perp = mx - mx_par @ Pin.T                            # (T,D)
    WL, WR, WD = blk.mlp.Left.weight, blk.mlp.Right.weight, blk.mlp.Down.weight
    CORE[li] = {
        'Pin': Pin, 'Pout': Pout,
        'perp_sq': (mx_perp*mx_perp).sum(-1),                # (T,)
        'uL': mx_perp @ WL.T, 'uR': mx_perp @ WR.T,          # (T,DFF)
        'AL': WL @ Pin, 'AR': WR @ Pin,                      # (DFF,K_in)
        'Dc': Pout.T @ WD,                                   # (K_out,DFF)
        'bc': (blk.mlp.Down_bias.unsqueeze(0) - MOMEAN[li]) @ Pout,  # (T,K_out)
    }

@torch.no_grad()
def mlp_restricted(li, x):
    """folded restricted-core MLP map; x = post-attn residual (b,T,D)."""
    P = CORE[li]
    cc = x @ P['Pin']                                        # (b,T,K_in) = Pin^T x
    ssq = (P['perp_sq'].unsqueeze(0) + (cc*cc).sum(-1)) / D  # rms^2 of xr
    hL = P['uL'].unsqueeze(0) + cc @ P['AL'].T
    hR = P['uR'].unsqueeze(0) + cc @ P['AR'].T
    hidden = (hL * hR) / ssq.clamp_min(1e-12).unsqueeze(-1)
    moc = hidden @ P['Dc'].T + P['bc'].unsqueeze(0)          # (b,T,K_out)
    return MOMEAN[li].unsqueeze(0) + moc @ P['Pout'].T

# =====================================================================================
# FLEXIBLE FORWARD (census forward VERBATIM + restrict_from + collect for linear proxy)
# =====================================================================================
@torch.no_grad()
def forward_flex(idx, ablate=None, restrict_from=None, collect=False):
    """ablate: None | ('head',li,h) | ('mlp',li,kk) mean-ablation (VERBATIM census).
    restrict_from: None | int k -> every MLP at layer >= k replaced by its folded
    restricted core (attention exact everywhere).
    collect: return (logits, x_final_pre_rms, deltas) where deltas[comp] is the exact
    per-position residual perturbation mean-ablation would inject at comp's layer."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    deltas = {} if collect else None
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        if collect:
            Wr = a.c_proj.weight.view(D, NH, HD)
            for (comp, kind, l, ix) in CANDS:
                if kind == 'head' and l == li:
                    deltas[comp] = torch.einsum('btc,dc->btd',
                        YHMEAN[li][:, ix].unsqueeze(0) - yh4[:, :, ix], Wr[:, ix])
        if ablate is not None and ablate[0] == 'head' and ablate[1] == li:
            yh4 = yh4.clone(); yh4[:, :, ablate[2]] = YHMEAN[li][:, ablate[2]].unsqueeze(0)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        if restrict_from is not None and li >= restrict_from:
            mo = mlp_restricted(li, x)
        else:
            mo = blk.mlp(F.rms_norm(x, (D,)))
        if collect:
            pr_all = torch.einsum('btd,nd->btn', mo, mlp_dirs[li])
            for (comp, kind, l, ix) in CANDS:
                if kind == 'mlp' and l == li:
                    deltas[comp] = (PROJMEAN[li][:, ix].unsqueeze(0)
                                    - pr_all[:, :, ix]).unsqueeze(-1) * mlp_dirs[li, ix]
        if ablate is not None and ablate[0] == 'mlp' and ablate[1] == li:
            kk = ablate[2]
            pr = torch.einsum('btd,d->bt', mo, mlp_dirs[li, kk])
            mo = mo - (pr - PROJMEAN[li][:, kk].unsqueeze(0)).unsqueeze(-1) * mlp_dirs[li, kk]
        x = x + mo
    xfinal = x
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    return (logits, xfinal, deltas) if collect else logits

def ce_of(logits, tgt):
    logp = F.log_softmax(logits[:, :SEQL-1].float(), dim=-1)
    return -logp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)   # (b,T-1)

@torch.no_grad()
def ce_from_resid(xf, tgt):
    """readout (final rms_norm + lm_head + 30*tanh cap) applied to a final residual."""
    lg = 30*torch.tanh(m.lm_head(F.rms_norm(xf[:, :SEQL-1], (D,)))/30).float()
    logp = F.log_softmax(lg, dim=-1)
    return -logp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)

def stats(s, sq, n):
    if n <= 1: return 0.0, 0.0
    mean = s/n; var = max(sq/n - mean*mean, 0.0)*n/(n-1)
    return mean, math.sqrt(var/n)

# =====================================================================================
# POSITIVE CONTROL 1: folded core == unfolded §85 restricted map (one batch, 3 layers).
# =====================================================================================
print("POSITIVE CONTROL: folded core vs unfolded restricted map ...", flush=True)
with torch.no_grad():
    idx = SUB[:4]
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    fold_check = {}
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        if li in (2, 9, 16):
            Pin, Pout = PIN[li], POUT[li]
            xr = MX[li].unsqueeze(0) + ((x - MX[li].unsqueeze(0)) @ Pin) @ Pin.T
            ref = blk.mlp(F.rms_norm(xr, (D,)))
            ref = MOMEAN[li].unsqueeze(0) + ((ref - MOMEAN[li].unsqueeze(0)) @ Pout) @ Pout.T
            got = mlp_restricted(li, x)
            fold_check[li] = {'max_abs_diff': float((ref-got).abs().max()),
                              'ref_scale': float(ref.abs().mean())}
            print(f"  L{li}: max|folded-unfolded|={fold_check[li]['max_abs_diff']:.2e} "
                  f"(mean|ref|={fold_check[li]['ref_scale']:.3f})", flush=True)
        mo = blk.mlp(F.rms_norm(x, (D,)))
        x = x + mo
for li, c in fold_check.items():
    assert c['max_abs_diff'] < 1e-2 * max(1.0, c['ref_scale']*100), f"fold mismatch L{li}"

# =====================================================================================
# POSITIVE CONTROL 2 / HARNESS VERIFICATION: recompute census GLOBAL dCE on the FULL
# held slice for 4 candidates -- must match qk_census_difficulty.json.
# =====================================================================================
print("HARNESS VERIFICATION: recompute full-held global dCE for 4 candidates ...", flush=True)
VER = ['mlp.L16.d0', 'h.L0.3', 'h.L6.3', 'h.L16.2']
ver_acc = {c: [0.0, 0.0, 0] for c in VER}
tgt_all = torch.from_numpy(held_np).to(DEV)
for i in range(0, NHELD, BATCH):
    sb = slice(i, min(i+BATCH, NHELD))
    idx = HELD[sb]; tgt = tgt_all[sb]
    base_ce = ce_of(forward_flex(idx), tgt)
    for c in VER:
        kind, li, ix = parse(c)
        d = (ce_of(forward_flex(idx, ablate=(kind, li, ix)), tgt) - base_ce)
        ver_acc[c][0] += float(d.sum()); ver_acc[c][1] += float((d*d).sum()); ver_acc[c][2] += int(d.numel())
harness = {}
for c in VER:
    mn, se = stats(*ver_acc[c])
    harness[c] = {'recomputed_global_dCE': round(mn, 6), 'SE': round(se, 6),
                  'census_global_dCE': cen[c]['global_dCE'], 'census_SE': cen[c]['global_dCE_SE'],
                  'abs_diff': round(abs(mn - cen[c]['global_dCE']), 6)}
    print(f"  {c:12s} recomputed {mn:+.6f}±{se:.6f}  census {cen[c]['global_dCE']:+.6f}"
          f"  diff {harness[c]['abs_diff']:.6f}", flush=True)

# =====================================================================================
# MAIN LOOP over the 32-seq subsample: TRUE (full ablation), LINEAR proxy, RESTRICTED-
# CORE proxy -- global + census-trigger-position dCE, paired accumulators + timing.
# =====================================================================================
acc = {c: {est: {sc: [0.0, 0.0, 0] for sc in ('g', 't')} for est in ('true', 'lin', 'restr')}
       for c, *_ in CANDS}
rbase_cost = {k: [0.0, 0.0, 0] for k in KEYS}                # CE(restricted base)-CE(base)
t_full = {c: 0.0 for c, *_ in CANDS}; t_restr = {c: 0.0 for c, *_ in CANDS}
t_lin = {c: 0.0 for c, *_ in CANDS}; t_rbase = {k: 0.0 for k in KEYS}
sub_trig = {c: trig_mask[c][SUBIDX] for c, *_ in CANDS}      # (NSUB,SEQL)

def sync(): torch.cuda.synchronize()

print(f"MAIN LOOP: {NSUB} subsample sequences, {len(CANDS)} candidates ...", flush=True)
t0 = time.time()
tgt_sub = SUB
for bi, i in enumerate(range(0, NSUB, BATCH)):
    sb = slice(i, min(i+BATCH, NSUB))
    idx = SUB[sb]; tgt = idx
    base_logits, xfinal, deltas = forward_flex(idx, collect=True)
    base_ce = ce_of(base_logits, tgt); del base_logits
    # restricted bases (one per key, shared across candidates)
    rce = {}
    for k in KEYS:
        sync(); tA = time.time()
        rce[k] = ce_of(forward_flex(idx, restrict_from=k), tgt)
        sync(); t_rbase[k] += time.time() - tA
        dv = (rce[k] - base_ce)
        rbase_cost[k][0] += float(dv.sum()); rbase_cost[k][1] += float((dv*dv).sum()); rbase_cost[k][2] += int(dv.numel())
    for (comp, kind, li, ix) in CANDS:
        key = rkey(kind, li)
        # TRUE: full-model ablation
        sync(); tA = time.time()
        d_true = ce_of(forward_flex(idx, ablate=(kind, li, ix)), tgt) - base_ce
        sync(); t_full[comp] += time.time() - tA
        # LINEAR: perturbation straight to readout (downstream = identity)
        sync(); tA = time.time()
        d_lin = ce_from_resid(xfinal + deltas[comp], tgt) - base_ce
        sync(); t_lin[comp] += time.time() - tA
        # RESTRICTED-CORE: ablate at layer, propagate through restricted downstream
        if key >= NL:                                        # no downstream MLPs (mlp.L17.*)
            d_restr = d_true                                 # restricted forward == full forward
        else:
            sync(); tA = time.time()
            d_restr = ce_of(forward_flex(idx, ablate=(kind, li, ix), restrict_from=key), tgt) - rce[key]
            sync(); t_restr[comp] += time.time() - tA
        tm = torch.from_numpy(sub_trig[comp][sb, :SEQL-1]).to(DEV)
        for est, d in (('true', d_true), ('lin', d_lin), ('restr', d_restr)):
            a_ = acc[comp][est]
            a_['g'][0] += float(d.sum()); a_['g'][1] += float((d*d).sum()); a_['g'][2] += int(d.numel())
            if tm.any():
                dt = d[tm]
                a_['t'][0] += float(dt.sum()); a_['t'][1] += float((dt*dt).sum()); a_['t'][2] += int(dt.numel())
    print(f"  batch {bi+1}/{(NSUB+BATCH-1)//BATCH}  elapsed {time.time()-t0:.0f}s", flush=True)
print(f"MAIN LOOP done in {time.time()-t0:.0f}s", flush=True)

NB = (NSUB + BATCH - 1) // BATCH

# =====================================================================================
# §67 FAILURE CASES at census trigger positions on the FULL held slice (h.L16.2 = the
# cleanest path with NEGATIVE true effect; h.L14.2 = negative global, positive trigger).
# =====================================================================================
FOCUS = ['h.L16.2', 'h.L14.2']
print("FOCUS: §67 failure cases, trigger positions, FULL held slice ...", flush=True)
focus = {}
for comp in FOCUS:
    kind, li, ix = parse(comp); key = rkey(kind, li)
    mk_all = trig_mask[comp]
    seqs = np.where(mk_all[:, :SEQL-1].any(axis=1))[0]
    ac = {est: [0.0, 0.0, 0] for est in ('true', 'lin', 'restr')}
    for i in range(0, len(seqs), BATCH):
        sb = seqs[i:i+BATCH]; idx = HELD[sb]; tgt = tgt_all[sb]
        _bl, xfinal, deltas = forward_flex(idx, collect=True)
        base_ce = ce_of(_bl, tgt); del _bl
        r_base = ce_of(forward_flex(idx, restrict_from=key), tgt)
        d_true = ce_of(forward_flex(idx, ablate=(kind, li, ix)), tgt) - base_ce
        d_lin = ce_from_resid(xfinal + deltas[comp], tgt) - base_ce
        d_restr = ce_of(forward_flex(idx, ablate=(kind, li, ix), restrict_from=key), tgt) - r_base
        tm = torch.from_numpy(mk_all[sb, :SEQL-1]).to(DEV)
        for est, d in (('true', d_true), ('lin', d_lin), ('restr', d_restr)):
            dt = d[tm]
            ac[est][0] += float(dt.sum()); ac[est][1] += float((dt*dt).sum()); ac[est][2] += int(dt.numel())
    focus[comp] = {}
    for est in ('true', 'lin', 'restr'):
        mn, se = stats(*ac[est])
        focus[comp][est] = {'trigger_dCE': round(mn, 5), 'SE': round(se, 5),
                            'z': round(mn/se, 2) if se > 0 else 0.0, 'n': ac[est][2]}
    focus[comp]['census_trigger_dCE'] = cen[comp]['trigger_dCE']
    focus[comp]['census_trigger_dCE_z'] = cen[comp]['trigger_dCE_z']
    focus[comp]['cleanliness'] = cen[comp]['cleanliness']
    print(f"  {comp}: true {focus[comp]['true']['trigger_dCE']:+.4f}±{focus[comp]['true']['SE']:.4f}"
          f"  linear {focus[comp]['lin']['trigger_dCE']:+.4f}±{focus[comp]['lin']['SE']:.4f}"
          f"  restricted {focus[comp]['restr']['trigger_dCE']:+.4f}±{focus[comp]['restr']['SE']:.4f}"
          f"  (census true {cen[comp]['trigger_dCE']:+.4f})", flush=True)

# =====================================================================================
# ASSEMBLE per-candidate records + metrics
# =====================================================================================
records = []
for (comp, kind, li, ix) in CANDS:
    key = rkey(kind, li)
    r = {'comp': comp, 'kind': kind, 'li': li, 'idx': ix,
         'n_restricted_downstream_mlps': max(0, NL - key),
         'census_global_dCE': cen[comp]['global_dCE'], 'census_global_dCE_SE': cen[comp]['global_dCE_SE'],
         'census_trigger_dCE': cen[comp]['trigger_dCE'], 'cleanliness': cen[comp]['cleanliness']}
    for est in ('true', 'lin', 'restr'):
        for sc, tag in (('g', 'global'), ('t', 'trigger')):
            mn, se = stats(*acc[comp][est][sc])
            r[f'{est}_{tag}_dCE'] = round(mn, 6); r[f'{est}_{tag}_SE'] = round(se, 6)
            r[f'{est}_{tag}_z'] = round(mn/se, 2) if se > 0 else 0.0
    # cost (per candidate, whole subsample)
    r['sec_full_ablation'] = round(t_full[comp], 3)
    r['sec_restricted_ablation'] = round(t_restr[comp], 3) if key < NL else None
    r['sec_linear_proxy'] = round(t_lin[comp], 3)
    records.append(r)

def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() == 0 or b.std() == 0: return 0.0
    return float(np.corrcoef(a, b)[0, 1])
def spearman(a, b):
    return pearson(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))

def metric_block(recs, scope):
    tr = np.array([r[f'true_{scope}_dCE'] for r in recs])
    li_ = np.array([r[f'lin_{scope}_dCE'] for r in recs])
    re_ = np.array([r[f'restr_{scope}_dCE'] for r in recs])
    tz = np.array([r[f'true_{scope}_z'] for r in recs])
    clear = np.abs(tz) >= 2
    def sign_agree(p, mask=None):
        s = (np.sign(p) == np.sign(tr))
        if mask is not None: s = s[mask]
        return round(float(s.mean()), 3) if len(s) else None
    return {
        'n': len(recs), 'n_signclear_true': int(clear.sum()),
        'linear': {'spearman': round(spearman(li_, tr), 3), 'pearson': round(pearson(li_, tr), 3),
                   'sign_agreement_all': sign_agree(li_), 'sign_agreement_signclear': sign_agree(li_, clear),
                   'mean_abs_err': round(float(np.abs(li_-tr).mean()), 5)},
        'restricted': {'spearman': round(spearman(re_, tr), 3), 'pearson': round(pearson(re_, tr), 3),
                       'sign_agreement_all': sign_agree(re_), 'sign_agreement_signclear': sign_agree(re_, clear),
                       'mean_abs_err': round(float(np.abs(re_-tr).mean()), 5)},
    }

metrics = {
    'global_all24': metric_block(records, 'global'),
    'trigger_all24': metric_block(records, 'trigger'),
    # honest subset: candidates with >=1 restricted downstream MLP (proxy nontrivial)
    'global_nontrivial': metric_block([r for r in records if r['n_restricted_downstream_mlps'] > 0], 'global'),
    'trigger_nontrivial': metric_block([r for r in records if r['n_restricted_downstream_mlps'] > 0], 'trigger'),
    # certification: subsample truth vs full-slice census truth (sampling-error bound)
    'subsample_vs_census_truth': {
        'spearman': round(spearman([r['true_global_dCE'] for r in records],
                                   [r['census_global_dCE'] for r in records]), 3),
        'pearson': round(pearson([r['true_global_dCE'] for r in records],
                                 [r['census_global_dCE'] for r in records]), 3),
        'mean_abs_diff': round(float(np.mean([abs(r['true_global_dCE']-r['census_global_dCE'])
                                              for r in records])), 6),
        'max_abs_diff': round(float(np.max([abs(r['true_global_dCE']-r['census_global_dCE'])
                                            for r in records])), 6)},
}

# restriction cost per key (context: how lossy is the restricted downstream view itself)
rb = {}
for k in KEYS:
    mn, se = stats(*rbase_cost[k])
    rb[f'restrict_from_L{k}'] = {'base_dCE_vs_full': round(mn, 5), 'SE': round(se, 5),
                                 'n_restricted_mlps': NL - k}

# analytic multiply-accumulate accounting (per position)
mlp_full_mac = 3 * D * DFF
mlp_restr_mac = D*K_IN + 2*K_IN*DFF + DFF*K_OUT + K_OUT*D
attn_mac = 6*D*D + 4*SEQL*D                                  # qkv/proj + score/mix (approx, avg)
readout_mac = D*V
def fwd_mac(nrestr):
    return (NL - nrestr)*(attn_mac + mlp_full_mac) + nrestr*(attn_mac + mlp_restr_mac) + readout_mac
cost = {
    'mlp_macs_per_position_full': mlp_full_mac, 'mlp_macs_per_position_restricted': mlp_restr_mac,
    'mlp_mac_ratio': round(mlp_full_mac / mlp_restr_mac, 2),
    'whole_forward_mac_ratio_if_all18_restricted': round(fwd_mac(0)/fwd_mac(NL), 2),
    'note': 'attention exact by design; readout (lm_head) unrestricted; MAC savings apply '
            'only to the restricted downstream MLPs, so realized per-candidate savings depend '
            'on candidate layer. Wall times below include Python/kernel overhead at batch 6.',
    'mean_sec_per_candidate_full_ablation_subsample': round(float(np.mean(list(t_full.values()))), 3),
    'mean_sec_per_candidate_restricted_ablation_subsample': round(float(np.mean(
        [t_restr[c] for c, k_, l_, _ in CANDS if rkey(k_, l_) < NL])), 3),
    'mean_sec_per_candidate_linear_proxy_subsample': round(float(np.mean(list(t_lin.values()))), 3),
    'restricted_base_total_sec_per_key': {f'L{k}': round(t_rbase[k], 3) for k in KEYS},
    'per_candidate_whole_forward_mac_ratio': {
        c: round(fwd_mac(max(0, NL - rkey(k_, l_)))/fwd_mac(0), 3) for c, k_, l_, _ in CANDS},
}

out = {
    'meta': {
        'model': 'bilin18', 'held_slice': 'FW[448:600,:128]', 'subsample_rows': SUBIDX.tolist(),
        'n_candidates': len(CANDS), 'K_in': K_IN, 'K_out': K_OUT, 'KCAUSAL': KCAUSAL, 'BATCH': BATCH,
        'ground_truth': 'full-model per-position mean-ablation delta cross-entropy (census conventions '
                        'verbatim), paired standard errors over positions',
        'linear_proxy': 'direct-to-logits: exact per-position residual perturbation of the ablation, '
                        'added straight to the final residual (downstream layers = identity), true '
                        'readout (rms_norm + lm_head + 30*tanh). Strongest fair version of the '
                        'top_boost/class-summed direct-to-logits construction.',
        'restricted_proxy': 'mean-ablate at the path layer, propagate through remaining layers with '
                            'each downstream MLP replaced by its folded restricted core (input top-288 '
                            'x output top-144 train-gram bases, per-position held means; map math '
                            'verbatim qk_hub_maprestrict), attention exact; paired against a same-'
                            'restriction no-ablation base.',
        'trivial_note': 'mlp.L17.* candidates have zero downstream MLPs, so the restricted proxy is '
                        'exactly the true effect for them by construction; the honest test is the '
                        'nontrivial subset.',
    },
    'positive_controls': {'fold_vs_unfolded_maxabsdiff': {f'L{k}': v for k, v in fold_check.items()},
                          'harness_recompute_vs_census': harness},
    'metrics': metrics,
    'restriction_base_cost': rb,
    'focus_sec67_cases': focus,
    'cost': cost,
    'records': records,
}
json.dump(out, open(f'{QK}/qk_certified_proxy.json', 'w'), indent=2)

print("\n===== CERTIFIED PROXY SUMMARY =====", flush=True)
for scope in ('global_all24', 'global_nontrivial', 'trigger_all24', 'trigger_nontrivial'):
    mb = metrics[scope]
    print(f"{scope}: linear spearman {mb['linear']['spearman']} sign {mb['linear']['sign_agreement_signclear']}"
          f" | restricted spearman {mb['restricted']['spearman']} sign {mb['restricted']['sign_agreement_signclear']}", flush=True)
print(f"subsample vs census truth: spearman {metrics['subsample_vs_census_truth']['spearman']}"
      f" maxabsdiff {metrics['subsample_vs_census_truth']['max_abs_diff']}", flush=True)
print("\nper-candidate (global): comp | true | linear | restricted", flush=True)
for r in sorted(records, key=lambda r: -r['true_global_dCE']):
    print(f"  {r['comp']:12s} true {r['true_global_dCE']:+.5f}±{r['true_global_SE']:.5f}"
          f"  lin {r['lin_global_dCE']:+.5f}  restr {r['restr_global_dCE']:+.5f}"
          f"  (downstream mlps restricted: {r['n_restricted_downstream_mlps']})", flush=True)
print("\nSaved qk_certified_proxy.json", flush=True)
print("QK CERTIFIED PROXY DONE", flush=True)
