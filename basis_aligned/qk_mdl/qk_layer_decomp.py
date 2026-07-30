"""Four-ledger decomposition battery for a SINGLE layer L of bilin18 (parametrized; run per layer).

Usage:  python qk_layer_decomp.py <L>          (L in 0..17; intended sweep 1..17)
        python qk_layer_decomp.py <L> smoke    (tiny batches, sanity only -- numbers not reportable)

Ledgers (machinery COPIED from the working scripts, not rewritten):
  (a) REPRESENTATION -- MLP composed-fold gauge identity at layer L (rms-as-gauge, expect ~1e-7),
      copied from qk_mlp1_composed_fold.py's gate g1 generalized to layer L via the shared T_ev fold.
  (b) SUBSTITUTABILITY -- MARGINAL causal dCE of replacing ONLY layer L's attention output
      (causally, through its PCA-64/head bottleneck) and layer L's MLP (exact composed fold T_ev on
      the analytically reconstructed stream with the bottlenecked a_L; streams j<L exact), everything
      else exact. Measured on held-back FW[448:600], base-relative paired per-token dCE with per-token
      AND row-clustered (per-sequence) standard errors, vs a head-span-restricted random-basis null
      (2 seeds) and a layer-L mean-input MLP floor, against the uniform ceiling ln V. CO recurrence,
      T_ev, make_bases, collect, per_tok copied from qk_wholemodel_substitutable.py.
      NOTE one deliberate difference from the whole-model chain: there the residual keeps the FULL
      attention output and the bottleneck applies only where MLP folds read a_j; here layer L's
      attention output is bottlenecked ON THE RESIDUAL too (a true causal module swap), per spec.
      Positive control: the CO-reconstructed stream must match the actual residual (~1e-6).
  (c) FUNCTION -- selection-predicate census for layer L's 9 heads (predicate library, lstsq fit on
      cooc, held-out predicate gain vs template-only), copied from qk_selection_census.py restricted
      to layer L (no coded-pattern substitution gate here -- that is a whole-model measurement).
  (d) JSON summary keyed by layer -> qk_layer_decomp_L{L}.json.
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
L = int(sys.argv[1])
SMOKE = len(sys.argv) > 2 and sys.argv[2] == 'smoke'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
assert 0 <= L < NL, f"layer index {L} out of range 0..{NL-1}"
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
HELD = FW[448:456][:, :129] if SMOKE else FW[448:600]
EB = 2 if SMOKE else 4          # eval batch (memory-safe: <=4)
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
UNIFORM = float(np.log(V))
BLKS = [m.transformer.h[i] for i in range(NL)]
WT = [(b.mlp.Left.weight.detach().float(), b.mlp.Right.weight.detach().float(),
       b.mlp.Down.weight.detach().float(), b.mlp.Down_bias.detach().float()) for b in BLKS]
def T_ev(li, u, v):
    Lw, Rw, Dw, _ = WT[li]
    return 0.5*(((u @ Lw.T) * (v @ Rw.T)) @ Dw.T + ((v @ Lw.T) * (u @ Rw.T)) @ Dw.T)
lam = [(b.lambdas[0].item(), b.lambdas[1].item()) for b in BLKS]
# coefficient recurrence for x_pre_l over streams {e, a_0..a_l, m_0..m_{l-1}}  [copied verbatim]
CO = []; cur = {'e': lam[0][0]+lam[0][1]}
for l in range(NL):
    xp = dict(cur); xp[('a', l)] = 1.0; CO.append(xp)
    nx = dict(xp); nx[('m', l)] = 1.0
    if l < NL-1:
        cur = {k: lam[l+1][0]*v for k, v in nx.items()}; cur['e'] = cur.get('e', 0.0)+lam[l+1][1]

# ===================== (a) REPRESENTATION: MLP composed-fold gauge at layer L =====================
@torch.no_grad()
def streams_at_L(idx):
    """real forward to layer L; returns (x_pre_L flattened, true MLP-L output flattened)."""
    B, T = idx.shape; x0 = F.rms_norm(m.transformer.wte(idx), (D,)); x = None; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(L+1):
        b = BLKS[li]; a = b.attn
        x = (b.lambdas[0]+b.lambdas[1])*x0 if li == 0 else b.lambdas[0]*x + b.lambdas[1]*x0
        hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        pat = ((torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)).masked_fill(~mask, 0.0)
        yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh.reshape(B, T, -1))
        mo = b.mlp(F.rms_norm(x, (D,)))
        if li == L: return x.reshape(-1, D), mo.reshape(-1, D)
        x = x + mo

XPg, Yg = streams_at_L(COOC[:4 if SMOKE else 8].to(DEV)[:, :128])
rho2 = XPg.pow(2).sum(1)/D
gauge = float(((T_ev(L, XPg, XPg)/rho2.unsqueeze(1) + WT[L][3]) - Yg).norm() / Yg.norm())
del XPg, Yg, rho2
print(f"[L{L}] (a) GATE rms-gauge MLP{L} composed fold: relerr {gauge:.2e}", flush=True)

# ===================== bases for (b): PCA-64/head of layer-L head outputs =====================
# [copied from qk_wholemodel_substitutable collect/make_bases, restricted to layer L]
acc = torch.zeros(NH, HD, HD, device=DEV, dtype=torch.float64)
hinsum = torch.zeros(D, device=DEV, dtype=torch.float64); hn = [0]
@torch.no_grad()
def collect(idx):
    B, T = idx.shape; x0 = F.rms_norm(m.transformer.wte(idx), (D,)); x = None; v1 = None
    global acc
    cos, sin = rope_tables(T, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(L+1):
        b = BLKS[li]; a = b.attn
        x = (b.lambdas[0]+b.lambdas[1])*x0 if li == 0 else b.lambdas[0]*x + b.lambdas[1]*x0
        hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        pat = ((torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)).masked_fill(~mask, 0.0)
        yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh.reshape(B, T, -1))
        if li == L:
            acc += torch.einsum('nhd,nhe->hde', yh.reshape(-1, NH, HD).double(), yh.reshape(-1, NH, HD).double())
            hh_ = F.rms_norm(x, (D,)); hinsum.add_(hh_.reshape(-1, D).double().sum(0)); hn[0] += B*T
            return
        x = x + b.mlp(F.rms_norm(x, (D,)))
for i in range(0, 16 if SMOKE else 64, 8):
    collect(COOC[i:i+8].to(DEV)[:, :128])
def make_basis(kind, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    cw = BLKS[L].attn.c_proj.weight.detach().float()
    cs = []
    for hh in range(NH):
        if kind == 'pca':
            ev, evec = torch.linalg.eigh(acc[hh])
            cs.append(cw[:, hh*HD:(hh+1)*HD] @ evec[:, ev.argsort(descending=True)[:64]].float())
        else:  # headspan random
            Rh, _ = torch.linalg.qr(torch.randn(HD, 64, generator=g, device=DEV))
            cs.append(cw[:, hh*HD:(hh+1)*HD] @ Rh)
    Qx, _ = torch.linalg.qr(torch.cat(cs, 1))
    return Qx
QB = {'pca': make_basis('pca'), 'null1': make_basis('null', 11), 'null2': make_basis('null', 12)}
MU = F.rms_norm((hinsum/hn[0]).float(), (D,))
print(f"[L{L}] bases ready", flush=True)

# ===================== (b) SUBSTITUTABILITY: layer-L-only causal surrogate =====================
RECON_CHECK = [None]   # positive control: CO-reconstructed stream vs actual residual at layer L
@torch.no_grad()
def forward(idx, mode, Q=None):
    """mode None real | 'sub' layer-L attn bottleneck + composed-fold MLP | 'floor' layer-L mean-input MLP."""
    B, T2 = idx.shape; x0 = F.rms_norm(m.transformer.wte(idx), (D,)); x = None; v1 = None
    cos, sin = rope_tables(T2, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T2, T2, device=DEV, dtype=torch.bool))
    a_list = []; mh = []
    for li in range(NL):
        b = BLKS[li]; a = b.attn
        x = (b.lambdas[0]+b.lambdas[1])*x0 if li == 0 else b.lambdas[0]*x + b.lambdas[1]*x0
        hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T2, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T2, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        pat = ((torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)).masked_fill(~mask, 0.0)
        aout = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T2, -1))
        if mode == 'sub' and li == L:
            af = aout.reshape(-1, D)
            afb = (af @ Q) @ Q.T                       # PCA/head (or null) bottleneck, causal
            x = x + afb.view(B, T2, D).to(x.dtype)
            co = CO[L]; xp = co['e']*x0.reshape(-1, D)
            for j in range(L+1):
                if ('a', j) in co: xp = xp + co[('a', j)]*(afb if j == L else a_list[j])
                if ('m', j) in co: xp = xp + co[('m', j)]*mh[j]
            if RECON_CHECK[0] is None:
                xr = x.reshape(-1, D)
                RECON_CHECK[0] = float((xp - xr).norm() / xr.norm())
            r = xp.pow(2).sum(1)/D
            mo = (T_ev(L, xp, xp)/r.unsqueeze(1) + WT[L][3])
            x = x + mo.view(B, T2, D).to(x.dtype)
            a_list = []; mh = []
            continue
        x = x + aout
        if mode == 'floor' and li == L:
            x = x + b.mlp(MU.expand(B, T2, D).to(x.dtype)); continue
        if mode == 'sub' and li < L:
            a_list.append(aout.reshape(-1, D))
            mo = b.mlp(F.rms_norm(x, (D,)))
            mh.append(mo.reshape(-1, D)); x = x + mo
        else:
            x = x + b.mlp(F.rms_norm(x, (D,)))
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float()

@torch.no_grad()
def per_tok(mode, Q=None):
    ces = []
    for i in range(0, len(HELD), EB):
        bb = HELD[i:i+EB].to(DEV)
        lg = forward(bb[:, :-1], mode, Q)
        ce = F.cross_entropy(lg.reshape(-1, V), bb[:, 1:].reshape(-1), reduction='none')
        ces.append(ce.cpu())
    return torch.cat(ces)

ce_real = per_tok(None)
base = float(ce_real.mean())
TPS = HELD.shape[1] - 1   # tokens per sequence
def rep(ce, ref):
    d = ce - ref
    seq_means = d.view(len(HELD), TPS).mean(1)
    return {'dCE': round(float(d.mean()), 5),
            'SE_pertok': round(float(d.std()/np.sqrt(d.numel())), 6),
            'SE_rowclust': round(float(seq_means.std()/np.sqrt(len(HELD))), 6)}
sub = {'base_CE': round(base, 5), 'uniform_ceiling_lnV': round(UNIFORM, 4), 'n_tokens': int(ce_real.numel())}
sub['sub_pca'] = rep(per_tok('sub', QB['pca']), ce_real)
sub['stream_recon_relerr'] = RECON_CHECK[0]          # positive control (~1e-6)
sub['null_headspan_s1'] = rep(per_tok('sub', QB['null1']), ce_real)
sub['null_headspan_s2'] = rep(per_tok('sub', QB['null2']), ce_real)
sub['mlp_floor_marginal'] = rep(per_tok('floor'), ce_real)
h = sub['sub_pca']['dCE']
sub['frac_of_uniform_ceiling'] = round(1 - h/(UNIFORM - base), 5)
fl = sub['mlp_floor_marginal']['dCE']
sub['frac_of_layer_floor_captured'] = round(1 - h/fl, 4) if fl > 1e-4 else None
print(f"[L{L}] (b) base CE {base:.4f} | sub(pca) dCE +{h:.5f} (SEtok {sub['sub_pca']['SE_pertok']}, "
      f"SErow {sub['sub_pca']['SE_rowclust']}) | recon ctrl {RECON_CHECK[0]:.2e}", flush=True)
print(f"[L{L}]     null s1/s2: +{sub['null_headspan_s1']['dCE']}/+{sub['null_headspan_s2']['dCE']} | "
      f"MLP floor +{fl} | uniform-headroom kept {sub['frac_of_uniform_ceiling']:.3%}", flush=True)

# ===================== (c) FUNCTION: selection-predicate census, layer L's 9 heads =====================
# [predicate library + lstsq fit copied from qk_selection_census.py, restricted to layer L]
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('gpt2')
import string as _string
_P = set(_string.punctuation)
FUNC = {'the','of','and','to','a','in','is','that','it','for','was','as','with','on','be','at','by','this','are','from','or','an','but','not','which'}
KP = torch.zeros(V, dtype=torch.bool); KF = torch.zeros(V, dtype=torch.bool); KC = torch.zeros(V, dtype=torch.bool)
for i in range(50257):
    s = tok.convert_ids_to_tokens(i)
    if s is None: continue
    core = s.replace('Ġ', ''); lead = s.startswith('Ġ')
    if len(core) and all(c in _P for c in core): KP[i] = True
    if core.lower() in FUNC: KF[i] = True
    if lead and len(core) and core[0].isupper(): KC[i] = True
KP, KF, KC = KP.to(DEV), KF.to(DEV), KC.to(DEV)
T0 = 128
FEATN = ['MATCH_prev', 'MATCH_same', 'KEY_punct', 'KEY_func', 'KEY_cap', 'FIRST', 'PREV1', 'PREV2']
NF = len(FEATN)

def feats(idx):
    """(B,NF,T,T) predicate features + causal mask."""
    B, T = idx.shape
    causal = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    prevtok = torch.roll(idx, 1, dims=1); prevtok[:, 0] = -1
    Fs = torch.zeros(B, NF, T, T, device=DEV)
    Fs[:, 0] = (prevtok.unsqueeze(1) == idx.unsqueeze(2)).float()        # tok_{j-1}==tok_i
    Fs[:, 1] = (idx.unsqueeze(1) == idx.unsqueeze(2)).float()            # tok_j==tok_i
    Fs[:, 2] = KP[idx].float().unsqueeze(1).expand(B, T, T)
    Fs[:, 3] = KF[idx].float().unsqueeze(1).expand(B, T, T)
    Fs[:, 4] = KC[idx].float().unsqueeze(1).expand(B, T, T)
    Fs[:, 5, :, 0] = 1.0
    eye1 = torch.diag(torch.ones(T-1, device=DEV), -1); Fs[:, 6] = eye1
    eye2 = torch.diag(torch.ones(T-2, device=DEV), -2); Fs[:, 7] = eye2
    return Fs * causal, causal

@torch.no_grad()
def patterns_L(idx):
    """(B,NH,T,T) layer-L pattern from the real forward (stops at layer L)."""
    B, T = idx.shape; x0 = F.rms_norm(m.transformer.wte(idx), (D,)); x = None; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(L+1):
        b = BLKS[li]; a = b.attn
        x = (b.lambdas[0]+b.lambdas[1])*x0 if li == 0 else b.lambdas[0]*x + b.lambdas[1]*x0
        hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        pat = ((torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)).masked_fill(~mask, 0.0)
        if li == L: return pat
        yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh.reshape(B, T, -1)); x = x + b.mlp(F.rms_norm(x, (D,)))

K = NF + 2   # features = [NF predicates, TEMPLATE, 1]
AtA = torch.zeros(NH, K, K, device=DEV, dtype=torch.float64)
Aty = torch.zeros(NH, K, device=DEV, dtype=torch.float64)
# first pass: template (mean pattern per head at layer L)
TPL = torch.zeros(NH, T0, T0, device=DEV); nb = 0
for i in range(0, 8 if SMOKE else 48, 8):
    TPL += patterns_L(COOC[i:i+8].to(DEV)[:, :T0]).mean(0); nb += 1
TPL /= nb
print(f"[L{L}] (c) templates ready", flush=True)
# second pass: normal equations
for i in range(48, 64 if SMOKE else 168, 8):
    idx = COOC[i:i+8].to(DEV)[:, :T0]
    Fs, causal = feats(idx); B = idx.shape[0]
    pats = patterns_L(idx)
    mask_flat = causal.expand(B, T0, T0).reshape(-1)
    Xbase = torch.cat([Fs.reshape(B, NF, -1), torch.zeros(B, 2, T0*T0, device=DEV)], 1)
    tplf = (TPL.unsqueeze(0).expand(B, NH, T0, T0) * causal).reshape(B, NH, -1)
    for h in range(NH):
        X = Xbase.clone(); X[:, NF] = tplf[:, h]; X[:, NF+1] = causal.expand(B, T0, T0).reshape(B, -1).float()
        Xf = X.permute(0, 2, 1).reshape(-1, K)[mask_flat].double()
        yf = pats[:, h].reshape(-1)[mask_flat].double()
        AtA[h] += Xf.T @ Xf; Aty[h] += Xf.T @ yf
print(f"[L{L}] (c) normal equations done", flush=True)
W = torch.linalg.solve(AtA + 1e-6*torch.eye(K, device=DEV, dtype=torch.float64), Aty.unsqueeze(-1)).squeeze(-1).float()

# held-out R^2: full model vs template-only
@torch.no_grad()
def heldout_r2():
    ss_res = torch.zeros(NH, device=DEV); ss_tpl = torch.zeros(NH, device=DEV); ss_tot = torch.zeros(NH, device=DEV)
    for i in range(200, 208 if SMOKE else 240, 8):
        idx = COOC[i:i+8].to(DEV)[:, :T0]
        Fs, causal = feats(idx); B = idx.shape[0]
        pats = patterns_L(idx)
        for h in range(NH):
            tpl = (TPL[h].unsqueeze(0) * causal)
            y = pats[:, h]
            pred = (Fs * W[h, :NF].view(1, NF, 1, 1)).sum(1) + W[h, NF]*tpl + W[h, NF+1]*causal
            pred_t = W[h, NF]*tpl + W[h, NF+1]*causal
            mu = y[:, causal].mean()
            ss_res[h] += ((pred - y)[:, causal]**2).sum(); ss_tpl[h] += ((pred_t - y)[:, causal]**2).sum()
            ss_tot[h] += ((y[:, causal] - mu)**2).sum()
    return 1 - ss_res/ss_tot, 1 - ss_tpl/ss_tot
R2full, R2tpl = heldout_r2()
gain = R2full - R2tpl
census = []
for h in range(NH):
    coef = W[h, :NF]
    top = int(coef.abs().argmax())
    g = float(gain[h])
    census.append({'head': h, 'r2_full': round(float(R2full[h]), 3), 'r2_template': round(float(R2tpl[h]), 3),
                   'predicate_gain': round(g, 3), 'top_predicate': FEATN[top],
                   'top_coef': round(float(coef[top]), 4), 'programmatic': bool(g >= 0.05)})
for c in census:
    tag = 'PROGRAMMATIC' if c['programmatic'] else ('positional' if c['r2_template'] >= 0.2 else 'unexplained')
    print(f"[L{L}]   H{c['head']}: r2_full {c['r2_full']} r2_tpl {c['r2_template']} gain {c['predicate_gain']} "
          f"top={c['top_predicate']} ({c['top_coef']}) [{tag}]", flush=True)

# ===================== (d) summary =====================
res = {'layer': L, 'smoke': SMOKE,
       'representation': {'mlp_gauge_relerr': gauge},
       'substitutability': sub,
       'function': {'census': census,
                    'n_programmatic': sum(c['programmatic'] for c in census)}}
json.dump(res, open(f'{QK}/qk_layer_decomp_L{L}.json', 'w'), indent=2)
print(json.dumps({'layer': L, 'gauge': f"{gauge:.1e}", 'sub_dCE': sub['sub_pca']['dCE'],
                  'null_dCE': sub['null_headspan_s1']['dCE'], 'floor_dCE': fl,
                  'n_programmatic': res['function']['n_programmatic']}), flush=True)
print(f"QK LAYER DECOMP L{L} DONE", flush=True)
