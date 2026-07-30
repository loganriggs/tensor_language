"""HYGIENE FIX for the per-layer substitutability table (bilin18, layers 2-17).

qk_l217_symbolgen.py measured, per layer, the causal cross-entropy cost of
replacing each layer's attention pattern with a growing symbol-basis fold, versus
a per-token TABLE baseline and a RANDOM-basis null. But the honest denominator --
the POSITIONAL-MEAN pattern floor (replace the pattern with its position-
conditioned mean, an in-distribution zero point) -- was only measured for layers
2-5 (RESULTS_l0_mdl.md §12q, tick 262), which showed zero-ablation inflates loads
10-60x and that at layer 5 the symbol fold LOSES to the positional mean.

This script copies qk_l217_symbolgen.py's exact machinery (growing symbol basis,
per-layer pattern substitution, eval on the held-back slice) VERBATIM, and adds
for EVERY layer 2..17:
  (a) a POSITIONAL-MEAN pattern arm -- replace the layer's pattern with its
      position-conditioned mean, EXACTLY the §12q / tick-262 construction copied
      verbatim from qk_mean_ablation.py lines 136-140:
          pat = pat.mean(0, keepdim=True).expand_as(pat).contiguous()
      (mean over the batch / sequence dimension at each (head, query, key)
      offset; full positional structure kept, all content dependence removed);
  (b) paired PER-TOKEN standard errors on the base-relative cross-entropy for the
      symbol, table, random, and mean arms, plus the paired sym-vs-mean-floor
      difference (the real content-function test), measured on FINEWEB[448:600].

Per layer it prints [sym / table / random / positional-mean] delta-CE each with
its standard error, and flags which layers the symbol fold BEATS the positional-
mean floor (a genuine content-function win) versus merely matches it (positional
structure). A per-layer JSON is saved.

NOTE ON THE POSITIONAL MEAN (assumption, flagged): §12q's arm is a PER-MINIBATCH
mean (mean over the 4 sequences in each eval batch), copied verbatim here to match
the tick-262 construction exactly. It is not a global mean over the whole slice; a
global mean would be a (slightly tighter, less noisy) floor, but the LESSON is to
copy the working code verbatim rather than paraphrase, so the batch-mean is kept.
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
wte = m.transformer.wte.weight.detach().float().to(DEV)
BASE = 3.07630
LAYERS = list(range(2, 18))
MAXL = max(LAYERS)
EVAL = (448, 600)  # held-back FineWeb slice for the substitutability audit

# layer-0 named archetype value-directions (PA), embedding PCs (EW)
mh = torch.load(f'{QK}/qk_minimal_heads.pt', map_location=DEV)
pol0 = torch.load(f'{QK}/qk_h0_polish_g025.pt', map_location=DEV); pol4 = torch.load(f'{QK}/qk_h04_polish.pt', map_location=DEV)
PA = {}
for hh in range(NH):
    if hh in (0, 4):
        bb = pol0 if hh == 0 else pol4; Dv = bb[f'h{hh}_v_Dm'].to(DEV); Dv = Dv / Dv.norm(dim=1, keepdim=True).clamp_min(1e-8)
        Vdir = Dv.T @ bb[f'h{hh}_CJ'][:, :16].to(DEV)
    else:
        Pp = mh[f'h{hh}']; Dn_ = Pp['Dm'].to(DEV); Dn_ = Dn_ / Dn_.norm(dim=1, keepdim=True).clamp_min(1e-8)
        Vdir = Dn_[:, 2*HD:].T @ Pp['U'].to(DEV)[:, :16]
    if Vdir.shape[1] < 16: Vdir = torch.cat([Vdir, torch.zeros(HD, 16-Vdir.shape[1], device=DEV)], 1)
    PA[hh] = (Vdir / Vdir.norm(dim=0, keepdim=True).clamp_min(1e-9)).contiguous()
EPC96 = torch.linalg.svd(F.rms_norm(wte, (D,)) - F.rms_norm(wte, (D,)).mean(0), full_matrices=False).Vh[:96].T.contiguous()
EW = F.rms_norm(wte, (D,)) @ EPC96


@torch.no_grad()
def run(idx, upto):
    """Run blocks 0..upto-1; return dict yh[li] (B,T,NH,HD) and x_in[li] (residual entering block li's attn)."""
    dt = m.transformer.wte.weight.dtype
    x = m.transformer.wte(idx); x = F.rms_norm(x, (D,)); x0 = x; v1 = None
    B, T = idx.shape
    cos, sin = rope_tables(T, HD, idx.device, dt, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
    yh = {}; xin = {}
    for li in range(upto):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; xin[li] = x
        a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v); yh[li] = yh4
        x = x + a.c_proj(yh4.reshape(B, T, -1)); x = x + blk.mlp(F.rms_norm(x, (D,)))
    return yh, xin

# ---- pass 1: per-layer PCA bases (layers 1..MAXL-1) + layer means ----
cov = {l: torch.zeros(NH, HD, HD, device=DEV, dtype=torch.float64) for l in range(MAXL)}
mu = {l: torch.zeros(NH, HD, device=DEV, dtype=torch.float64) for l in range(MAXL)}
nt = 0
for i in range(0, 256, 4):
    yh, _ = run(COOC[i:i+4].to(DEV)[:, :-1], MAXL)
    for l in range(MAXL):
        f = yh[l].reshape(-1, NH, HD).double(); mu[l] += f.sum(0); cov[l] += torch.einsum('nhd,nhe->hde', f, f)
    nt += yh[0].reshape(-1, NH, HD).shape[0]
for l in range(MAXL):
    mu[l] /= nt; cov[l] = cov[l]/nt - torch.einsum('hd,he->hde', mu[l], mu[l])
PB = {}  # per-head 16-d PCA basis per layer 1..MAXL-1
for l in range(1, MAXL):
    PB[l] = {}
    for h in range(NH):
        ev, evec = torch.linalg.eigh(cov[l][h]); PB[l][h] = evec[:, ev.argsort(descending=True)[:16]].float().contiguous()
# random-null bases (same shapes)
g = torch.Generator(device=DEV).manual_seed(1)
def rand_basis():
    Qr, _ = torch.linalg.qr(torch.randn(HD, HD, generator=g, device=DEV)); return Qr[:, :16].contiguous()
PAr = {h: rand_basis() for h in range(NH)}
PBr = {l: {h: rand_basis() for h in range(NH)} for l in range(1, MAXL)}
print('bases ready', flush=True)


def codes(L, idx, yh, named=True):
    B, T = idx.shape; ce = EW[idx.reshape(-1)]
    pa = PA if named else PAr; pb = PB if named else PBr
    c = [ce, torch.cat([(yh[0][..., h, :].reshape(-1, HD) - mu[0][h].float()) @ pa[h] for h in range(NH)], 1)]
    for l in range(1, L):
        c.append(torch.cat([(yh[l][..., h, :].reshape(-1, HD) - mu[l][h].float()) @ pb[l][h] for h in range(NH)], 1))
    return torch.cat(c, 1)

# ---- pass 2: fit ridge generators (sym + random) per layer; token tables ----
Wsym, Wrnd = {}, {}; R2 = {}
tabsum = {L: torch.zeros(V, D, device=DEV, dtype=torch.float64) for L in LAYERS}; tabcnt = torch.zeros(V, device=DEV, dtype=torch.float64)
for named, store in [(True, Wsym), (False, Wrnd)]:
    AtA = {L: torch.zeros(96+144*L+1, 96+144*L+1, device=DEV, dtype=torch.float64) for L in LAYERS}
    AtY = {L: torch.zeros(96+144*L+1, D, device=DEV, dtype=torch.float64) for L in LAYERS}
    for i in range(0, 512, 4):
        b = COOC[i:i+4].to(DEV)[:, :-1]; yh, xin = run(b, MAXL+1)
        for L in LAYERS:
            Cd = codes(L, b, yh, named).double(); Cd = torch.cat([Cd, torch.ones(Cd.shape[0], 1, device=DEV, dtype=torch.float64)], 1)
            Y = xin[L].reshape(-1, D).double(); AtA[L] += Cd.T @ Cd; AtY[L] += Cd.T @ Y
            if named:
                tok = b.reshape(-1); tabsum[L].index_add_(0, tok, Y)
        if named:
            tabcnt.index_add_(0, b.reshape(-1), torch.ones(b.numel(), device=DEV, dtype=torch.float64))
    for L in LAYERS:
        store[L] = torch.linalg.solve(AtA[L] + 10.0*torch.eye(96+144*L+1, device=DEV, dtype=torch.float64), AtY[L]).float()
TAB = {L: (tabsum[L] / tabcnt.clamp_min(1).unsqueeze(1)).float() for L in LAYERS}
# held-out R2 (sym) per layer
b = COOC[512:516].to(DEV)[:, :-1]; yh, xin = run(b, MAXL+1)
for L in LAYERS:
    Cd = codes(L, b, yh, True); Cd = torch.cat([Cd, torch.ones(Cd.shape[0], 1, device=DEV)], 1)
    Y = xin[L].reshape(-1, D); R2[L] = 1 - float((Cd @ Wsym[L] - Y).pow(2).sum() / (Y - Y.mean(0)).pow(2).sum())
print('generators fit; R2', {L: round(R2[L], 3) for L in LAYERS}, flush=True)
# offload all per-layer tables/generators to CPU; stage only the active layer to GPU in audit
del tabsum, AtA, AtY; torch.cuda.empty_cache()
for L in LAYERS:
    TAB[L] = TAB[L].cpu(); Wsym[L] = Wsym[L].cpu(); Wrnd[L] = Wrnd[L].cpu()
GP = {}  # active-layer GPU staging


@torch.no_grad()
def audit_pertoken(Ltgt, mode):
    """Return per-token cross-entropy (1-D CPU tensor) over FINEWEB[EVAL] for one arm.

    mode:
      'base' -> unmodified forward (denominator for the paired deltas);
      'sym'/'rand'/'tab' -> replace layer-Ltgt QK input with the generated x_L
          (VERBATIM from qk_l217_symbolgen.py audit);
      'mean' -> replace layer-Ltgt attention PATTERN with its position-conditioned
          mean (§12q / tick-262, verbatim from qk_mean_ablation.py):
                pat = pat.mean(0, keepdim=True).expand_as(pat).contiguous()
    """
    out = []
    for i in range(EVAL[0], EVAL[1], 4):
        idx = FINEWEB[i:i+4].to(DEV)[:, :-1]; B, T = idx.shape
        dt = m.transformer.wte.weight.dtype; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        cos, sin = rope_tables(T, HD, DEV, dt, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool)); yh = {}
        for li in range(NL):
            blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0
            a = blk.attn; hcur = F.rms_norm(x, (D,))
            def qkf(lin, src): z = F.rms_norm(lin(src).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
            v = a.c_v(hcur).view(B, T, NH, HD)
            if v1 is None: v1 = v
            v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
            if li == Ltgt and mode in ('sym', 'rand', 'tab'):
                if mode == 'tab': xh = GP['tab'][idx.reshape(-1)].view(B, T, D)
                else:
                    Cd = codes(Ltgt, idx, yh, mode == 'sym'); Cd = torch.cat([Cd, torch.ones(Cd.shape[0], 1, device=DEV)], 1)
                    xh = (Cd @ GP['sym' if mode == 'sym' else 'rnd']).view(B, T, D)
                hs = F.rms_norm(xh, (D,)).to(hcur.dtype)
                q, k, q2, k2 = qkf(a.c_q, hs), qkf(a.c_k, hs), qkf(a.c_q2, hs), qkf(a.c_k2, hs)
            else:
                q, k, q2, k2 = qkf(a.c_q, hcur), qkf(a.c_k, hcur), qkf(a.c_q2, hcur), qkf(a.c_k2, hcur)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
            pat = (s1*s2).masked_fill(~mask, 0.0)
            if li == Ltgt and mode == 'mean':
                # VERBATIM §12q / tick-262 positional-mean arm (qk_mean_ablation.py L136-140)
                pat = pat.mean(0, keepdim=True).expand_as(pat).contiguous()
            yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v); yh[li] = yh4
            x = x + a.c_proj(yh4.reshape(B, T, -1)); x = x + blk.mlp(F.rms_norm(x, (D,)))
        lg = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float()
        tgt = FINEWEB[i:i+4].to(DEV)[:, 1:].reshape(-1)
        ce = F.cross_entropy(lg.reshape(-1, V), tgt, reduction='none')
        out.append(ce.detach().cpu())
    return torch.cat(out)


def mean_se(x):
    x = x.double()
    n = x.numel()
    return float(x.mean()), float(x.std(unbiased=True) / (n ** 0.5)), int(n)


# base arm is identical regardless of Ltgt -> compute once
base_ce = audit_pertoken(LAYERS[0], 'base')
print(f'base per-token CE {base_ce.mean():.5f} over {base_ce.numel()} tokens '
      f'(scalar reference BASE={BASE}); eval FINEWEB[{EVAL[0]}:{EVAL[1]}]', flush=True)

res = {'_meta': {'eval_slice': list(EVAL), 'n_tokens': int(base_ce.numel()),
                 'base_ce_mean': round(float(base_ce.mean()), 5), 'BASE_const': BASE,
                 'positional_mean_note': 'per-minibatch (4-seq) mean, verbatim §12q/tick-262 qk_mean_ablation.py'}}
for L in LAYERS:
    GP['sym'] = Wsym[L].to(DEV); GP['rnd'] = Wrnd[L].to(DEV); GP['tab'] = TAB[L].to(DEV)
    arms = {}
    for mode in ('sym', 'tab', 'rand', 'mean'):
        ce = audit_pertoken(L, mode)
        d = ce - base_ce                      # paired per-token base-relative delta
        mn, se, n = mean_se(d)
        arms[mode] = {'dce': round(mn, 5), 'se': round(se, 5)}
    # paired sym-vs-positional-mean-floor test (the real content-function win)
    dsm = (audit_pertoken(L, 'sym') - audit_pertoken(L, 'mean'))
    smn, sse, _ = mean_se(dsm)
    GP.clear(); torch.cuda.empty_cache()
    # sym BEATS the floor if its CE is significantly LOWER than the positional mean's
    beats = smn < 0 and abs(smn) > 2 * sse
    verdict = 'SYM BEATS MEAN-FLOOR' if beats else ('sym~=floor' if abs(smn) <= 2 * sse else 'FLOOR BEATS SYM')
    r = {'R2': round(R2[L], 4),
         'sym': arms['sym'], 'table': arms['tab'], 'random': arms['rand'], 'positional_mean': arms['mean'],
         'sym_minus_mean': {'delta': round(smn, 5), 'se': round(sse, 5)}, 'verdict': verdict}
    res[f'layer{L}'] = r
    print(f"L{L:2d}: sym {arms['sym']['dce']:+.5f}±{arms['sym']['se']:.5f} | "
          f"table {arms['tab']['dce']:+.5f}±{arms['tab']['se']:.5f} | "
          f"random {arms['rand']['dce']:+.5f}±{arms['rand']['se']:.5f} | "
          f"pos-mean {arms['mean']['dce']:+.5f}±{arms['mean']['se']:.5f}  "
          f"|| sym-mean {smn:+.5f}±{sse:.5f}  {verdict}  (R2 {r['R2']})", flush=True)
    json.dump(res, open(f'{QK}/qk_symbolgen_meanfloor.json', 'w'), indent=2)
print('QK SYMBOLGEN MEANFLOOR DONE', flush=True)
