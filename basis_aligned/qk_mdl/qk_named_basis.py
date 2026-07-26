"""TICK 229 (the reframe, first fit): NAMED-BASIS interface map.

Features are LEDGER OBJECTS ONLY — no MLP-output codes, no weight references: the
96-dim embedding code of the current token, plus per-l0-head ARCHETYPE ACTIVATIONS
g_{h,r}(i) = <attn0_h(i), v-direction of archetype r> (top-16 archetypes per head,
value-mode directions from the validated inventories; 144 dims). attn0 comes from the
exact fold tables already charged in the global ledger. Arms: linear (240->576) and
gated bilinear with skip. Targets/audit as before. Beat: pairwise 29% @ 14 Mb;
stretch: weight-referencing 45%. Entity-carry check: report R2 restricted to the
worst-200 entity positions.

Original tick-211 header:
"""
"""TICK 211: code-composition sweep — where does the ungenerated half of the
interface signal live? Arms: linear on 128/512-wide MLP codes (fine-spectrum test);
mixed code (MLP-64 + per-l0-head attention-output PCA-8 (72 dims) + norm scalars,
140 dims) linear and swiglu (attention-source test). Anchors: lin64 +0.0363,
nonlinear-64 plateau +0.0334, oracle +0.0113, static +0.0515.

Derived from tick 210b; original header:
"""
"""TICK 210b (training-hygiene fix of the zoo: whitened codes,
linear-skip wrappers on all nonlinear arms, grad clip, lr 1e-3 with decay, 8000 steps.
Round 1 diverged on the gated arms and underfit mlp/hier — logged as training failure,
not function-class evidence.)

Original header:
"""
"""TICK 210 (Logan): the nonlinear interface generator — architecture stress test,
parameter-matched (~140-150k trunk parameters each unless noted).

Task: predict, from upstream-only inputs, the 16 adapter coordinates per layer-1
factor channel (36 channels x 16 = 576 targets), i.e. the context signal layer 1's
pattern consumes. Inputs: c = P64^T (mlp_out - mean) (64-dim code from the block-0 MLP
output), optionally plus the residual normalization scalars (the known divisive
nonlinearity). Arms:
  lin64      : linear 64 -> 576 (37k params; tick-205 reference, refit on more data)
  linscal    : linear on [c, s0, 1/s0, rms(mo), 1/rms(mo)] (68-dim; 39k) — tests the
               "the nonlinearity is mostly the divisive norms" hypothesis cheaply
  linwide    : linear on a 256-dim code (P256; 147k) — is it just code width?
  bilin      : W3((W1 c) * (W2 c)), gate width 200 (141k) — the model's own primitive
  swiglu     : W3((W1 c) * silu(W2 c)), width 200 (141k) — gated, non-quadratic
  mlp        : W2 gelu(W1 c), hidden 230 (147k) — generic single-encoder MLP
  hier       : shared 64->32 bottleneck, then per-map (q1/k1/q2/k2) subnet
               32 -> 200 -> 9x16 (145k) — hierarchical sharing
Training: Adam on 118k train positions (256 cooc docs, split 90/10), MSE on adapter
coordinates; val R^2 reported. Evaluation: FULL-AUDIT dCE (307k) with each arm's
generated corrections in the real forward. Also per-arm weight-sparsity (top-1% mass)
for Logan's "sparsest fit = correct prior" hypothesis — reported with the caveat that
weight sparsity is basis-dependent (Section 8's lesson).
Anchors: static tables +0.0515; oracle rank-16 +0.0113 (tick 204).
"""
import json
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
N_EST, TAU, N_CAP, R_AD = 1024, 8.0, 256, 16

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
blk0 = m.transformer.h[0]
a1 = m.transformer.h[1].attn
MAPS = (('q1', a1.c_q), ('k1', a1.c_k), ('q2', a1.c_q2), ('k2', a1.c_k2))


@torch.no_grad()
def block01(idx):
    dt = m.transformer.wte.weight.dtype
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    B, T = idx.shape
    x = blk0.lambdas[0] * x + blk0.lambdas[1] * x0
    a = blk0.attn
    hcur = F.rms_norm(x, (x.size(-1),))
    cos, sin = rope_tables(T, HD, idx.device, dt, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]

    def qk(lin):
        z = lin(hcur).view(B, T, NH, HD)
        return apply_rot(F.rms_norm(z, (HD,)), cosb, sinb)

    v = a.c_v(hcur).view(B, T, NH, HD)
    mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
    q, k = qk(a.c_q), qk(a.c_k)
    q2, k2 = qk(a.c_q2), qk(a.c_k2)
    s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
    s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
    pat = (s1 * s2).masked_fill(~mask, 0.0)
    yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
    y = yh.reshape(B, T, -1)
    x = x + a.c_proj(y)
    mo = blk0.mlp(F.rms_norm(x, (x.size(-1),)))
    x = x + mo
    blk1 = m.transformer.h[1]
    return blk1.lambdas[0] * x + blk1.lambdas[1] * x0, mo, yh


# ---- shrunk tables ----
print('tables...', flush=True)
sum_x = torch.zeros(V, D, device=DEV)
cnt = torch.zeros(V, device=DEV)
with torch.no_grad():
    for i in range(0, N_EST, 4):
        idx = COOC[i:i + 4].to(DEV)[:, :-1]
        x, _, _ = block01(idx)
        sum_x.index_add_(0, idx.reshape(-1), x.float().reshape(-1, D))
        cnt.index_add_(0, idx.reshape(-1), torch.ones(idx.numel(), device=DEV))
wte = m.transformer.wte.weight.detach().float().to(DEV)
mean_x = torch.where((cnt > 0)[:, None], sum_x / cnt[:, None].clamp_min(1), wte)
shr = (cnt / (cnt + TAU))[:, None] * mean_x + (TAU / (cnt + TAU))[:, None] * wte
TABLES = {}
with torch.no_grad():
    xn = F.rms_norm(shr, (D,))
    for name, lin in MAPS:
        TABLES[name] = F.rms_norm(lin(xn).view(V, NH, HD).float(), (HD,)).contiguous()
del sum_x, mean_x, shr, xn
torch.cuda.empty_cache()

# ---- capture ----
print('capturing...', flush=True)
MOs, S0s, YHs, DEVS = [], [], [], {n: [] for n, _ in MAPS}
with torch.no_grad():
    for i in range(0, N_CAP, 4):
        idx = COOC[i:i + 4].to(DEV)[:, :-1]
        xin1, mo, yh = block01(idx)
        h1n = F.rms_norm(xin1, (D,))
        ids = idx.reshape(-1)
        MOs.append(mo.float().reshape(-1, D).cpu())
        YHs.append(yh.float().reshape(-1, NH, HD).cpu())
        S0s.append(xin1.float().reshape(-1, D).pow(2).mean(1).sqrt().cpu())
        for name, lin in MAPS:
            fa = F.rms_norm(lin(h1n).view(*idx.shape, NH, HD).float(), (HD,))
            DEVS[name].append((fa.reshape(-1, NH, HD) - TABLES[name][ids]).cpu())
MO = torch.cat(MOs)
YH = torch.cat(YHs)
S0 = torch.cat(S0s)
N = MO.shape[0]
ntr = int(N * 0.9)
print(f'{N} positions ({ntr} train)', flush=True)
MO_MEAN = MO[:ntr].mean(0)
Xc = (MO[:ntr] - MO_MEAN).to(DEV)
_, _, VhM = torch.linalg.svd(Xc, full_matrices=False)
P64 = VhM[:64].T.contiguous()
P128 = VhM[:128].T.contiguous()
P512 = VhM[:512].T.contiguous()
P256 = VhM[:256].T.contiguous()
del Xc
UB, MUB = {}, {}
Ys = []
for name, _ in MAPS:
    Dv = torch.cat(DEVS[name])
    for h in range(NH):
        X = Dv[:, h].to(DEV)
        mu = X[:ntr].mean(0)
        _, _, Vh = torch.linalg.svd(X[:ntr] - mu, full_matrices=False)
        U = Vh[:R_AD].T.contiguous()
        UB[(name, h)] = U
        MUB[(name, h)] = mu
        Ys.append(((X - mu) @ U).cpu())
        del X
    del Dv
    torch.cuda.empty_cache()
DEVS = None
Y = torch.cat(Ys, 1).to(DEV)                                  # (N, 576)
C64 = ((MO - MO_MEAN) @ P64.cpu()).to(DEV)
C256 = ((MO - MO_MEAN) @ P256.cpu()).to(DEV)
C128 = ((MO - MO_MEAN) @ P128.cpu()).to(DEV)
C512 = ((MO - MO_MEAN) @ P512.cpu()).to(DEV)
STD64 = C64[:ntr].std(0).clamp_min(1e-6)
STD128 = C128[:ntr].std(0).clamp_min(1e-6)
STD512 = C512[:ntr].std(0).clamp_min(1e-6)
STD256 = C256[:ntr].std(0).clamp_min(1e-6)
C64 = C64 / STD64
C128 = C128 / STD128
C512 = C512 / STD512
C256 = C256 / STD256
mh_pt2 = torch.load(f'{QK}/qk_minimal_heads.pt', map_location=DEV)
pol0 = torch.load(f'{QK}/qk_h0_polish_g025.pt', map_location=DEV)
pol4 = torch.load(f'{QK}/qk_h04_polish.pt', map_location=DEV)
PA, YMU = {}, {}
attn_codes = []
for hh in range(NH):
    if hh in (0, 4):
        bb = pol0 if hh == 0 else pol4
        Dv = bb[f'h{hh}_v_Dm'].to(DEV)
        Dv = Dv / Dv.norm(dim=1, keepdim=True).clamp_min(1e-8)
        Vdir = Dv.T @ bb[f'h{hh}_CJ'][:, :16].to(DEV)
    else:
        P = mh_pt2[f'h{hh}']
        Dn_ = P['Dm'].to(DEV)
        Dn_ = Dn_ / Dn_.norm(dim=1, keepdim=True).clamp_min(1e-8)
        U_ = P['U'].to(DEV)
        Vdir = Dn_[:, 2 * HD:].T @ U_[:, :min(16, U_.shape[1])]
    if Vdir.shape[1] < 16:
        Vdir = torch.cat([Vdir, torch.zeros(HD, 16 - Vdir.shape[1], device=DEV)], 1)
    Vdir = Vdir / Vdir.norm(dim=0, keepdim=True).clamp_min(1e-9)
    PA[hh] = Vdir.contiguous()
    Xh = YH[:, hh].to(DEV)
    YMU[hh] = Xh[:ntr].mean(0)
    attn_codes.append(((Xh - YMU[hh]) @ PA[hh]).cpu())
    del Xh
CATT = torch.cat(attn_codes, 1).to(DEV)
STDA = CATT[:ntr].std(0).clamp_min(1e-6)
CATT = CATT / STDA

mrms = MO.pow(2).mean(1).sqrt()
SCAL = torch.stack([S0, 1 / S0.clamp_min(1e-6), mrms, 1 / mrms.clamp_min(1e-6)], 1).to(DEV)
SCAL_MU = SCAL[:ntr].mean(0)
SCAL_SD = SCAL[:ntr].std(0).clamp_min(1e-6)
SCAL = (SCAL - SCAL_MU) / SCAL_SD
EPC96 = torch.linalg.svd(F.rms_norm(wte, (D,)) - F.rms_norm(wte, (D,)).mean(0),
                         full_matrices=False).Vh[:96].T.contiguous()
EWcur = (F.rms_norm(wte, (D,)) @ EPC96)
EWcur = EWcur / EWcur.std(0).clamp_min(1e-6)
IDS_ALL2 = COOC[:N_CAP, :-1].reshape(-1)[:N]
CEMB = EWcur[IDS_ALL2.to(DEV)]
CMIX = torch.cat([CEMB, CATT], 1)
CS = torch.cat([C64, SCAL], 1)
print('features ready', flush=True)


class Bilin(nn.Module):
    def __init__(self, din, q, dout, gate=None):
        super().__init__()
        self.a = nn.Linear(din, q, bias=False)
        self.b = nn.Linear(din, q, bias=False)
        self.o = nn.Linear(q, dout, bias=True)
        self.gate = gate

    def forward(self, x):
        g = self.b(x)
        if self.gate == 'silu':
            g = F.silu(g)
        return self.o(self.a(x) * g)


class Skip(nn.Module):
    def __init__(self, din, inner, dout):
        super().__init__()
        self.lin = nn.Linear(din, dout)
        self.inner = inner
        for p in self.inner.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, std=0.02)

    def forward(self, x):
        return self.lin(x) + self.inner(x)


class Hier(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Linear(64, 32, bias=False)
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(32, 150), nn.GELU(),
                                                  nn.Linear(150, 144)) for _ in range(4)])

    def forward(self, x):
        t = self.trunk(x)
        return torch.cat([hd(t) for hd in self.heads], 1)


ARMS = {
    'named_lin': (nn.Linear(240, 576), CMIX),
    'named_swiglu': (Skip(240, Bilin(240, 140, 576, gate='silu'), 576), CMIX),
}
out = {}
models = {}
for aname, (mod, X) in ARMS.items():
    mod = mod.to(DEV)
    nparam = sum(p.numel() for p in mod.parameters())
    opt = torch.optim.Adam(mod.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(0)
    for step in range(8000):
        bi = torch.randint(0, ntr, (8192,), generator=g).to(DEV)
        loss = F.mse_loss(mod(X[bi]), Y[bi])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
        opt.step()
        if step in (5000, 7000):
            for pg in opt.param_groups:
                pg['lr'] *= 0.3
    with torch.no_grad():
        pred = mod(X[ntr:])
        r2 = 1 - float(((pred - Y[ntr:]) ** 2).sum()) / float(
            ((Y[ntr:] - Y[:ntr].mean(0)) ** 2).sum())
        w = torch.cat([p.abs().flatten() for p in mod.parameters()])
        top1 = float(w.sort(descending=True).values[:max(1, len(w) // 100)].sum() / w.sum())
    out[aname] = {'params': nparam, 'val_r2': round(r2, 4), 'weight_top1pct': round(top1, 3)}
    models[aname] = mod.eval()
    print(f'{aname}: {nparam} params, val R2 {r2:.4f}, top-1% weight mass {top1:.3f}',
          flush=True)
    json.dump(out, open(f'{QK}/qk_named_basis.json', 'w'), indent=2)

# ---- full-audit evaluation ----
MO_MEAN_G = MO_MEAN.to(DEV)
P64g, P256g = P64, P256


@torch.no_grad()
def forward_gen(idx, arm):
    dt = m.transformer.wte.weight.dtype
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    v1 = None
    B, T = idx.shape
    cos, sin = rope_tables(T, HD, idx.device, dt, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
    mo_c = [None]
    xin1_c = [None]
    yh_c = [None]
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        if li == 1:
            xin1_c[0] = x
        a = blk.attn
        hcur = F.rms_norm(x, (x.size(-1),))

        def factors(lin, name=None):
            z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,))
            if li == 1 and arm is not None and name is not None:
                mo = mo_c[0].float()
                if arm == '__static__':
                    pred = None
                else:
                    mod, Xk = models[arm], None
                    cm = (mo - MO_MEAN_G).reshape(-1, D)
                    yh = yh_c[0].float().reshape(-1, NH, HD)
                    ac = torch.cat([((yh[:, hh] - YMU[hh]) @ PA[hh]) for hh in range(NH)], 1) / STDA
                    ce = EWcur[idx.reshape(-1)]
                    feat = torch.cat([ce, ac], 1)
                    pred = mod(feat)                          # (B*T, 576)
                tab = TABLES[name][idx]
                zc = tab.clone()
                base = {'q1': 0, 'k1': 1, 'q2': 2, 'k2': 3}[name] * NH * R_AD
                for h in range(NH):
                    U = UB[(name, h)]
                    mu = MUB[(name, h)]
                    corr = mu if pred is None else \
                        mu + pred[:, base + h * R_AD: base + (h + 1) * R_AD] @ U.T
                    zc[:, :, h] += corr.reshape(B, T, HD) if pred is not None else mu
                z = zc.to(hcur.dtype)
            return apply_rot(z, cosb, sinb)

        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        q, k = factors(a.c_q, 'q1'), factors(a.c_k, 'k1')
        q2, k2 = factors(a.c_q2, 'q2'), factors(a.c_k2, 'k2')
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        y = yh4.reshape(B, T, -1)
        x = x + a.c_proj(y)
        mo_term = blk.mlp(F.rms_norm(x, (x.size(-1),)))
        if li == 0:
            mo_c[0] = mo_term
            yh_c[0] = yh4
        x = x + mo_term
    x = F.rms_norm(x, (x.size(-1),))
    return 30 * torch.tanh(m.lm_head(x) / 30)


@torch.no_grad()
def audit(arm, batch=4):
    tot, n = 0.0, 0
    for i in range(0, len(FINEWEB), batch):
        b = FINEWEB[i:i + batch].to(DEV)
        idx = b[:, :-1]
        logits = forward_gen(idx, arm).float()
        ce = F.cross_entropy(logits.reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


BASE = 3.07630
for aname in ARMS:
    ce = audit(aname)
    out[aname]['audit_dce'] = round(ce - BASE, 5)
    print(f'{aname}: full-audit dCE {ce - BASE:+.5f}', flush=True)
    json.dump(out, open(f'{QK}/qk_named_basis.json', 'w'), indent=2)
print('CTX GEN ZOO DONE', flush=True)
