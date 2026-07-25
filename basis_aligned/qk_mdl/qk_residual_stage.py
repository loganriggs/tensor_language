"""TICK 214 (Logan): (b-fixed) per-head JOINT oracle repairs (tick-213 mode bug
fixed — asserts on the patch this time) + RESIDUAL-STAGE zoo: train second-stage
models ON THE RESIDUAL (true adapter coords minus frozen stage-1 prediction) with NEW
inputs — a local WINDOW of token-identity codes (embedding-PCA-32 at offsets 0..-3),
testing the tick-212/213 hypothesis that the missing computation is fine lexical
context of recent tokens. Arms: linear window-only; swiglu window-only; linear
window+stage-1 code (control: old inputs alone should add ~nothing).

Original tick-213 header:
"""
"""TICK 213 (Logan): CLASSIFY the missing signal and LOCALIZE the damaged
interactions.

(a) Cluster the worst-512 positions by residual DIRECTION (cosine k-means, k=6) —
Logan's attribution-similarity classification. Per cluster: size, dominant channels,
subword statistics, text snippets.
(b) PER-HEAD JOINT oracle repair: restore oracle corrections for ALL FOUR maps of one
layer-1 head at a time (generated elsewhere). Tick 212 showed per-map repairs backfire
(errors couple through the pattern product); per-head-joint repair is the coupling-
respecting granularity — its ranking names which head's missing context causes the
damage, testably.
(c) MISSED-LINK analysis: at the worst-200 positions, compare exact vs generated
layer-1 attention patterns; per head, the distribution of |dPattern| over key-offsets
and whether the damaged keys are subword fragments — which INTERACTIONS are missed.

Derived from tick 212; original header:
"""
"""TICK 212 (Logan): what is the generator MISSING? Error analysis of the best
generated interface (mixed swiglu, +0.0319), in the tick-164 tradition that found the
anchor structure.

On 64 held-out documents (route-only patching: generated corrections enter ONLY the
layer-1 QK factors; the residual carries the true MLP output for all other readers):

(a) Per-position dCE of the generated model vs exact. Worst-200 positions dumped with
    context snippets.
(b) Commonality statistics, worst-200 vs all: fraction where the TARGET is a subword
    continuation (no leading space, alphabetic); fraction within 3 tokens after a
    newline; fraction where the current token is a subword fragment; mean distance to
    previous newline.
(c) The MISSING RESIDUAL in adapter coordinates: R = (true deviation - generated
    correction) per channel (576 dims total). Worst-200 vs median positions: norm
    ratio, channel breakdown (which map x head carries the miss), and PCA of the
    worst-position residuals (is what's missing COMMON structure — low rank — or
    idiosyncratic?).
(d) REPAIR ATTRIBUTION: on the same 64 documents, replace generated with ORACLE
    corrections for one factor map at a time (q1/k1/q2/k2) — which route's missing
    context carries the CE cost.
(e) NAMING the interface: R^2 of each of the 16 oracle code dims (per the three most
    important channels) against cheap position features: log distance to previous
    newline, subword-continuation flags (current and previous token), position index,
    inside-quote parity. Is the ten-dimensional signal humanly simple?
"""
import json
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
from transformers import AutoTokenizer

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
N_EST, TAU, N_CAP, R_AD, N_EVAL = 1024, 8.0, 256, 16, 64
tok = AutoTokenizer.from_pretrained('gpt2')

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
blk0 = m.transformer.h[0]
a1 = m.transformer.h[1].attn
MAPS = (('q1', a1.c_q), ('k1', a1.c_k), ('q2', a1.c_q2), ('k2', a1.c_k2))
SEQS = FINEWEB[:N_EVAL]


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
    x = x + a.c_proj(yh.reshape(B, T, -1))
    mo = blk0.mlp(F.rms_norm(x, (x.size(-1),)))
    x = x + mo
    blk1 = m.transformer.h[1]
    return blk1.lambdas[0] * x + blk1.lambdas[1] * x0, mo, yh


# ---- tables (shrunk) ----
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

# ---- refit generator pieces exactly as tick 211 (train split of cooc) ----
print('capture + refit generator...', flush=True)
MOs, S0s, YHs, DEVSd = [], [], [], {n: [] for n, _ in MAPS}
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
            DEVSd[name].append((fa.reshape(-1, NH, HD) - TABLES[name][ids]).cpu())
MO = torch.cat(MOs)
YH = torch.cat(YHs)
S0 = torch.cat(S0s)
N = MO.shape[0]
ntr = int(N * 0.9)
MO_MEAN = MO[:ntr].mean(0)
Xc = (MO[:ntr] - MO_MEAN).to(DEV)
_, _, VhM = torch.linalg.svd(Xc, full_matrices=False)
P64 = VhM[:64].T.contiguous()
del Xc
UB, MUB = {}, {}
Ys = []
for name, _ in MAPS:
    Dv = torch.cat(DEVSd[name])
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
DEVSd = None
Y = torch.cat(Ys, 1).to(DEV)
C64 = ((MO - MO_MEAN) @ P64.cpu()).to(DEV)
STD64 = C64[:ntr].std(0).clamp_min(1e-6)
C64 = C64 / STD64
PA, YMU = {}, {}
acs = []
for hh in range(NH):
    Xh = YH[:, hh].to(DEV)
    YMU[hh] = Xh[:ntr].mean(0)
    _, _, Vhh = torch.linalg.svd(Xh[:ntr] - YMU[hh], full_matrices=False)
    PA[hh] = Vhh[:8].T.contiguous()
    acs.append(((Xh - YMU[hh]) @ PA[hh]).cpu())
    del Xh
CATT = torch.cat(acs, 1).to(DEV)
STDA = CATT[:ntr].std(0).clamp_min(1e-6)
CATT = CATT / STDA
mrms = MO.pow(2).mean(1).sqrt()
SCAL = torch.stack([S0, 1 / S0.clamp_min(1e-6), mrms, 1 / mrms.clamp_min(1e-6)], 1).to(DEV)
SCAL_MU = SCAL[:ntr].mean(0)
SCAL_SD = SCAL[:ntr].std(0).clamp_min(1e-6)
SCAL = (SCAL - SCAL_MU) / SCAL_SD
CMIX = torch.cat([C64, CATT, SCAL], 1)


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


gen = Skip(140, Bilin(140, 130, 576, gate='silu'), 576).to(DEV)
opt = torch.optim.Adam(gen.parameters(), lr=1e-3)
g = torch.Generator().manual_seed(0)
for step in range(8000):
    bi = torch.randint(0, ntr, (8192,), generator=g).to(DEV)
    loss = F.mse_loss(gen(CMIX[bi]), Y[bi])
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
    opt.step()
    if step in (5000, 7000):
        for pg in opt.param_groups:
            pg['lr'] *= 0.3
gen.eval()
print('generator refit', flush=True)

# ---- eval pass on 64 held-out docs: per-position dCE (generated vs exact) + residuals ----


@torch.no_grad()
def feats_of(mo, yh, xin1):
    cm = (mo.float().reshape(-1, D) - MO_MEAN.to(DEV))
    ac = torch.cat([((yh.float().reshape(-1, NH, HD)[:, hh] - YMU[hh]) @ PA[hh])
                    for hh in range(NH)], 1) / STDA
    s0 = xin1.float().pow(2).mean(-1).sqrt().reshape(-1, 1)
    mr = mo.float().pow(2).mean(-1).sqrt().reshape(-1, 1)
    sc = torch.cat([s0, 1 / s0.clamp_min(1e-6), mr, 1 / mr.clamp_min(1e-6)], 1)
    sc = (sc - SCAL_MU) / SCAL_SD
    return torch.cat([(cm @ P64) / STD64, ac, sc], 1)


@torch.no_grad()
def forward_arm(idx, mode):
    """mode: None exact; 'gen' generated; ('oracle_map', name) oracle for that map,
    generated for others."""
    dt = m.transformer.wte.weight.dtype
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    v1 = None
    B, T = idx.shape
    cos, sin = rope_tables(T, HD, idx.device, dt, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
    mo_c, yh_c, xin1_c = [None], [None], [None]
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        if li == 1:
            xin1_c[0] = x
        a = blk.attn
        hcur = F.rms_norm(x, (x.size(-1),))

        def factors(lin, name=None):
            z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,))
            if li == 1 and mode is not None and name is not None:
                pred = gen(feats_of(mo_c[0], yh_c[0], xin1_c[0]))
                tab = TABLES[name][idx]
                zc = tab.clone()
                base = {'q1': 0, 'k1': 1, 'q2': 2, 'k2': 3}[name] * NH * R_AD
                oh = mode[1] if (isinstance(mode, tuple) and mode[0] == 'oracle_head') else None
                use_oracle = isinstance(mode, tuple) and mode[0] == 'oracle_map' and mode[1] == name
                for h in range(NH):
                    U = UB[(name, h)]
                    mu = MUB[(name, h)]
                    if use_oracle or (oh is not None and h == oh):
                        dv = z.float()[:, :, h].reshape(-1, HD) - tab[:, :, h].reshape(-1, HD)
                        coord = (dv - mu) @ U
                    else:
                        coord = pred[:, base + h * R_AD: base + (h + 1) * R_AD]
                    zc[:, :, h] += (mu + coord @ U.T).reshape(B, T, HD)
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
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        mo_term = blk.mlp(F.rms_norm(x, (x.size(-1),)))
        if li == 0:
            mo_c[0] = mo_term
            yh_c[0] = yh4
        x = x + mo_term
    x = F.rms_norm(x, (x.size(-1),))
    return 30 * torch.tanh(m.lm_head(x) / 30)


@torch.no_grad()
def per_pos(mode, batch=4):
    outs = []
    for i in range(0, N_EVAL, batch):
        b = SEQS[i:i + batch].to(DEV)
        idx = b[:, :-1]
        logits = forward_arm(idx, mode).float()
        ls = F.cross_entropy(logits.reshape(-1, V), b[:, 1:].reshape(-1), reduction='none')
        outs.append(ls.view(b.shape[0], -1).cpu())
    return torch.cat(outs, 0)



base = per_pos(None)
genl = per_pos('gen')
delta = (genl - base).flatten()
T1 = SEQS.shape[1] - 1
out = {'eval_mean_dce': round(float(delta.mean()), 5)}
print(f"generated mean dCE {float(delta.mean()):+.5f}", flush=True)

# ---- residual capture (as tick 212) ----
RES = []
with torch.no_grad():
    for i in range(0, N_EVAL, 4):
        idx = SEQS[i:i + 4].to(DEV)[:, :-1]
        xin1, mo, yh = block01(idx)
        h1n = F.rms_norm(xin1, (D,))
        pred = gen(feats_of(mo, yh, xin1))
        rows = []
        for name, lin in MAPS:
            fa = F.rms_norm(lin(h1n).view(*idx.shape, NH, HD).float(), (HD,))
            dv = fa.reshape(-1, NH, HD) - TABLES[name][idx.reshape(-1)]
            base_i = {'q1': 0, 'k1': 1, 'q2': 2, 'k2': 3}[name] * NH * R_AD
            for h in range(NH):
                true_c = (dv[:, h] - MUB[(name, h)]) @ UB[(name, h)]
                rows.append(true_c - pred[:, base_i + h * R_AD: base_i + (h + 1) * R_AD])
        RES.append(torch.cat(rows, 1).cpu())
RES = torch.cat(RES)
names36 = [f"{n}_h{h}" for n, _ in MAPS for h in range(NH)]


# ---- (b) per-head joint oracle repairs, FIXED ----
for h in range(NH):
    lm_ = per_pos(('oracle_head', h))
    d = float((lm_ - base).flatten().mean())
    out[f'repair_head_{h}'] = round(d, 5)
    print(f"joint oracle repair of l1-h{h}: dCE {d:+.5f} (generated "
          f"{float(delta.mean()):+.5f})", flush=True)
    json.dump(out, open(f'{QK}/qk_residual_stage.json', 'w'), indent=2)

# ---- residual-stage zoo ----
print('residual stage...', flush=True)
EPC = torch.linalg.svd(F.rms_norm(wte, (D,)) - F.rms_norm(wte, (D,)).mean(0),
                       full_matrices=False).Vh[:32].T.contiguous()
EW = (F.rms_norm(wte, (D,)) @ EPC)                       # (V, 32) token window code
EW = EW / EW.std(0).clamp_min(1e-6)


def window_feats_from_ids(ids2d):
    """(B,T) -> (B*T, 128): codes of tokens at offsets 0,-1,-2,-3 (clamped)."""
    B, T = ids2d.shape
    cs = []
    for off in range(4):
        sh = torch.cat([ids2d[:, :1].expand(B, off), ids2d[:, :T - off]], 1) if off else ids2d
        cs.append(EW[sh.reshape(-1)])
    return torch.cat(cs, 1)


# training features from the cooc capture: rebuild ids窗口
IDS_ALL = COOC[:N_CAP, :-1].reshape(-1)[:N]
IDS2D = COOC[:N_CAP, :-1]
WFEAT = window_feats_from_ids(IDS2D.to(DEV))[:N]
with torch.no_grad():
    PRED1 = torch.cat([gen(CMIX[i:i + 65536]) for i in range(0, N, 65536)])
RTGT = Y - PRED1
arms2 = {
    'lin_window': (nn.Linear(128, 576), WFEAT),
    'swiglu_window': (Skip(128, Bilin(128, 120, 576, gate='silu'), 576), WFEAT),
    'lin_win_plus_code': (nn.Linear(268, 576), torch.cat([WFEAT, CMIX], 1)),
}
stage2 = {}
for aname, (mod, Xf) in arms2.items():
    mod = mod.to(DEV)
    opt = torch.optim.Adam(mod.parameters(), lr=1e-3)
    g2 = torch.Generator().manual_seed(0)
    for step in range(8000):
        bi = torch.randint(0, ntr, (8192,), generator=g2).to(DEV)
        loss = F.mse_loss(mod(Xf[bi]), RTGT[bi])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
        opt.step()
        if step in (5000, 7000):
            for pg in opt.param_groups:
                pg['lr'] *= 0.3
    with torch.no_grad():
        pr = mod(Xf[ntr:])
        r2 = 1 - float(((pr - RTGT[ntr:]) ** 2).sum()) / float(
            ((RTGT[ntr:] - RTGT[:ntr].mean(0)) ** 2).sum())
    out[f's2_{aname}_val_r2'] = round(r2, 4)
    stage2[aname] = mod.eval()
    print(f"stage2 {aname}: residual val R2 {r2:.4f}", flush=True)
    json.dump(out, open(f'{QK}/qk_residual_stage.json', 'w'), indent=2)

# audit the best window arm end-to-end (stage1 + stage2)
best2 = max([k for k in arms2], key=lambda k: out[f's2_{k}_val_r2'])
print(f"auditing stage1+stage2 ({best2})...", flush=True)
S2 = stage2[best2]


@torch.no_grad()
def per_pos_two(batch=4):
    outs = []
    for i in range(0, N_EVAL, batch):
        b = SEQS[i:i + batch].to(DEV)
        idx = b[:, :-1]
        dt = m.transformer.wte.weight.dtype
        x = m.transformer.wte(idx)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        v1 = None
        B, T = idx.shape
        cos, sin = rope_tables(T, HD, idx.device, dt, 'bf16')
        cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
        mo_c, yh_c, xin1_c = [None], [None], [None]
        for li, blk in enumerate(m.transformer.h):
            x = blk.lambdas[0] * x + blk.lambdas[1] * x0
            if li == 1:
                xin1_c[0] = x
            a = blk.attn
            hcur = F.rms_norm(x, (x.size(-1),))

            def factors(lin, name=None):
                z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,))
                if li == 1 and name is not None:
                    pred = gen(feats_of(mo_c[0], yh_c[0], xin1_c[0]))
                    wf = window_feats_from_ids(idx)
                    if best2 == 'lin_win_plus_code':
                        wf = torch.cat([wf, feats_of(mo_c[0], yh_c[0], xin1_c[0])], 1)
                    pred = pred + S2(wf)
                    tab = TABLES[name][idx]
                    zc = tab.clone()
                    base_i = {'q1': 0, 'k1': 1, 'q2': 2, 'k2': 3}[name] * NH * R_AD
                    for h in range(NH):
                        U = UB[(name, h)]
                        mu = MUB[(name, h)]
                        coord = pred[:, base_i + h * R_AD: base_i + (h + 1) * R_AD]
                        zc[:, :, h] += (mu + coord @ U.T).reshape(B, T, HD)
                    z = zc.to(hcur.dtype)
                return apply_rot(z, cosb, sinb)

            v = a.c_v(hcur).view(B, T, NH, HD)
            if v1 is None:
                v1 = v
            v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
            q, k = factors(a.c_q, 'q1'), factors(a.c_k, 'k1')
            q2f, k2f = factors(a.c_q2, 'q2'), factors(a.c_k2, 'k2')
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
            s2v = torch.einsum('bqhd,bkhd->bhqk', q2f, k2f) / HD
            pat = (s1 * s2v).masked_fill(~mask, 0.0)
            yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
            x = x + a.c_proj(yh4.reshape(B, T, -1))
            mo_term = blk.mlp(F.rms_norm(x, (x.size(-1),)))
            if li == 0:
                mo_c[0] = mo_term
                yh_c[0] = yh4
            x = x + mo_term
        x = F.rms_norm(x, (x.size(-1),))
        logits = 30 * torch.tanh(m.lm_head(x) / 30)
        ls = F.cross_entropy(logits.float().reshape(-1, V), b[:, 1:].reshape(-1),
                             reduction='none')
        outs.append(ls.view(b.shape[0], -1).cpu())
    return torch.cat(outs, 0)


two = per_pos_two()
d2 = float((two - base).flatten().mean())
out['stage12_eval_dce'] = round(d2, 5)
print(f"stage1+stage2 ({best2}) eval dCE {d2:+.5f} (stage1 {float(delta.mean()):+.5f}, "
      f"oracle-16 ~+0.0113)", flush=True)
json.dump(out, open(f'{QK}/qk_residual_stage.json', 'w'), indent=2)
print('RESIDUAL STAGE DONE', flush=True)
