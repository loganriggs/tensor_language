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
                use_oracle = isinstance(mode, tuple) and mode[1] == name
                for h in range(NH):
                    U = UB[(name, h)]
                    mu = MUB[(name, h)]
                    if use_oracle:
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
out = {'eval_mean_dce': round(float(delta.mean()), 5)}
print(f'generated model on eval docs: mean dCE {float(delta.mean()):+.5f}', flush=True)

# (b) worst-200 commonalities
T1 = SEQS.shape[1] - 1
top = delta.topk(200).indices


def feats_pos(flatidx):
    i, p = flatidx // T1, flatidx % T1
    tgt = int(SEQS[i, p + 1])
    cur = int(SEQS[i, p])
    tgt_s = tok.decode([tgt])
    cur_s = tok.decode([cur])
    row = SEQS[i, :p + 1].tolist()
    nl = 10121
    dist_nl = next((j for j, t in enumerate(reversed(row)) if tok.decode([t]).startswith('\n')), len(row))
    return {'sub_tgt': (not tgt_s.startswith(' ')) and tgt_s[:1].isalpha(),
            'sub_cur': (not cur_s.startswith(' ')) and cur_s[:1].isalpha(),
            'near_nl': dist_nl <= 3, 'dist_nl': dist_nl}


worst_f = [feats_pos(int(t)) for t in top.tolist()]
rand_idx = torch.randperm(len(delta))[:2000]
rand_f = [feats_pos(int(t)) for t in rand_idx.tolist()]


def frac(fs, k):
    return round(float(np.mean([f[k] for f in fs])), 3)


out['commonality'] = {k: {'worst200': frac(worst_f, k), 'baseline': frac(rand_f, k)}
                      for k in ('sub_tgt', 'sub_cur', 'near_nl')}
print('commonalities (worst vs baseline):', out['commonality'], flush=True)
snips = []
for t in top[:12].tolist():
    i, p = t // T1, t % T1
    snips.append({'dce': round(float(delta[t]), 2),
                  'ctx': tok.decode(SEQS[i, max(0, p - 12):p + 1].tolist()).replace('\n', '⏎'),
                  'tgt': tok.decode([int(SEQS[i, p + 1])]).replace('\n', '⏎')})
out['worst_snippets'] = snips

# (c) missing residual structure on eval docs
print('residual structure...', flush=True)
RES, DCEP = [], []
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
RES = torch.cat(RES)                                       # (P, 576)
res_norm = RES.norm(dim=1)
w_norm = float(res_norm[top].mean())
m_norm = float(res_norm.median())
out['residual'] = {'worst200_norm': round(w_norm, 3), 'median_norm': round(m_norm, 3),
                   'ratio': round(w_norm / m_norm, 2)}
Rw = RES[top].to(DEV)
_, Sv, _ = torch.linalg.svd(Rw - Rw.mean(0), full_matrices=False)
e2 = Sv ** 2
out['residual']['worst200_pca'] = {str(r): round(float(e2[:r].sum() / e2.sum()), 3)
                                   for r in (1, 4, 16, 64)}
ch_norm = RES[top].view(200, 36, R_AD).norm(dim=(0, 2))
names36 = [f'{n}_h{h}' for n, _ in MAPS for h in range(NH)]
topch = ch_norm.argsort(descending=True)[:5]
out['residual']['top_channels'] = [names36[i] for i in topch.tolist()]
print(f'residual: worst/median norm ratio {w_norm / m_norm:.2f}; worst-PCA top16 '
      f'{out["residual"]["worst200_pca"]["16"]}; top channels '
      f'{out["residual"]["top_channels"]}', flush=True)
json.dump(out, open(f'{QK}/qk_gen_error_analysis.json', 'w'), indent=2)

# (d) repair attribution per map
for name, _ in MAPS:
    lm_ = per_pos(('oracle_map', name))
    d = float((lm_ - base).flatten().mean())
    out[f'repair_{name}'] = round(d, 5)
    print(f'oracle repair of {name}: mean dCE {d:+.5f} (generated {float(delta.mean()):+.5f})',
          flush=True)
    json.dump(out, open(f'{QK}/qk_gen_error_analysis.json', 'w'), indent=2)

# (e) naming: oracle code dims vs nameable features (on cooc capture, train rows)
print('naming the interface...', flush=True)
feat_rows = []
for i in range(0, N_CAP, 4):
    for bi in range(4):
        seq = COOC[i + bi]
        row = seq[:-1].tolist()
        dist = 0
        for p in range(len(row)):
            s = tok.decode([row[p]])
            dist = 0 if s.startswith('\n') else dist + 1
            feat_rows.append(dist)
dist_nl = torch.tensor(feat_rows, dtype=torch.float)[:N]
sub_cur = torch.tensor([0.0] * N)
toks_flat = COOC[:N_CAP, :-1].reshape(-1)[:N]
dec_flags = []
for t in range(V):
    s = tok.decode([t])
    dec_flags.append(1.0 if (not s.startswith(' ') and s[:1].isalpha()) else 0.0)
dec_flags = torch.tensor(dec_flags)
sub_cur = dec_flags[toks_flat]
posidx = torch.arange(N).float() % 511
FEAT = torch.stack([torch.log1p(dist_nl), sub_cur, posidx / 511], 1).to(DEV)
FEAT = (FEAT - FEAT.mean(0)) / FEAT.std(0).clamp_min(1e-6)
naming = {}
for key in (('k1', 1), ('q1', 7), ('k1', 3)):
    yidx = [i for i, (n, _) in enumerate(MAPS) if n == key[0]][0] * NH * R_AD + key[1] * R_AD
    Yc = Y[:, yidx:yidx + R_AD]
    W = torch.linalg.lstsq(FEAT, Yc).solution
    pred = FEAT @ W
    r2 = 1 - float(((pred - Yc) ** 2).sum()) / float(((Yc - Yc.mean(0)) ** 2).sum())
    naming[f'{key[0]}_h{key[1]}'] = round(r2, 3)
out['naming_r2_cheap_features'] = naming
print('interface vs nameable features R2:', naming, flush=True)
json.dump(out, open(f'{QK}/qk_gen_error_analysis.json', 'w'), indent=2)
print('GEN ERROR ANALYSIS DONE', flush=True)
