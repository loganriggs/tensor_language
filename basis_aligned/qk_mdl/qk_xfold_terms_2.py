"""Fix for qk_xfold_terms.py PASS 4: the token-conditional means were fitted on the HELD slice
itself (singleton tokens fit perfectly -> inflated R2; shuffled control -0.51 exposed it).
Here the token-conditional term means are fitted on TRAIN FW[0:256,:128] and the residual
variance evaluated on HELD FW[448:600,:128] (unseen-token fallback = train global term mean),
with the shuffled-token control re-run the same way (expected ~0 now). Also reports the
context-versus-token split restricted to train-count>=5 tokens. Everything else (terms, groups,
forward) VERBATIM from qk_xfold_terms.py. Updates qk_xfold_terms.json in place under
'token_variance_fraction_trainfit'. Batch 4, <4GB."""
import json, sys, time, subprocess
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_xfold_terms.json'

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
V = cfg['vocab_size']; NL = len(m.transformer.h)
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
TRAIN = FW[0:256, :128].to(DEV)
HELD = FW[448:600, :128].to(DEV)
B0 = 4; LI = 3
STR, _ = TRAIN.shape; S_, T_ = HELD.shape

GNAMES = ['E', 'Ae', 'Ar', 'M0', 'M1', 'M2']
NG = 6
PAIRS = [(i, j) for i in range(NG) for j in range(i, NG)]
PNAMES = [f'{GNAMES[i]}x{GNAMES[j]}' for (i, j) in PAIRS]
NT = len(PAIRS)
b3 = m.transformer.h[LI].mlp
Lw = b3.Left.weight.detach().float(); Rw = b3.Right.weight.detach().float()
Dw = b3.Down.weight.detach().float(); bias = b3.Down_bias.detach().float()

res = json.load(open(OUT))
TOPN = res['energy_rank'][:6]
TOPK = [PNAMES.index(n) for n in TOPN]
print("terms:", TOPN, flush=True)

def pair_terms(groups, xpre):
    rho2 = xpre.pow(2).sum(-1, keepdim=True) / D
    PL = [g @ Lw.T for g in groups]; PR = [g @ Rw.T for g in groups]
    terms = []
    for (i, j) in PAIRS:
        t_ = 0.5 * ((PL[i] * PR[j] + PL[j] * PR[i]) @ Dw.T)
        if i != j: t_ = 2.0 * t_
        terms.append(t_ / rho2)
    return terms

@torch.no_grad()
def fwd_terms(idx):
    """Truncated forward through layer 3, returns the 6 tracked terms (VERBATIM qk_xfold_terms)."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cE = torch.ones((), device=DEV); SA = torch.zeros_like(x); Ml = []
    for li in range(LI+1):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        cE = blk.lambdas[0]*cE + blk.lambdas[1]
        SA = blk.lambdas[0]*SA; Ml = [blk.lambdas[0]*mm for mm in Ml]
        def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        aout = a.c_proj(yh.reshape(B, T, -1)); x = x + aout
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if li == LI:
            groups = [cE*x0, SA, aout, Ml[0], Ml[1], Ml[2]]
            return {kk: terms_k for kk, terms_k in
                    zip(range(NT), pair_terms(groups, x))}
        x = x + mo
        SA = SA + aout; Ml.append(mo)

# compact vocab over train+held
ALLTOK = torch.unique(torch.cat([TRAIN.reshape(-1), HELD.reshape(-1)]))
NAV = ALLTOK.numel()
LUT = torch.zeros(V, dtype=torch.long, device=DEV); LUT[ALLTOK] = torch.arange(NAV, device=DEV)
print(f"compact vocab {NAV}", flush=True)

# PASS A: train-fitted token-conditional means of the 6 terms
tsum = {kk: torch.zeros(NAV, D, device=DEV) for kk in TOPK}
tcnt = torch.zeros(NAV, device=DEV)
gsum = {kk: torch.zeros(D, device=DEV) for kk in TOPK}
ntr = 0
for i in range(0, STR, B0):
    idx = TRAIN[i:i+B0]
    T6 = fwd_terms(idx)
    fl = LUT[idx.reshape(-1)]
    for kk in TOPK:
        tv = T6[kk].reshape(-1, D).float()
        tsum[kk].index_add_(0, fl, tv); gsum[kk] += tv.sum(0)
    tcnt.index_add_(0, fl, torch.ones_like(fl, dtype=torch.float))
    ntr += fl.numel()
GM = {kk: gsum[kk]/ntr for kk in TOPK}
seen = tcnt > 0
TOKMEAN = {kk: torch.where(seen[:, None], tsum[kk]/tcnt.clamp_min(1)[:, None], GM[kk][None]) for kk in TOPK}
cnt5 = tcnt >= 5
print(f"train-seen tokens {int(seen.sum())}, count>=5 tokens {int(cnt5.sum())}", flush=True)
del tsum, gsum

# PASS B: held residuals (train-fitted token means, shuffled control, count>=5 restriction)
perm = torch.randperm(HELD.numel(), device=DEV)
SHUF = HELD.reshape(-1)[perm].reshape(S_, T_)
ss = {kk: {'tok': 0.0, 'shuf': 0.0, 'tot': 0.0, 'tok5': 0.0, 'tot5': 0.0} for kk in TOPK}
for i in range(0, S_, B0):
    idx = HELD[i:i+B0]; sidx = SHUF[i:i+B0]
    T6 = fwd_terms(idx)
    fl = LUT[idx.reshape(-1)]; fls = LUT[sidx.reshape(-1)]
    m5 = cnt5[fl]
    for kk in TOPK:
        tv = T6[kk].reshape(-1, D).float()
        ss[kk]['tok'] += float((tv - TOKMEAN[kk][fl]).pow(2).sum())
        ss[kk]['shuf'] += float((tv - TOKMEAN[kk][fls]).pow(2).sum())
        ss[kk]['tot'] += float((tv - GM[kk][None]).pow(2).sum())
        ss[kk]['tok5'] += float((tv - TOKMEAN[kk][fl])[m5].pow(2).sum())
        ss[kk]['tot5'] += float((tv - GM[kk][None])[m5].pow(2).sum())
out = {}
for kk in TOPK:
    r2 = 1 - ss[kk]['tok']/ss[kk]['tot']
    r2s = 1 - ss[kk]['shuf']/ss[kk]['tot']
    r25 = 1 - ss[kk]['tok5']/ss[kk]['tot5']
    out[PNAMES[kk]] = {'token_R2_trainfit': round(r2, 4), 'shuffled_control_R2': round(r2s, 4),
                       'token_R2_count5plus': round(r25, 4)}
    print(f"  {PNAMES[kk]:8s} token R2 {r2:.4f} (shuffled {r2s:+.4f}; count>=5 subset {r25:.4f})", flush=True)
res['token_variance_fraction_trainfit'] = {
    'method': 'token-conditional term means fitted on TRAIN FW[0:256], residuals on HELD; '
              'fallback global train mean for unseen tokens; shuffled-token control',
    'per_term': out}
json.dump(res, open(OUT, 'w'), indent=1)
print("QK XFOLD TERMS 2 DONE", flush=True)
