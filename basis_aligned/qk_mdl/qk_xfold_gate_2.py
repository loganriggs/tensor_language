"""Follow-up to qk_xfold_gate.py: HOW does the Mr partner gate h.L7.0 -- amplitude or direction?

qk_xfold_gate found: partner-group norm is statistically IDENTICAL at positions where the
h x Mr term helps vs hurts (1340 vs 1286 vs inert 1329) -- so the gate is not a scalar volume
knob. Hypothesis: the partner RE-AIMS h -- the head's own write points the same way at helping
and hurting positions, but the elementwise product with Mr in the hidden basis sends the term
output in opposite directions. Test with centroid cosines over the top-200 helping and top-200
hurting positions (top-quartile h-norm, same selection as the parent script):
  cos( centroid h_dev[help], centroid h_dev[hurt] )      -- expect HIGH (head signal similar)
  cos( centroid term_dev[help], centroid term_dev[hurt] ) -- expect LOW/NEGATIVE (output re-aimed)
plus the same cosines for the Mr partner stream itself, and per-example numbers for the parent
script's concrete examples. All machinery verbatim from qk_xfold_gate.py. Batch 4, <4GB."""
import json, sys, time, subprocess
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_xfold_gate.json'
LI_T, H_T = 7, 0; LSTAR = 7

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
HELD = FW[448:600, :128].to(DEV); B0 = 4
S_, T_ = HELD.shape

res = json.load(open(OUT))
partner_term = res['fold']['partner_term']
assert partner_term == 'Mrxh', partner_term

GN = ['E', 'Ae', 'Ar', 'Me', 'Mr', 'h']
NG = 6
PAIRS = [(i, j) for i in range(NG) for j in range(i, NG)]
PNAMES = [f'{GN[i]}x{GN[j]}' for (i, j) in PAIRS]
NT = len(PAIRS)
KP = PNAMES.index('Mrxh')

b = m.transformer.h[LSTAR].mlp
Lw = b.Left.weight.detach().float(); Rw = b.Right.weight.detach().float()
Dw = b.Down.weight.detach().float(); bias = b.Down_bias.detach().float()

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
def fwd(idx, mode, TMEAN=None, MEANF=None, cap=None):
    """mode: 'base' | 'collect' (term/mo means at L7) | 'dropMrxh' | 'capture' (streams+term at L7)."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cE = torch.ones((), device=DEV)
    SA = torch.zeros_like(x); SM = torch.zeros_like(x); MR = torch.zeros_like(x)
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        if li <= LSTAR:
            cE = blk.lambdas[0]*cE + blk.lambdas[1]
            SA = blk.lambdas[0]*SA; SM = blk.lambdas[0]*SM; MR = blk.lambdas[0]*MR
        def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        aout = a.c_proj(yh4.reshape(B, T, -1)); x = x + aout
        if li == LI_T:
            Wr = a.c_proj.weight.view(D, NH, HD)
            rh = torch.einsum('btc,dc->btd', yh4[:, :, H_T, :], Wr[:, H_T, :])
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if li == LSTAR and mode != 'base':
            groups = [cE*x0, SA, aout - rh, SM, MR, rh]
            terms = pair_terms(groups, x)
            if mode == 'collect':
                for kk in range(NT): cap['tsum'][kk] += terms[kk].sum(0)
                cap['rhsum'] += rh.sum(0); cap['mrsum'] += MR.sum(0)
            elif mode == 'dropMrxh':
                new = MEANF.unsqueeze(0).expand(B, -1, -1)
                for kk in range(NT):
                    if kk != KP: new = new + (terms[kk] - TMEAN[kk])
                mo = new.to(x.dtype)
            elif mode == 'capture':
                cap['h'].append(rh.cpu()); cap['mr'].append(MR.cpu())
                cap['term'].append((terms[KP] - TMEAN[KP]).cpu())
            del terms, groups
        x = x + mo
        if li < LSTAR:
            SA = SA + aout; SM = SM + MR; MR = mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1), reduction='none').view(B, T-1)
    return ce

print("PASS 1: means ...", flush=True)
cap = {'tsum': [torch.zeros(T_, D, device=DEV) for _ in range(NT)],
       'rhsum': torch.zeros(T_, D, device=DEV), 'mrsum': torch.zeros(T_, D, device=DEV)}
base_l = []
for i in range(0, S_, B0):
    base_l.append(fwd(HELD[i:i+B0], 'collect', cap=cap).cpu())
base = torch.cat(base_l, 0)
TMEAN = torch.stack([t/S_ for t in cap['tsum']])
MEANF = TMEAN.sum(0) + bias
RH_MEAN = cap['rhsum']/S_; MR_MEAN = cap['mrsum']/S_

print("PASS 2: drop Mrxh per-position ...", flush=True)
drop_l = []
for i in range(0, S_, B0):
    drop_l.append(fwd(HELD[i:i+B0], 'dropMrxh', TMEAN=TMEAN, MEANF=MEANF).cpu())
dce = (torch.cat(drop_l, 0) - base).numpy()          # (S,T-1)
mn = float(dce.mean())
print(f"drop_Mrxh dCE {mn:+.5f}  (parent {res['fold']['configs']['drop_Mrxh']['dCE']:+.5f})", flush=True)

print("PASS 3: capture streams ...", flush=True)
cap2 = {'h': [], 'mr': [], 'term': []}
for i in range(0, S_, B0):
    fwd(HELD[i:i+B0], 'capture', TMEAN=TMEAN, cap=cap2)
Hs = torch.cat(cap2['h'], 0).numpy()                 # (S,T,D) raw h write
Mr = torch.cat(cap2['mr'], 0).numpy()
Tm = torch.cat(cap2['term'], 0).numpy()              # term deviation
Hdev = Hs - RH_MEAN.cpu().numpy()[None]
Mrdev = Mr - MR_MEAN.cpu().numpy()[None]

hnorm = np.linalg.norm(Hs, axis=-1)
hq = np.quantile(hnorm[:, :T_-1], 0.75)
sel = hnorm[:, :T_-1] >= hq
d = dce.copy(); d[~sel] = 0.0
flatidx = np.argsort(-d, axis=None)
si, pi = np.unravel_index(flatidx, d.shape)
help_pos = list(zip(si[:200], pi[:200]))
flatidx2 = np.argsort(d, axis=None)
si2, pi2 = np.unravel_index(flatidx2, d.shape)
hurt_pos = list(zip(si2[:200], pi2[:200]))

def centroid(X, poss):
    v = np.stack([X[s, p] for s, p in poss], 0)
    return v.mean(0), v

def coss(a, b): return float(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))

cH_help, vH_help = centroid(Hdev, help_pos); cH_hurt, vH_hurt = centroid(Hdev, hurt_pos)
cT_help, vT_help = centroid(Tm, help_pos);   cT_hurt, vT_hurt = centroid(Tm, hurt_pos)
cM_help, _ = centroid(Mrdev, help_pos);      cM_hurt, _ = centroid(Mrdev, hurt_pos)
# raw (uncentered) h too -- the head write itself
cHr_help, _ = centroid(Hs, help_pos); cHr_hurt, _ = centroid(Hs, hurt_pos)

out2 = {
 'selection': 'top-quartile h-norm; top-200 helping vs top-200 hurting positions by per-position '
              'paired dCE of dropping the h x Mr term (parent-script convention)',
 'centroid_cosines': {
    'h_dev_help_vs_hurt': round(coss(cH_help, cH_hurt), 4),
    'h_raw_help_vs_hurt': round(coss(cHr_help, cHr_hurt), 4),
    'Mr_dev_help_vs_hurt': round(coss(cM_help, cM_hurt), 4),
    'term_dev_help_vs_hurt': round(coss(cT_help, cT_hurt), 4)},
 'within_set_coherence': {
    'term_help_mean_cos_to_centroid': round(float(np.mean([coss(v, cT_help) for v in vT_help])), 4),
    'term_hurt_mean_cos_to_centroid': round(float(np.mean([coss(v, cT_hurt) for v in vT_hurt])), 4),
    'h_help_mean_cos_to_centroid': round(float(np.mean([coss(v, cH_help) for v in vH_help])), 4),
    'h_hurt_mean_cos_to_centroid': round(float(np.mean([coss(v, cH_hurt) for v in vH_hurt])), 4)},
 'drop_Mrxh_dCE_check': round(mn, 5)}
# per-example: the parent's concrete examples
ex_out = []
for grp in ['help_examples', 'hurt_examples']:
    for e in res['examples'][grp][:3]:
        s, p = e['seq'], e['pos']
        ex_out.append({'context_tail': e['context'][-45:], 'kind': grp[:4],
                       'dCE_here': e['dCE_drop_term_here'],
                       'h_cos_to_help_centroid': round(coss(Hdev[s, p], cH_help), 3),
                       'term_cos_to_help_centroid': round(coss(Tm[s, p], cT_help), 3),
                       'term_cos_to_hurt_centroid': round(coss(Tm[s, p], cT_hurt), 3)})
out2['per_example'] = ex_out
print(json.dumps(out2, indent=1), flush=True)
res['gate_mechanism'] = out2
json.dump(res, open(OUT, 'w'), indent=1)
print("QK XFOLD GATE 2 DONE", flush=True)
