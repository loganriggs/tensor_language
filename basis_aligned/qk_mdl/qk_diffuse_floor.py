"""TICK 221 (Logan): WHAT is the diffuse floor? H1 duty-cycle (damage tracks the
position's det-channel firing), H2 bias-supply (mean pattern-weighted write restored
as a constant cancels low-firing damage), H3 neither.

Original tick-219b header:
"""
"""TICK 219b: the layer-0 double dissociation (contrast to the layer-1 negative).
Same four-condition protocol on l0 head 3's determiner channel (its top archetypes
{a/an} and {The} from the minimal inventory): T0 exact, T1 channel removed, T2 channel
only (h3 keys rank-reduced to the det detectors), T3 head zeroed. Split metric: dCE on
positions whose current token is in the archetypes' top-token set vs others. At layer
0, head 3 is near-load-bearing, so the prediction is a TRUE dissociation.

Original tick-199 header:
"""
"""TICK 199 (Logan): are the layer-0 head-3 archetype directions PRIVILEGED, or would
any equal-sized ablation do the same? Hypothesis: the archetypes align with the sparse
interaction structure (with itself, the embedding, and the bilinear read-out), so
removing them should (i) cost more per unit of pattern actually removed and (ii)
concentrate its damage on few positions, versus energy-matched generic controls.

Arms (all on layer-0 head 3, displayed minimal inventory m=512 k=4):
  arch1   : top-1 archetype channel projected out of both key tables
  arch10  : top-10 archetype span projected out
  pca10   : top-10 PCA directions of the p-weighted key tables projected out
            (biggest variance directions, NOT interaction-fitted)
  rand10  : random 10-dim subspace projected out
  shrink1 : uniform score shrink beta on both branches, calibrated so the removed
            pattern energy E_p x p[(dP)^2] matches arch1
  shrink10: same, matched to arch10
Matching metric: pattern energy removed, E_{i,t ~ p x p}[(P_ablated - P)^2], sampled
over 8 x 4096^2 token pairs. Evaluation: per-position CE on 128 held-out documents
(65k predictions): mean dCE, dCE per unit removed energy, and concentration (share of
total positive damage carried by the top 0.1% / 1% of positions; fraction of positions
with |dCE| > 0.01).
"""
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, reference_forward, rope_tables, apply_rot
from tier2_folding import branch_factors, scores_from_factors

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
H, N_SEQ = 3, 128

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
TAB = {}
for br, (qn, kn) in ((1, ('q1', 'k1')), (2, ('q2', 'k2'))):
    qh, kh = branch_factors(m, br)
    TAB[qn], TAB[kn] = qh.float().to(DEV), kh.float().to(DEV)
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQS = FINEWEB[:N_SEQ]
QP = (torch.bincount(FINEWEB.flatten(), minlength=V).float() + 0.5).to(DEV)
QP = QP / QP.sum()

mh_pt = torch.load(f'{QK}/qk_minimal_heads.pt', map_location=DEV)
P3 = mh_pt[f'h{H}']
Dn3 = P3['Dm'].to(DEV)
Dn3 = Dn3 / Dn3.norm(dim=1, keepdim=True).clamp_min(1e-8)
U3 = P3['U'].to(DEV)


def arch_dirs(rs):
    g1s, g2s = [], []
    for r in rs:
        g1 = Dn3[:, :HD].T @ U3[:, r]
        g2 = Dn3[:, HD:2 * HD].T @ U3[:, r]
        g1s.append(g1 / g1.norm().clamp_min(1e-12))
        g2s.append(g2 / g2.norm().clamp_min(1e-12))
    return (torch.linalg.qr(torch.stack(g1s, 1)).Q, torch.linalg.qr(torch.stack(g2s, 1)).Q)


def project_tabs(Q1, Q2):
    tabs = {k: v.clone() for k, v in TAB.items()}
    tabs['k1'][:, H] -= (tabs['k1'][:, H] @ Q1) @ Q1.T
    tabs['k2'][:, H] -= (tabs['k2'][:, H] @ Q2) @ Q2.T
    return tabs


def shrink_tabs(beta):
    tabs = {k: v.clone() for k, v in TAB.items()}
    tabs['k1'][:, H] *= beta
    tabs['k2'][:, H] *= beta
    return tabs


@torch.no_grad()
def pattern_energy(tabs, n_batch=8, n=4096, seed=0):
    """E_{i,t~pxp}[(P_abl - P)^2] with static (non-rotary) scores — the energy meter."""
    g = torch.Generator().manual_seed(seed)
    tot = 0.0
    for _ in range(n_batch):
        si = torch.multinomial(QP.cpu(), n, replacement=True, generator=g).to(DEV)
        ti = torch.multinomial(QP.cpu(), n, replacement=True, generator=g).to(DEV)
        s1 = TAB['q1'][si, H] @ TAB['k1'][ti, H].T / HD
        s2 = TAB['q2'][si, H] @ TAB['k2'][ti, H].T / HD
        a1 = TAB['q1'][si, H] @ tabs['k1'][ti, H].T / HD
        a2 = TAB['q2'][si, H] @ tabs['k2'][ti, H].T / HD
        tot += float(((s1 * s2 - a1 * a2) ** 2).mean())
    return tot / n_batch


@torch.no_grad()
def per_pos_loss(tabs, batch=4):
    outs = []
    for i in range(0, N_SEQ, batch):
        b = SEQS[i:i + batch].to(DEV)
        idx = b[:, :-1]

        def patch(li, s1, s2):
            if li != 0:
                return s1, s2
            n1 = scores_from_factors(tabs['q1'], tabs['k1'], idx, HD)
            n2 = scores_from_factors(tabs['q2'], tabs['k2'], idx, HD)
            return n1.to(s1.dtype), n2.to(s2.dtype)

        logits = reference_forward(m, idx, 'bf16',
                                   score_patch=None if tabs is None else patch).float()
        ls = F.cross_entropy(logits.reshape(-1, V), b[:, 1:].reshape(-1), reduction='none')
        outs.append(ls.view(b.shape[0], -1).cpu())
    return torch.cat(outs, 0)


base = per_pos_loss(None)
print(f'baseline mean CE {float(base.mean()):.5f}', flush=True)


from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('gpt2')
P3 = mh_pt['h3']
Dn3 = P3['Dm'].to(DEV)
Dn3 = Dn3 / Dn3.norm(dim=1, keepdim=True).clamp_min(1e-8)
U3 = P3['U'].to(DEV)
DET_RS = [0, 2]
G1 = torch.linalg.qr(torch.stack([Dn3[:, :HD].T @ U3[:, r] for r in DET_RS], 1)).Q
G2 = torch.linalg.qr(torch.stack([Dn3[:, HD:2 * HD].T @ U3[:, r] for r in DET_RS], 1)).Q
kc3 = 2
z3 = torch.relu((torch.cat([TAB['k1'][:, H], TAB['k2'][:, H],
                            torch.zeros(V, HD, device=DEV)], 1) - 0) @ torch.zeros(1, 1, device=DEV).expand(384, Dn3.shape[0]).T) if False else None
S3 = torch.zeros(V, Dn3.shape[0], device=DEV)
mh_js = __import__('json').load(open(f'{QK}/qk_minimal_heads.json'))
DET_TOKENS = set()
for a in [mh_js['h3']['arch'][r] for r in DET_RS]:
    for trow in a['tok'][:8]:
        pass
# token ids via encoding: use loadings from saved codes instead — rebuild codes quickly
import torch.nn.functional as F2
Yh3 = torch.cat([TAB['k1'][:, H], TAB['k2'][:, H]], 1)
# DET tokens from archetype dumps (strings) -> ids via tokenizer round trip
det_strs = set()
for r in DET_RS:
    for trow in mh_js['h3']['arch'][r]['tok'][:8]:
        det_strs.add(trow[0])
DET_MASK_V = torch.zeros(V, dtype=torch.bool)
n_found = 0
for t in range(V):
    if tok.decode([t]).replace('\n', '\\n') in det_strs:
        DET_MASK_V[t] = True
        n_found += 1
print(f'DET token variants found: {n_found}; classes: {sorted(det_strs)}', flush=True)


def tabs_variant(kind):
    tabs = {k: v.clone() for k, v in TAB.items()}
    if kind == 'T0':
        return tabs
    for name, Gd in (('k1', G1), ('k2', G2)):
        col = tabs[name][:, H]
        proj = (col @ Gd) @ Gd.T
        if kind == 'T1':
            tabs[name][:, H] = col - proj
        elif kind == 'T2':
            tabs[name][:, H] = proj
        elif kind == 'T3':
            tabs[name][:, H] = 0
    return tabs


base = per_pos_loss(None)
CUR = SEQS[:, :-1].reshape(-1)
DETPOS = DET_MASK_V[CUR]
out = {'n_det_positions': int(DETPOS.sum())}
print(f'DET positions: {int(DETPOS.sum())} of {len(CUR)}', flush=True)
import torch.nn.functional as F2
Vv3 = None
with torch.no_grad():
    a0m = m.transformer.h[0].attn
    E0 = F.rms_norm(m.transformer.wte.weight.detach().float(), (D,))
    Vv_all = a0m.c_v(E0).view(V, NH, HD)
    Wo_all = a0m.c_proj.weight.detach().float().view(D, NH, HD)

DET_IDS = DET_MASK_V.to(DEV)


@torch.no_grad()
def per_pos_and_fire(tabs, add_bias=None, batch=4):
    """Per-position loss; also per-position det-channel firing f(i) for the EXACT run."""
    outs, fires, dwr = [], [], torch.zeros(D, device=DEV)
    npos = 0
    for i in range(0, len(SEQS), batch):
        b = SEQS[i:i + batch].to(DEV)
        idx = b[:, :-1]

        def patch(li, s1, s2):
            if li != 0 or tabs is None:
                return s1, s2
            n1 = scores_from_factors(tabs['q1'], tabs['k1'], idx, HD)
            n2 = scores_from_factors(tabs['q2'], tabs['k2'], idx, HD)
            return n1.to(s1.dtype), n2.to(s2.dtype)

        # firing + channel write (exact tables) for H1/H2 bookkeeping
        if tabs is None:
            n1 = scores_from_factors(TAB['q1'], TAB['k1'], idx, HD)
            n2 = scores_from_factors(TAB['q2'], TAB['k2'], idx, HD)
            B, T = idx.shape
            mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
            pat3 = (n1[:, H] * n2[:, H]).masked_fill(~mask, 0.0)
            detm = DET_IDS[idx].float()
            fires.append((pat3.abs() * detm[:, None, :]).sum(-1).reshape(-1).cpu())
            vw = Vv_all[:, H][idx]
            contrib = torch.einsum('bqt,btd->bqd', pat3 * detm[:, None, :], vw)
            dwr += torch.einsum('bqd,ed->e', contrib,
                                Wo_all[:, H]).squeeze(-1) if False else                 (contrib.reshape(-1, HD) @ Wo_all[:, H].T).sum(0)
            npos += contrib.reshape(-1, HD).shape[0]
        logits = reference_forward(m, idx, 'bf16',
                                   score_patch=None if tabs is None and add_bias is None
                                   else patch).float()
        ls = F.cross_entropy(logits.reshape(-1, V), b[:, 1:].reshape(-1), reduction='none')
        outs.append(ls.view(b.shape[0], -1).cpu())
    fire = torch.cat(fires) if fires else None
    mean_write = dwr / max(npos, 1) if npos else None
    return torch.cat(outs, 0), fire, mean_write


print('exact pass + firing...', flush=True)
base, FIRE, MEANW = per_pos_and_fire(None)
print(f'mean |det-channel write| = {float(MEANW.norm()):.4f}', flush=True)
T1tabs = tabs_variant('T1')
la, _, _ = per_pos_and_fire(T1tabs)
delta = (la - base).flatten()
out = {}
# H1: damage by firing decile
qs = torch.quantile(FIRE, torch.linspace(0, 1, 11))
dec = []
for k in range(10):
    sel = (FIRE >= qs[k]) & (FIRE <= qs[k + 1])
    dec.append(round(float(delta[sel].mean()), 5))
out['damage_by_firing_decile'] = dec
mono = float(np.corrcoef(np.arange(10), np.array(dec))[0, 1])
out['decile_rank_corr'] = round(mono, 3)
print('damage by det-firing decile:', dec, f'(corr {mono:.2f})', flush=True)
json.dump(out, open(f'{QK}/qk_diffuse_floor.json', 'w'), indent=2)

# H2: bias restoration — add mean write back after block 0 (manual forward)


@torch.no_grad()
def per_pos_bias(tabs, bias, batch=4):
    outs = []
    for i in range(0, len(SEQS), batch):
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
        for li, blk in enumerate(m.transformer.h):
            x = blk.lambdas[0] * x + blk.lambdas[1] * x0
            a = blk.attn
            hcur = F.rms_norm(x, (x.size(-1),))

            def factors(lin, name=None):
                if li == 0 and tabs is not None and name is not None:
                    z = tabs[name][idx].to(hcur.dtype)
                else:
                    z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,))
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
            if li == 0 and bias is not None:
                x = x + bias[None, None, :].to(x.dtype)
            x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
        x = F.rms_norm(x, (x.size(-1),))
        logits = 30 * torch.tanh(m.lm_head(x) / 30)
        ls = F.cross_entropy(logits.float().reshape(-1, V), b[:, 1:].reshape(-1),
                             reduction='none')
        outs.append(ls.view(b.shape[0], -1).cpu())
    return torch.cat(outs, 0)


# note: factors() at li==0 uses the FULL exact tables when tabs given (incl. all heads)
la2 = per_pos_bias(T1tabs, MEANW)
d2 = (la2 - base).flatten()
out['T1_dce'] = round(float(delta.mean()), 5)
out['T1_plus_bias_dce'] = round(float(d2.mean()), 5)
lowfire = FIRE < qs[5]
out['lowfire_damage'] = {'T1': round(float(delta[lowfire].mean()), 5),
                         'T1_bias': round(float(d2[lowfire].mean()), 5)}
print(f"T1 {float(delta.mean()):+.5f} -> T1+mean-write bias {float(d2.mean()):+.5f}; "
      f"low-firing-half damage {out['lowfire_damage']}", flush=True)
json.dump(out, open(f'{QK}/qk_diffuse_floor.json', 'w'), indent=2)
print('DIFFUSE FLOOR DONE', flush=True)
