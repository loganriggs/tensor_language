"""TICK 197 (Logan's opening question for layer 1): WHO READS LAYER-0 HEAD 3?

Layer-0 head 3 (determiners/copulas) is ~60% of layer-0's causal load (+0.078 zeroed).
Where does its signal go? Two measurements:

(1) SENSITIVITY: re-estimate the token-conditional mean residual with l0-h3's pattern
zeroed, rebuild the layer-1 tables (same shrinkage recipe), and measure per-l1-head
relative change in its key/value tables, plus per-archetype loading shifts (encode the
h3-less rows with each head's saved Stage-1 SAE, project onto the saved archetypes).
The l1 heads/archetypes that move most are h3's readers — at the pattern-formation
level.

(2) PATH DECOMPOSITION (full-audit causal accounting of h3's +0.078):
    A  = baseline CE (3.07630)
    P  = l1 pattern patched with NORMAL token tables            (+0.027, tick 193)
    Pz = l1 pattern patched with h3-LESS token tables (model otherwise normal):
         isolates the l0h3 -> l1-pattern path alone; (Pz - P) = pattern-path share.
    Bz = l0-h3 zeroed AND l1 pattern patched with NORMAL tables: h3's damage with the
         l1-pattern route shielded; comparing (Bz - P) with the total h3 effect says
         how much of +0.078 flows through l1's pattern vs through values/MLP/deeper.
"""
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, reference_forward, rope_tables, apply_rot
from tier2_folding import scores_from_factors
from transformers import AutoTokenizer

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
N_EST, TAU = 1024, 8.0
tok = AutoTokenizer.from_pretrained('gpt2')

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
QP = (torch.bincount(FINEWEB.flatten(), minlength=V).float() + 0.5).to(DEV)
QP = QP / QP.sum()


@torch.no_grad()
def block1_input(idx, zero_h3=False):
    dt = m.transformer.wte.weight.dtype
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    B, T = idx.shape
    blk = m.transformer.h[0]
    x = blk.lambdas[0] * x + blk.lambdas[1] * x0
    a = blk.attn
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
    if zero_h3:
        s1 = s1.clone()
        s1[:, 3] = 0
    pat = (s1 * s2).masked_fill(~mask, 0.0)
    y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1)
    x = x + a.c_proj(y)
    x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    blk1 = m.transformer.h[1]
    return blk1.lambdas[0] * x + blk1.lambdas[1] * x0


def estimate(zero_h3):
    sum_x = torch.zeros(V, D, device=DEV)
    cnt = torch.zeros(V, device=DEV)
    with torch.no_grad():
        for i in range(0, N_EST, 4):
            b = COOC[i:i + 4].to(DEV)
            idx = b[:, :-1]
            x = block1_input(idx, zero_h3).float().reshape(-1, D)
            ids = idx.reshape(-1)
            sum_x.index_add_(0, ids, x)
            cnt.index_add_(0, ids, torch.ones_like(ids, dtype=torch.float))
    wte = m.transformer.wte.weight.detach().float().to(DEV)
    mean_x = torch.where((cnt > 0)[:, None], sum_x / cnt[:, None].clamp_min(1), wte)
    shr = (cnt / (cnt + TAU))[:, None] * mean_x + (TAU / (cnt + TAU))[:, None] * wte
    del sum_x, mean_x
    return shr, cnt


a0 = m.transformer.h[0].attn
a1 = m.transformer.h[1].attn
wte = m.transformer.wte.weight.detach().float().to(DEV)
Vv0 = a0.c_v(F.rms_norm(wte, (D,))).view(V, NH, HD).float()


def tables_from(mx):
    with torch.no_grad():
        xn = F.rms_norm(mx, (D,))
        T1 = {}
        for name, lin in (('q1', a1.c_q), ('k1', a1.c_k), ('q2', a1.c_q2), ('k2', a1.c_k2)):
            z = lin(xn).view(V, NH, HD).float()
            T1[name] = F.rms_norm(z, (HD,)).contiguous()
        v_new = a1.c_v(xn).view(V, NH, HD).float()
        T1['v'] = ((1 - a1.lamb) * v_new + a1.lamb * Vv0.view_as(v_new)).contiguous()
    return T1


print('estimating normal means...', flush=True)
shr_n, _ = estimate(False)
print('estimating h3-less means...', flush=True)
shr_z, _ = estimate(True)
Tn = tables_from(shr_n)
Tz = tables_from(shr_z)
del shr_n, shr_z
torch.cuda.empty_cache()

# ---- (1) sensitivity per l1 head ----
out = {}
sens = {}
for h in range(NH):
    row = {}
    for name in ('k1', 'k2', 'v'):
        dn = ((Tn[name][:, h] - Tz[name][:, h]) ** 2).sum(1)
        nn = (Tn[name][:, h] ** 2).sum(1)
        row[name] = round(float(((QP * dn).sum() / (QP * nn).sum().clamp_min(1e-30)) ** 0.5), 4)
    sens[f'h{h}'] = row
    print(f'l1 h{h} sensitivity to l0-h3 (p-weighted rel change): {row}', flush=True)
out['table_sensitivity'] = sens

# ---- archetype-level shifts ----
s1blob = torch.load(f'{QK}/qk_l1_stage1.pt', map_location=DEV)
s23 = torch.load(f'{QK}/qk_l1_stage23.pt', map_location=DEV)
s1js = json.load(open(f'{QK}/qk_l1_stage1.json'))
arch_shift = {}
for h in range(NH):
    mm = int(s1js[f'h{h}']['m'])
    kc = int(s1js[f'h{h}']['k'])
    Dn = s1blob[f'h{h}_Dn'].to(DEV)
    b_ = s1blob[f'h{h}_b'].to(DEV)
    We = s1blob[f'h{h}_We'].to(DEV)
    U = s23[f'h{h}_U'].to(DEV)
    lam = s23[f'h{h}_lam'].to(DEV)

    def loadings(T1):
        Y = torch.cat([T1['k1'][:, h], T1['k2'][:, h], T1['v'][:, h]], 1)
        z = torch.relu((Y - b_) @ We.T)
        vals, idx = z.topk(kc, dim=1)
        S = torch.zeros_like(z).scatter_(1, idx, vals)
        return S @ U                                    # (V, R)

    Ln, Lz = loadings(Tn), loadings(Tz)
    d = ((QP[:, None] * (Ln - Lz) ** 2).sum(0) /
         (QP[:, None] * Ln ** 2).sum(0).clamp_min(1e-30)) ** 0.5
    r = int(d.argmax())
    top = (QP * (Ln[:, r] - Lz[:, r]).abs()).argsort(descending=True)[:6]
    arch_shift[f'h{h}'] = {
        'max_shift': round(float(d.max()), 4), 'argmax_r': r,
        'mean_shift': round(float(d.mean()), 4),
        'moved_tokens': [tok.decode([t]).replace('\n', '\\n') for t in top.tolist()]}
    print(f'l1 h{h} archetype shifts: mean {float(d.mean()):.4f} max {float(d.max()):.4f} '
          f'(r{r}) moved: {arch_shift[f"h{h}"]["moved_tokens"][:4]}', flush=True)
out['archetype_shift'] = arch_shift
json.dump(out, open(f'{QK}/qk_who_reads_h3.json', 'w'), indent=2)

# ---- (2) path-decomposition audits ----
BASE, PORT = 3.07630, 0.02738


@torch.no_grad()
def audit(l1_tabs=None, zero_l0h3=False, batch=4):
    tot, n = 0.0, 0
    for i in range(0, len(FINEWEB), batch):
        b = FINEWEB[i:i + batch].to(DEV)
        idx = b[:, :-1]

        def patch(li, s1, s2):
            if li == 0 and zero_l0h3:
                s1 = s1.clone()
                s1[:, 3] = 0
                return s1, s2
            if li == 1 and l1_tabs is not None:
                n1 = scores_from_factors(l1_tabs['q1'], l1_tabs['k1'], idx, HD)
                n2 = scores_from_factors(l1_tabs['q2'], l1_tabs['k2'], idx, HD)
                return n1.to(s1.dtype), n2.to(s2.dtype)
            return s1, s2

        logits = reference_forward(m, idx, 'bf16', score_patch=patch).float()
        ce = F.cross_entropy(logits.reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


ce_Pz = audit(l1_tabs=Tz)
out['Pz_h3less_l1tables_dce'] = round(ce_Pz - BASE, 5)
out['pattern_path_share_dce'] = round(ce_Pz - BASE - PORT, 5)
print(f'Pz (h3-less l1 tables, model normal): dCE {ce_Pz - BASE:+.5f} '
      f'(pattern-path share vs port {ce_Pz - BASE - PORT:+.5f})', flush=True)
json.dump(out, open(f'{QK}/qk_who_reads_h3.json', 'w'), indent=2)
ce_Bz = audit(l1_tabs=Tn, zero_l0h3=True)
out['Bz_l0h3zero_l1shielded_dce'] = round(ce_Bz - BASE, 5)
print(f'Bz (l0-h3 zeroed, l1 pattern shielded with normal tables): dCE {ce_Bz - BASE:+.5f} '
      f'(vs total h3 effect +0.07795)', flush=True)
json.dump(out, open(f'{QK}/qk_who_reads_h3.json', 'w'), indent=2)
print('WHO READS H3 DONE', flush=True)
