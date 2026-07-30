"""Close §42's open thread: WHERE does the number (singular/plural) feature live?

Builds directly on qk_svagree_patch.py. The forward pass, prompts, metric (signed
margin logit[correct verb]-logit[wrong] at the verb position), and v1-swap machinery
(v1_at_pos, swap_eval) are copied VERBATIM below. New this file:
 (A) IDENTITY CONTROL the red-team flagged missing: swap the head noun's layer-0
     value with a DIFFERENT SAME-NUMBER noun's value (sing->sing, plur->plur). A
     valid harness must leave the verb unchanged (acc~1.0, margin~baseline).
 (B) LOCUS test: swap the head noun's value at EACH layer's value stream, and
     patch the head-position RESIDUAL at successive layers, between the sing/plural
     counterpart -> at which layer does the number become swappable/flips the verb?
 (C) REDUNDANCY: swap progressively more dims of the head-position layer-0 value
     and see whether flip-rate rises toward 100% (distributed layer-0) or stays low.

Verdict: mid-stack, distributed-layer-0, or a specific localizable layer.
"""
import json, sys
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoTokenizer
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
NL = len(m.transformer.h)
tok = AutoTokenizer.from_pretrained('gpt2')
IS = tok(' is')['input_ids'][0]; ARE = tok(' are')['input_ids'][0]

# --- exact prompts from qk_algoverify_sv_agreement.py ---
HEADS = [('key', 'keys'), ('door', 'doors'), ('book', 'books'), ('car', 'cars'), ('star', 'stars')]
ATTRS = [('cabinet', 'cabinets'), ('house', 'houses'), ('table', 'tables'), ('garden', 'gardens')]
prompts, meta = [], []   # meta: dict(hi, ai, head_pl, attr_pl)
for hi, (hs, hp) in enumerate(HEADS):
    for ai, (as_, ap) in enumerate(ATTRS):
        for head_pl in (0, 1):
            for attr_pl in (0, 1):
                head = hp if head_pl else hs
                attr = ap if attr_pl else as_
                prompts.append(f"The {head} to the {attr}")
                meta.append(dict(hi=hi, ai=ai, head_pl=head_pl, attr_pl=attr_pl))
IDS = torch.stack([tok(p, return_tensors='pt')['input_ids'][0] for p in prompts]).to(DEV)
NP, T = IDS.shape
assert T == 5, T
HPOS, APOS, QPOS = 1, 4, T - 1
assert APOS == QPOS
CHUNK = 8
head_pl = np.array([mm['head_pl'] for mm in meta])
attr_pl = np.array([mm['attr_pl'] for mm in meta])
congr = head_pl == attr_pl            # congruent cell
incongr = ~congr                      # incongruent-attractor cell (the real test)
# correct/wrong verb id per prompt (ARE if head plural else IS)
corr_id = np.where(head_pl == 1, ARE, IS)
wrong_id = np.where(head_pl == 1, IS, ARE)

# partner index maps (built from meta) for v1-swap
key_full = {(mm['hi'], mm['ai'], mm['head_pl'], mm['attr_pl']): i for i, mm in enumerate(meta)}
partner_head = np.array([key_full[(mm['hi'], mm['ai'], 1 - mm['head_pl'], mm['attr_pl'])] for mm in meta])
partner_attr = np.array([key_full[(mm['hi'], mm['ai'], mm['head_pl'], 1 - mm['attr_pl'])] for mm in meta])
# SAME-NUMBER, DIFFERENT-HEAD partner (identity control): different head noun, same number/attr
partner_head_samenum = np.array([key_full[((mm['hi'] + 1) % len(HEADS), mm['ai'], mm['head_pl'], mm['attr_pl'])] for mm in meta])


@torch.no_grad()
def forward(idx, ablate=None, means=None, no_v1=False, v1_patch=None, want_yh=False, want_patQ=False):
    """ablate: None | ('head',li,h) | ('attn',li) | ('mlp',li) | ('allattn',)
    means: per-layer per-position means 'yh'(T,NH,HD) 'attn'(T,D) 'mlp'(T,D).
    no_v1: force lamb->0.  v1_patch: (pos, donor_v1(B,NH,HD)) replaces v1[:,pos].
    want_yh: accumulate per-layer yh/attn/mlp sums.  want_patQ: return per-layer query-row pat."""
    B, Tt = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(Tt, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(Tt, Tt, device=DEV, dtype=torch.bool))
    acc = {'yh': [], 'attn': [], 'mlp': []} if want_yh else None
    patQ = [] if want_patQ else None
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn; hc = F.rms_norm(x, (D,))
        def qk(l): z = F.rms_norm(l(hc).view(B, Tt, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hc).view(B, Tt, NH, HD)
        if v1 is None:
            v1 = v
            if v1_patch is not None:
                pos, donor = v1_patch; v1 = v1.clone(); v1[:, pos] = donor
        lamb = 0.0 if no_v1 else a.lamb
        v = (1 - lamb) * v + lamb * v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        if want_patQ: patQ.append(pat[:, :, QPOS, :].clone())   # (B,NH,T)
        yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)             # (B,Tt,NH,HD)
        if ablate is not None and ablate[0] == 'head' and ablate[1] == li:
            yh = yh.clone(); yh[:, :, ablate[2]] = means['yh'][li][:, ablate[2]].unsqueeze(0)
        if want_yh: acc['yh'].append(yh.sum(0))
        attn = a.c_proj(yh.reshape(B, Tt, -1))
        if ablate is not None and ablate[0] == 'attn' and ablate[1] == li:
            attn = means['attn'][li].unsqueeze(0).expand(B, -1, -1)
        if ablate is not None and ablate[0] == 'allattn':
            attn = means['attn'][li].unsqueeze(0).expand(B, -1, -1)
        if want_yh: acc['attn'].append(attn.sum(0))
        x = x + attn
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if ablate is not None and ablate[0] == 'mlp' and ablate[1] == li:
            mo = means['mlp'][li].unsqueeze(0).expand(B, -1, -1)
        if want_yh: acc['mlp'].append(mo.sum(0))
        x = x + mo
    logits = 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)
    if want_yh: return logits, acc
    if want_patQ: return logits, patQ
    return logits


def run(ablate=None, means=None, no_v1=False, v1_patch=None):
    out = []
    for i in range(0, NP, CHUNK):
        out.append(forward(IDS[i:i + CHUNK], ablate, means, no_v1, v1_patch)[:, -1].float())
    return torch.cat(out)  # (NP,V)


def metrics(L):
    """signed margin (positive = correct verb) and accuracy. L:(NP,V)."""
    ci = torch.tensor(corr_id, device=DEV); wi = torch.tensor(wrong_id, device=DEV)
    idxr = torch.arange(NP, device=DEV)
    mg = (L[idxr, ci] - L[idxr, wi]).cpu().numpy()
    pred_pl = (L[:, ARE] > L[:, IS]).cpu().numpy().astype(int)
    ok = (pred_pl == head_pl).astype(int)
    return mg, ok, pred_pl


def cell_summary(mg, ok):
    return {'all': {'margin': round(float(mg.mean()), 3), 'acc': round(float(ok.mean()), 3)},
            'congruent': {'margin': round(float(mg[congr].mean()), 3), 'acc': round(float(ok[congr].mean()), 3)},
            'incongruent': {'margin': round(float(mg[incongr].mean()), 3), 'acc': round(float(ok[incongr].mean()), 3)}}


# --- v1-swap machinery copied VERBATIM from qk_svagree_patch.py ---
@torch.no_grad()
def v1_at_pos(idx, pos):
    B, Tt = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x
    blk = m.transformer.h[0]; x = blk.lambdas[0] * x + blk.lambdas[1] * x0
    hc = F.rms_norm(x, (D,))
    v = blk.attn.c_v(hc).view(B, Tt, NH, HD)
    return v[:, pos]   # (B,NH,HD)

def swap_eval(partner_idx, pos, label):
    """for every prompt: patch v1[pos] with the partner prompt's v1[pos]; measure verb flip.
    partner differs only in one token (head number for HPOS, attr number for APOS)."""
    plural_after = np.zeros(NP); pred_after = np.zeros(NP, dtype=int)
    for i in range(0, NP, CHUNK):
        src = IDS[i:i + CHUNK]
        donor_ids = IDS[partner_idx[i:i + CHUNK]]
        dv = v1_at_pos(donor_ids, pos)
        L = forward(src, v1_patch=(pos, dv))[:, -1].float()
        plural_after[i:i + CHUNK] = (L[:, ARE] - L[:, IS]).cpu().numpy()
        pred_after[i:i + CHUNK] = (L[:, ARE] > L[:, IS]).cpu().numpy().astype(int)
    return plural_after, pred_after


# =====================================================================================
# NEW: generalized forward for per-layer value / residual capture and patch.
# Copied from `forward` above with capture/patch insertion points added; the arithmetic
# (rms_norm, rope, s1*s2 unnormalized pattern, lamb value mixing, 30*tanh) is identical.
# =====================================================================================
@torch.no_grad()
def forward_gen(idx, val_patch=None, res_patch=None, cap_val_li=None, cap_res_li=None, v1_patch=None):
    """val_patch: (li,pos,donor(B,NH,HD),dmask|None) -> replace the post-lamb value[:,pos] at layer li.
    res_patch:  (li,pos,donor(B,D))                 -> replace residual x[:,pos] AFTER layer li completes.
    cap_val_li: int -> capture the post-lamb value at that layer (returns cap['val'] (B,T,NH,HD)).
    cap_res_li: int -> capture residual x after that layer   (returns cap['res'] (B,T,D)).
    v1_patch:   (pos,donor(B,NH,HD),dmask|None)     -> replace layer-0 value cache v1[:,pos] (dmask = subset of D dims)."""
    B, Tt = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(Tt, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(Tt, Tt, device=DEV, dtype=torch.bool))
    cap = {}
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn; hc = F.rms_norm(x, (D,))
        def qk(l): z = F.rms_norm(l(hc).view(B, Tt, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hc).view(B, Tt, NH, HD)
        if v1 is None:
            v1 = v
            if v1_patch is not None:
                pos, donor, dmask = v1_patch; v1 = v1.clone()
                if dmask is None:
                    v1[:, pos] = donor
                else:
                    fp = v1[:, pos].reshape(B, -1).clone(); fp[:, dmask] = donor.reshape(B, -1)[:, dmask]
                    v1[:, pos] = fp.view(B, NH, HD)
        lamb = a.lamb
        v = (1 - lamb) * v + lamb * v1.view_as(v)
        if val_patch is not None and val_patch[0] == li:
            _, pos, donor, dmask = val_patch; v = v.clone()
            if dmask is None:
                v[:, pos] = donor
            else:
                fp = v[:, pos].reshape(B, -1).clone(); fp[:, dmask] = donor.reshape(B, -1)[:, dmask]
                v[:, pos] = fp.view(B, NH, HD)
        if cap_val_li is not None and cap_val_li == li:
            cap['val'] = v.clone()
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        attn = a.c_proj(yh.reshape(B, Tt, -1))
        x = x + attn
        mo = blk.mlp(F.rms_norm(x, (D,)))
        x = x + mo
        if res_patch is not None and res_patch[0] == li:
            _, pos, donor = res_patch; x = x.clone(); x[:, pos] = donor
        if cap_res_li is not None and cap_res_li == li:
            cap['res'] = x.clone()
    logits = 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)
    return logits, cap


res = {}
Lbase = run()
mg_base, ok_base, _ = metrics(Lbase)
res['baseline'] = cell_summary(mg_base, ok_base)
plural_base = (Lbase[:, ARE] - Lbase[:, IS]).cpu().numpy()   # >0 -> predicts 'are'
# full number swing in the plural score: plural-head minus singular-head baseline gap
S_swing = float(plural_base[head_pl == 1].mean() - plural_base[head_pl == 0].mean())
res['number_swing_nats'] = round(S_swing, 3)


def swing_fraction(pa):
    """fraction of the full singular<->plural swing that a swap moved the plural score, sign-corrected."""
    delta_sing = (pa[head_pl == 0] - plural_base[head_pl == 0]).mean()   # expect + if flips toward plural donor
    delta_plur = (pa[head_pl == 1] - plural_base[head_pl == 1]).mean()   # expect - if flips toward sing donor
    return round(float((delta_sing - delta_plur) / (2 * S_swing)), 3)


def flip_stats(pa, pred):
    donor_num = 1 - head_pl
    flip = (pred == donor_num).astype(float)      # verb now follows donor number?
    ok = (pred == head_pl).astype(float)          # verb still correct (original number)?
    return {'flip_rate_all': round(float(flip.mean()), 3),
            'flip_rate_incongr': round(float(flip[incongr].mean()), 3),
            'acc_after': round(float(ok.mean()), 3),
            'swing_fraction': swing_fraction(pa),
            'mean_abs_plural_score_change': round(float(np.abs(pa - plural_base).mean()), 3)}


# --- (A) IDENTITY CONTROL: swap head value with a DIFFERENT SAME-NUMBER noun (validate harness) ---
pi_after, pred_i = swap_eval(partner_head_samenum, HPOS, 'identity')
ok_i = (pred_i == head_pl).astype(int)
res['A_identity_control_samenum_head_swap'] = {
    'desc': 'swap head layer-0 value with a DIFFERENT SAME-NUMBER noun (sing->sing, plur->plur); harness valid iff verb UNCHANGED',
    'acc_after (should stay ~1.0)': round(float(ok_i.mean()), 3),
    'acc_after_incongr': round(float(ok_i[incongr].mean()), 3),
    'baseline_acc': round(float(ok_base.mean()), 3),
    'mean_abs_plural_score_change (should be small)': round(float(np.abs(pi_after - plural_base).mean()), 3),
    'baseline_mean_abs_plural_score': round(float(np.abs(plural_base).mean()), 3)}

# --- reference: exact §42 full head layer-0 VALUE-CACHE swap (number-flipped counterpart) ---
ph_after, pred_h = swap_eval(partner_head, HPOS, 'head')
res['ref_v1cache_full_head_swap_(sec42)'] = flip_stats(ph_after, pred_h)


# --- (B1) LOCUS: swap head-noun VALUE at EACH layer's value stream (number-flipped donor) ---
def layer_val_swap(li):
    pa = np.zeros(NP); pred = np.zeros(NP, dtype=int)
    for i in range(0, NP, CHUNK):
        src = IDS[i:i + CHUNK]; don = IDS[partner_head[i:i + CHUNK]]
        _, capd = forward_gen(don, cap_val_li=li)
        dv = capd['val'][:, HPOS]                                  # (B,NH,HD) donor post-lamb value at layer li
        L, _ = forward_gen(src, val_patch=(li, HPOS, dv, None)); L = L[:, -1].float()
        pa[i:i + CHUNK] = (L[:, ARE] - L[:, IS]).cpu().numpy()
        pred[i:i + CHUNK] = (L[:, ARE] > L[:, IS]).cpu().numpy().astype(int)
    return pa, pred

res['B1_per_layer_VALUE_swap_head'] = {}
for li in range(NL):
    pa, pred = layer_val_swap(li)
    res['B1_per_layer_VALUE_swap_head'][f'L{li}'] = flip_stats(pa, pred)


# --- (B2) LOCUS: patch head-position RESIDUAL after successive layers (number-flipped donor) ---
def layer_res_swap(li):
    pa = np.zeros(NP); pred = np.zeros(NP, dtype=int)
    for i in range(0, NP, CHUNK):
        src = IDS[i:i + CHUNK]; don = IDS[partner_head[i:i + CHUNK]]
        _, capd = forward_gen(don, cap_res_li=li)
        dr = capd['res'][:, HPOS]                                  # (B,D) donor residual at HPOS after layer li
        L, _ = forward_gen(src, res_patch=(li, HPOS, dr)); L = L[:, -1].float()
        pa[i:i + CHUNK] = (L[:, ARE] - L[:, IS]).cpu().numpy()
        pred[i:i + CHUNK] = (L[:, ARE] > L[:, IS]).cpu().numpy().astype(int)
    return pa, pred

res['B2_per_layer_RESIDUAL_swap_head'] = {}
for li in range(NL):
    pa, pred = layer_res_swap(li)
    res['B2_per_layer_RESIDUAL_swap_head'][f'L{li}'] = flip_stats(pa, pred)


# --- (C) REDUNDANCY: swap progressively more dims of the head-position layer-0 value cache ---
# rank the D dims by mean |donor - source| difference so the most-changed dims go first
src_v1 = torch.cat([v1_at_pos(IDS[i:i+CHUNK], HPOS) for i in range(0, NP, CHUNK)], 0)      # (NP,NH,HD)
don_v1 = torch.cat([v1_at_pos(IDS[partner_head[i:i+CHUNK]], HPOS) for i in range(0, NP, CHUNK)], 0)
diff = (don_v1 - src_v1).reshape(NP, -1).abs().mean(0)                                      # (D,)
order = torch.argsort(diff, descending=True)                                               # dims most changed first

def dim_swap(k):
    dmask = order[:k]
    pa = np.zeros(NP); pred = np.zeros(NP, dtype=int)
    for i in range(0, NP, CHUNK):
        src = IDS[i:i + CHUNK]
        dv = v1_at_pos(IDS[partner_head[i:i + CHUNK]], HPOS)       # (B,NH,HD) donor layer-0 value cache
        L, _ = forward_gen(src, v1_patch=(HPOS, dv, dmask)); L = L[:, -1].float()
        pa[i:i + CHUNK] = (L[:, ARE] - L[:, IS]).cpu().numpy()
        pred[i:i + CHUNK] = (L[:, ARE] > L[:, IS]).cpu().numpy().astype(int)
    return pa, pred

res['C_redundancy_dimsweep_layer0_value'] = {'desc': 'swap top-|diff| k dims of head layer-0 value cache; flip-rate vs fraction'}
for frac in [0.1, 0.25, 0.5, 0.75, 1.0]:
    k = max(1, int(round(frac * D)))
    pa, pred = dim_swap(k)
    st = flip_stats(pa, pred)
    res['C_redundancy_dimsweep_layer0_value'][f'frac_{frac}_k{k}'] = st

print(json.dumps(res, indent=2), flush=True)
json.dump(res, open(f'{QK}/qk_svagree_locus.json', 'w'), indent=2)
print('DONE', flush=True)
