"""Decompose subject-verb-agreement in bilin18 (see qk_algoverify_sv_agreement.py).
Behavior: "The {head} to the {attr}" -> " is"/" are"; correct verb agrees with the HEAD
noun (pos 1), not the attractor (pos 4). acc=1.00 incl. 40 incongruent-attractor items.

Verify -> patch -> static-prior -> mechanism. Forward pass copied VERBATIM from
tier2_model.reference_forward / qk_conditional_redirect.py / qk_bracket_patch.py
(bilinear, UNNORMALIZED pat=(s1*s2), v1 layer-0 value cache mixed via a.lamb, 30*tanh).

Two lessons applied:
 (§40 greater-of-two was ~95% a STATIC PRIOR) -> (2) all-attention mean-ablation control:
   how much of the agreement margin survives with NO attention (a verb-number prior).
 (§41 bracket was a v1-router: QK decides WHERE, layer-0 value decides WHAT) -> (4)
   v1-swap of the HEAD noun's layer-0 value between singular/plural: does it FLIP the
   predicted verb number? Which head routes it, and from which position (head vs attractor)?

Prompts length 5: [The, head(1), to(2), the(3), attr(4)]; query = pos 4.
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


# --- per-position means over the full 80-prompt set + query-row patterns ---
ysum = [torch.zeros(T, NH, HD, device=DEV) for _ in range(NL)]
asum = [torch.zeros(T, D, device=DEV) for _ in range(NL)]
psum = [torch.zeros(T, D, device=DEV) for _ in range(NL)]
patQ_all = [[] for _ in range(NL)]
for i in range(0, NP, CHUNK):
    _, acc = forward(IDS[i:i + CHUNK], want_yh=True)
    for li in range(NL):
        ysum[li] += acc['yh'][li]; asum[li] += acc['attn'][li]; psum[li] += acc['mlp'][li]
    _, pq = forward(IDS[i:i + CHUNK], want_patQ=True)
    for li in range(NL):
        patQ_all[li].append(pq[li])
means = {'yh': [t / NP for t in ysum], 'attn': [t / NP for t in asum], 'mlp': [t / NP for t in psum]}
# patQ_all[li] -> (NP,NH,T); normalize each query row to a distribution for routing readout
patQ = [torch.cat(v, 0) for v in patQ_all]   # list len NL of (NP,NH,T)


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


res = {}
Lbase = run()
mg_base, ok_base, _ = metrics(Lbase)
res['baseline'] = cell_summary(mg_base, ok_base)

# --- (1) knockout ranking (rank by drop in incongruent-cell margin: the structural test) ---
records = []
for li in range(NL):
    for kind, arg in [('attn', ('attn', li))] + [('head', ('head', li, h)) for h in range(NH)] + [('mlp', ('mlp', li))]:
        L = run(ablate=arg, means=means)
        mg, ok, _ = metrics(L)
        name = f"L{li}.{'attn' if kind=='attn' else ('mlp' if kind=='mlp' else 'h'+str(arg[2]))}"
        records.append({'comp': name, 'kind': kind, 'li': li,
                        'drop_incongr_margin': round(float(mg_base[incongr].mean() - mg[incongr].mean()), 3),
                        'drop_all_margin': round(float(mg_base.mean() - mg.mean()), 3),
                        'incongr_margin_abl': round(float(mg[incongr].mean()), 3),
                        'incongr_acc_abl': round(float(ok[incongr].mean()), 3),
                        'congr_acc_abl': round(float(ok[congr].mean()), 3)})
records.sort(key=lambda r: -r['drop_incongr_margin'])
res['ranking_top20'] = records[:20]
res['incongr_margin_baseline'] = round(float(mg_base[incongr].mean()), 3)

# --- (2) static-prior control: ALL attention mean-ablated ---
Lall = run(ablate=('allattn',), means=means)
mg_all, ok_all, _ = metrics(Lall)
res['all_attn_ablated'] = cell_summary(mg_all, ok_all)
res['static_prior_fraction_incongr'] = round(float(mg_all[incongr].mean() / mg_base[incongr].mean()), 3)
res['static_prior_fraction_all'] = round(float(mg_all.mean() / mg_base.mean()), 3)

# --- (3) no-v1 (lamb=0) global routing ablation ---
Lnv = run(no_v1=True)
mg_nv, ok_nv, _ = metrics(Lnv)
res['no_v1_lamb0'] = cell_summary(mg_nv, ok_nv)

# --- (4a) router readout: dominant heads' query-row attention to head(1) vs attractor(4) ---
def routing_profile(li, h):
    p = patQ[li][:, h, :]                     # (NP,T) RAW signed unnormalized pat weights (=value weights)
    ab = p.abs()
    share = ab / ab.sum(-1, keepdim=True).clamp_min(1e-12)   # share of |weight| per key position
    argmax_pos = ab.argmax(-1).cpu().numpy()
    return {'mean_raw_weight_pos0-4': [round(float(v), 4) for v in p.mean(0)],
            'mean_absshare_pos0-4': [round(float(v), 3) for v in share.mean(0)],
            'absshare_head_pos1': round(float(share[:, HPOS].mean()), 3),
            'absshare_attr_pos4': round(float(share[:, APOS].mean()), 3),
            'absshare_head_incongr': round(float(share[incongr, HPOS].mean()), 3),
            'absshare_attr_incongr': round(float(share[incongr, APOS].mean()), 3),
            'frac_prompts_argmax_at_head_pos1': round(float((argmax_pos == HPOS).mean()), 3),
            'frac_prompts_argmax_at_attr_pos4': round(float((argmax_pos == APOS).mean()), 3)}
top_heads = [r for r in records if r['kind'] == 'head'][:5]
res['router_readout'] = {r['comp']: routing_profile(r['li'], int(r['comp'].split('h')[-1])) for r in top_heads}

# --- (4b) v1-swap: swap HEAD noun layer-0 value between singular/plural subject ---
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

plural_base = (Lbase[:, ARE] - Lbase[:, IS]).cpu().numpy()   # >0 -> predicts 'are'
# HEAD-v1 swap: donor = head-number-flipped counterpart; donor's head number = 1-head_pl
ph_after, pred_h = swap_eval(partner_head, HPOS, 'head')
donor_head_pl = 1 - head_pl
flip_to_donor_head = (pred_h == donor_head_pl).astype(int)          # verb now follows donor number?
res['v1_swap_HEAD_pos1'] = {
    'desc': "swap head-noun layer-0 value with number-flipped counterpart; expect verb to FLIP",
    'baseline_plural_score_sing_head': round(float(plural_base[head_pl == 0].mean()), 3),
    'baseline_plural_score_plur_head': round(float(plural_base[head_pl == 1].mean()), 3),
    'after_plural_score_sing_head(->plural donor)': round(float(ph_after[head_pl == 0].mean()), 3),
    'after_plural_score_plur_head(->sing donor)': round(float(ph_after[head_pl == 1].mean()), 3),
    'flip_rate_to_donor_number_all': round(float(flip_to_donor_head.mean()), 3),
    'flip_rate_incongr': round(float(flip_to_donor_head[incongr].mean()), 3),
    'flip_rate_congr': round(float(flip_to_donor_head[congr].mean()), 3)}
# ATTRACTOR-v1 swap CONTROL: donor = attractor-number-flipped; verb should NOT flip
pa_after, pred_a = swap_eval(partner_attr, APOS, 'attr')
donor_attr_pl = 1 - attr_pl
still_correct_a = (pred_a == head_pl).astype(int)
res['v1_swap_ATTRACTOR_pos4_control'] = {
    'desc': "swap attractor layer-0 value with number-flipped attractor; expect verb NOT to flip",
    'acc_after (should stay ~1.0)': round(float(still_correct_a.mean()), 3),
    'follows_attr_donor_number_rate (should be ~0)': round(float((pred_a == donor_attr_pl).mean()), 3),
    'mean_abs_plural_score_change': round(float(np.abs(pa_after - plural_base).mean()), 3)}

print(json.dumps(res, indent=2), flush=True)
json.dump(res, open(f'{QK}/qk_svagree_patch.json', 'w'), indent=2)
print('DONE', flush=True)
