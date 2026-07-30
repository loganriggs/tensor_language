"""Decompose bracket-TYPE-matching in bilin18 (see qk_algoverify_bracket_match.py).
Behavior: '(' -> ')' 1.00, '[' -> ']' 1.00, but '{' -> ')' 100% (built-in negative control).

Verify -> patch -> minimal -> red-team. Forward pass copied VERBATIM from
tier2_model.reference_forward / qk_conditional_redirect.py (bilinear, UNNORMALIZED
pat=(s1*s2), v1 layer-0 value cache mixed via a.lamb, 30*tanh head).

Parts:
 (1) mean-ablation knockout: per attn-layer / per head / per MLP. Graded metric =
     logit[matching closer] - mean(logit[other two closers]) at final position.
     Rank components by drop in the metric averaged over the WORKING openers '(' '['.
 (2) static-prior control: ALL attention mean-ablated (in-distribution per-position
     mean over the 60-prompt set). Fraction of the closer margin that survives.
 (3) curly-hole: does closer-identity live in the layer-0 value cache v1?
     v1-swap at the opener position (pos 2) between openers; and lamb=0 (no v1
     routing) ablation. Contrast '{' vs '(' '['.
All prompts are length 11, opener at index 2, final query at index 10.
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

def tid(s):
    t = tok(s)['input_ids']; assert len(t) == 1, s; return t[0]

# --- exact prompt construction from qk_algoverify_bracket_match.py ---
rng = np.random.RandomState(0)
NOUNS = ['red', 'blue', 'green', 'dogs', 'cats', 'birds', 'cars', 'trees', 'books', 'stars']
PAIRS = [('(', ')'), ('[', ']'), ('{', '}')]
CLOSERS = {c: tid(c) for _, c in PAIRS}
CLIST = [CLOSERS[c] for _, c in PAIRS]  # [')',']','}'] ids

prompts, op_of, correct_of = [], [], []
for op, cl in PAIRS:
    for i in range(20):
        w = rng.choice(NOUNS, 3, replace=False)
        prompts.append(f"The {w[0]} {op}which was near the {w[1]} and the {w[2]}")
        op_of.append(op); correct_of.append(CLOSERS[cl])
IDS = torch.stack([tok(p, return_tensors='pt')['input_ids'][0] for p in prompts]).to(DEV)
NP, T = IDS.shape
assert T == 11, T
OPPOS, QPOS = 2, T - 1
CHUNK = 8
op_arr = np.array(op_of); correct_arr = np.array(correct_of)
work_mask = (op_arr == '(') | (op_arr == '[')   # openers that work at baseline


@torch.no_grad()
def forward(idx, ablate=None, means=None, no_v1=False, v1_patch=None, want_yh=False):
    """ablate: None | ('head',li,h) | ('attn',li) | ('mlp',li) | ('allattn',)
    means: dict with per-layer 'yh'(T,NH,HD) 'attn'(T,D) 'mlp'(T,D) per-position means.
    no_v1: force lamb->0 (no layer-0 value routing).
    v1_patch: (pos, donor_v1) donor_v1 shape (NH,HD) replaces v1[:,pos].
    want_yh: also accumulate per-layer yh/attn/mlp sums (for mean collection)."""
    B, Tt = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(Tt, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(Tt, Tt, device=DEV, dtype=torch.bool))
    acc = {'yh': [], 'attn': [], 'mlp': []} if want_yh else None
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
        yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)          # (B,Tt,NH,HD)
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
    return (logits, acc) if want_yh else logits


def run(ablate=None, means=None, no_v1=False, v1_patch=None):
    out = []
    for i in range(0, NP, CHUNK):
        out.append(forward(IDS[i:i + CHUNK], ablate, means, no_v1, v1_patch)[:, -1].float())
    return torch.cat(out)  # (NP,V)


# --- collect per-position means over the full 60-prompt set ---
ysum = [torch.zeros(T, NH, HD, device=DEV) for _ in range(NL)]
asum = [torch.zeros(T, D, device=DEV) for _ in range(NL)]
psum = [torch.zeros(T, D, device=DEV) for _ in range(NL)]
for i in range(0, NP, CHUNK):
    _, acc = forward(IDS[i:i + CHUNK], want_yh=True)
    for li in range(NL):
        ysum[li] += acc['yh'][li]; asum[li] += acc['attn'][li]; psum[li] += acc['mlp'][li]
means = {'yh': [t / NP for t in ysum], 'attn': [t / NP for t in asum], 'mlp': [t / NP for t in psum]}


def metrics(L):
    """graded closer metric per prompt + accuracy. L:(NP,V)."""
    lc = L[:, CLIST]                                     # (NP,3) logits for ) ] }
    corr = torch.tensor(correct_arr, device=DEV)
    # per prompt: logit[correct] - mean(logit[other two])
    mg = []
    pred_ok = []
    for i in range(NP):
        cid = int(correct_arr[i])
        others = [c for c in CLIST if c != cid]
        mg.append(float(L[i, cid] - L[i, others].mean()))
        pred_ok.append(int(cid == CLIST[int(lc[i].argmax())]))
    mg = np.array(mg); pred_ok = np.array(pred_ok)
    return mg, pred_ok


Lbase = run()
mg_base, ok_base = metrics(Lbase)


def opener_summary(mg, ok):
    d = {}
    for op in ['(', '[', '{']:
        mk = op_arr == op
        d[op] = {'margin': round(float(mg[mk].mean()), 3), 'acc': round(float(ok[mk].mean()), 3)}
    return d


res = {}
res['baseline'] = opener_summary(mg_base, ok_base)

# --- (1) knockout ranking ---
records = []
for li in range(NL):
    for kind, arg in [('attn', ('attn', li))] + [('head', ('head', li, h)) for h in range(NH)] + [('mlp', ('mlp', li))]:
        L = run(ablate=arg, means=means)
        mg, ok = metrics(L)
        # importance = drop in working-opener margin
        drop_work = float(mg_base[work_mask].mean() - mg[work_mask].mean())
        name = f"L{li}.{'attn' if kind=='attn' else ('mlp' if kind=='mlp' else 'h'+str(arg[2]))}"
        rec = {'comp': name, 'kind': kind, 'li': li,
               'drop_work_margin': round(drop_work, 3),
               'work_margin_abl': round(float(mg[work_mask].mean()), 3),
               'curly_margin_abl': round(float(mg[op_arr == '{'].mean()), 3),
               'work_acc_abl': round(float(ok[work_mask].mean()), 3),
               'curly_acc_abl': round(float(ok[op_arr == '{'].mean()), 3)}
        records.append(rec)
records.sort(key=lambda r: -r['drop_work_margin'])
res['ranking_top20'] = records[:20]
res['work_margin_baseline'] = round(float(mg_base[work_mask].mean()), 3)

# --- (2) static-prior control: all attention ablated ---
Lall = run(ablate=('allattn',), means=means)
mg_all, ok_all = metrics(Lall)
res['all_attn_ablated'] = opener_summary(mg_all, ok_all)
# fraction of working-opener margin surviving without attention
res['static_prior_fraction_work'] = round(float(mg_all[work_mask].mean() / mg_base[work_mask].mean()), 3)
# also: closer logits across openers under ablation (constancy check)
lc_all = Lall[:, CLIST]
res['ablated_closer_logits_std_across_openers'] = round(float(
    torch.stack([lc_all[op_arr == op].mean(0) for op in ['(', '[', '{']]).std(0).mean()), 4)
res['ablated_closer_logit_profile'] = {op: [round(float(v), 3) for v in lc_all[op_arr == op].mean(0)]
                                       for op in ['(', '[', '{']}

# --- (3a) no-v1 (lamb=0) ablation ---
Lnv = run(no_v1=True)
mg_nv, ok_nv = metrics(Lnv)
res['no_v1_lamb0'] = opener_summary(mg_nv, ok_nv)

# --- (3b) v1-swap at opener position between opener types ---
# donor v1 at pos 2 = layer-0 value of the donor opener token (per-prompt).
@torch.no_grad()
def v1_at_pos(idx, pos):
    B, Tt = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x
    blk = m.transformer.h[0]; x = blk.lambdas[0] * x + blk.lambdas[1] * x0
    hc = F.rms_norm(x, (D,))
    v = blk.attn.c_v(hc).view(B, Tt, NH, HD)
    return v[:, pos]  # (B,NH,HD)

# build donor-opener prompt tensors (same nouns, swapped opener) per source group
def swap_run(src_op, donor_op):
    """run src_op prompts but with pos-2 v1 replaced by donor_op's v1."""
    mk = np.where(op_arr == src_op)[0]
    src_ids = IDS[mk]
    # donor: same prompts but opener token swapped to donor_op
    donor_tid = tid(f" {donor_op}")   # ' (' etc, standalone token id at pos 2
    donor_ids = src_ids.clone(); donor_ids[:, OPPOS] = donor_tid
    outs = []
    for i in range(0, len(mk), CHUNK):
        dv = v1_at_pos(donor_ids[i:i+CHUNK], OPPOS)
        outs.append(forward(src_ids[i:i+CHUNK], v1_patch=(OPPOS, dv))[:, -1].float())
    L = torch.cat(outs)
    lc = L[:, CLIST]
    # distribution over the 3 closers (argmax) + logits
    pred = np.array([CLIST[int(lc[j].argmax())] for j in range(len(mk))])
    frac = {c: round(float((pred == CLOSERS[c]).mean()), 3) for c, _ in [(')', 0), (']', 0), ('}', 0)]}
    return {'closer_argmax_frac': frac,
            'mean_closer_logits_)]}': [round(float(v), 3) for v in lc.mean(0)]}

# opener token ids at pos2 confirm standalone
res['v1_swap'] = {
    'curly_gets_square_v1 ({ prompt, [ value)': swap_run('{', '['),
    'curly_gets_paren_v1 ({ prompt, ( value)': swap_run('{', '('),
    'square_gets_curly_v1 ([ prompt, { value)': swap_run('[', '{'),
    'paren_gets_curly_v1 (( prompt, { value)': swap_run('(', '{'),
    'square_gets_square_v1 (control)': swap_run('[', '['),
}

print(json.dumps(res, indent=2), flush=True)
json.dump(res, open(f'{QK}/qk_bracket_patch.json', 'w'), indent=2)
print('DONE', flush=True)
