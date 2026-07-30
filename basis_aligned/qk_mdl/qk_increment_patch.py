"""Decompose numbered-list increment-with-carry in bilin18 (see qk_algoverify_increment_carry.py).
Behavior: "8. x\n9. y\n" -> "10"; 19->20; 29->30; 99->100, reported carry_acc=1.00 (n=20).

CRITICAL LESSON (the whole point): the sibling greater-of-two arc was ~95% NOT computation
-- a static/in-context prior that tracked the few-shot demo answers (§40, corrected 3x). So we
do NOT assume this is a carry ALGORITHM. First establish genuine positional/computation vs
lookup/prior, THEN localize, THEN test held-out generalization. A clean "it's a lookup/prior"
is a valuable result -- apply the same honesty as the greater-of-two deflation.

Forward pass copied VERBATIM from qk_bracket_patch.py / tier2_model.reference_forward
(bilinear, UNNORMALIZED pat=(s1*s2), v1 layer-0 value cache mixed via a.lamb, 30*tanh head).

All prompts share the aligned token structure  [n1, '.', ' w1', '\n', n2, '.', ' w2', '\n'] (T=8;
every list number 0..1000 is a single GPT-2 token, every noun a single token). n2 at pos 4,
final query at pos 7.

Parts:
 (0) reproduce the algoverify baseline (carry_acc, control_acc) with this forward.
 (1) STATIC-PRIOR / lookup control: mean-ablate ALL attention (in-distribution per-position
     mean over the prompt set). Fraction of the carry margin that survives. With attention
     gone the final '\n' cannot read the n2 number, so anything context-dependent (bigram
     lookup OR carry computation) must collapse; a surviving margin would mean "10" is a
     position/prior default.
 (2) INDUCTION / in-context control (the greater-of-two failure mode): "10" never appears in
     context so answer-copying is impossible by construction; test in-context STEP induction
     instead. successor_only (single line, no +1 demonstrated), broken_increment (n1 != n2-1),
     step-2 / step-4 (does it track the demonstrated step or compute successor-of-last?).
 (3) knockout ranking: per attn-layer / per head / per MLP mean-ablation, ranked by drop in the
     carry margin. Localize; is it the L13H8 v1-router, a layer-8 successor payload, or spread?
 (4) HELD-OUT generalization: broad sweep of (n, n+1)->n+2 carry boundaries + within-decade
     controls across the number line, including rare/large boundaries no dedicated table entry
     is plausible for. Generalization -> computation; only-round-boundaries -> lookup table.

Graded carry margin = logit[correct=n2+1] - logit[no_carry_alt = (n2//10)*10] at the final
position (the direct carry-vs-dropped-carry contrast). Also report restricted-argmax accuracy
identical to qk_algoverify_increment_carry.py for comparability.
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

def one_tok(n):
    return len(tok(str(n))['input_ids']) == 1

# --- exact prompt construction from qk_algoverify_increment_carry.py ---
rng = np.random.RandomState(0)
NOUNS = ['red', 'blue', 'green', 'dogs', 'cats', 'birds', 'cars', 'trees', 'books', 'stars']
CARRY = [(8, 9, 10, [9, 11, 1, 8, 20]), (18, 19, 20, [19, 21, 1, 18, 10]),
         (28, 29, 30, [29, 31, 2, 28, 20]), (98, 99, 100, [99, 9, 1, 98, 10])]
CTRL = [(10, 11, 12, [11, 13, 1, 10, 20]), (20, 21, 22, [21, 23, 2, 20, 12]),
        (94, 95, 96, [95, 97, 9, 94, 86])]
REPS = 5  # matches algoverify n=20 carry (4 cases x 5)

# Build prompt records, reproducing the algoverify rng draw order exactly (CARRY then CTRL,
# each case drawing REPS noun-pairs) so the prompts are identical to the verified ones.
prompts, kind_of, n2_of, cor_of, dis_of = [], [], [], [], []
def add_case(n1, n2, cor, dis, kind):
    for i in range(REPS):
        w = rng.choice(NOUNS, 2, replace=False)
        prompts.append(f"{n1}. {w[0]}\n{n2}. {w[1]}\n")
        kind_of.append(kind); n2_of.append(n2); cor_of.append(cor); dis_of.append(dis)
for n1, n2, cor, dis in CARRY: add_case(n1, n2, cor, dis, 'carry')
for n1, n2, cor, dis in CTRL: add_case(n1, n2, cor, dis, 'control')

IDS = torch.stack([tok(p, return_tensors='pt')['input_ids'][0] for p in prompts]).to(DEV)
NP, T = IDS.shape
assert T == 8, (T, prompts[0])
QPOS = T - 1; N2POS = 4
CHUNK = 8
kind_arr = np.array(kind_of); n2_arr = np.array(n2_of)
carry_mask = kind_arr == 'carry'; ctrl_mask = kind_arr == 'control'
# per-prompt token ids
cor_id = np.array([tid(str(c)) for c in cor_of])
no_carry_alt = np.array([tid(str((n2 // 10) * 10)) for n2 in n2_of])  # dropped-carry number


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


def run(ids, ablate=None, means=None, no_v1=False):
    out = []
    for i in range(0, len(ids), CHUNK):
        out.append(forward(ids[i:i + CHUNK], ablate, means, no_v1)[:, -1].float())
    return torch.cat(out)  # (N,V)


# --- collect per-position means over the full prompt set (bracket_patch convention) ---
ysum = [torch.zeros(T, NH, HD, device=DEV) for _ in range(NL)]
asum = [torch.zeros(T, D, device=DEV) for _ in range(NL)]
psum = [torch.zeros(T, D, device=DEV) for _ in range(NL)]
for i in range(0, NP, CHUNK):
    _, acc = forward(IDS[i:i + CHUNK], want_yh=True)
    for li in range(NL):
        ysum[li] += acc['yh'][li]; asum[li] += acc['attn'][li]; psum[li] += acc['mlp'][li]
means = {'yh': [t / NP for t in ysum], 'attn': [t / NP for t in asum], 'mlp': [t / NP for t in psum]}


def carry_margin(L):
    """logit[correct] - logit[no_carry_alt], per prompt (carry-vs-dropped-carry)."""
    corr = L[torch.arange(len(L)), torch.tensor(cor_id, device=DEV)]
    alt = L[torch.arange(len(L)), torch.tensor(no_carry_alt, device=DEV)]
    return (corr - alt).cpu().numpy()


def restricted_acc(L, idx_mask):
    """restricted-argmax accuracy over {correct}+distractors, identical metric to algoverify."""
    ok = []
    for i in np.where(idx_mask)[0]:
        cand = {c: float(L[i, tid(str(c))]) for c in [cor_of[i]] + dis_of[i]}
        ok.append(max(cand, key=cand.get) == cor_of[i])
    return float(np.mean(ok)) if ok else float('nan')


Lbase = run(IDS)
mg_base = carry_margin(Lbase)
res = {}
res['baseline'] = {
    'carry_acc_restricted': round(restricted_acc(Lbase, carry_mask), 3),
    'control_acc_restricted': round(restricted_acc(Lbase, ctrl_mask), 3),
    'carry_margin_mean': round(float(mg_base[carry_mask].mean()), 3),
    'carry_margin_per_case': {f"{n2_of[i]}->{cor_of[i]}": round(float(mg_base[i]), 3)
                              for i in np.where(carry_mask)[0][::REPS]},
}

# --- (1) STATIC-PRIOR control: all attention mean-ablated ---
Lall = run(IDS, ablate=('allattn',), means=means)
mg_all = carry_margin(Lall)
res['static_prior'] = {
    'carry_margin_all_attn_ablated': round(float(mg_all[carry_mask].mean()), 3),
    'static_prior_fraction': round(float(mg_all[carry_mask].mean() / mg_base[carry_mask].mean()), 3),
    'carry_acc_restricted_ablated': round(restricted_acc(Lall, carry_mask), 3),
    'note': 'fraction of carry margin surviving with ALL attention replaced by per-position mean',
}

# --- (2) INDUCTION / in-context step control ---
@torch.no_grad()
def predict_number(prompt, correct, alts):
    """restricted argmax over {correct}+alts at final position; also correct's rank vs alts."""
    idx = tok(prompt, return_tensors='pt')['input_ids'].to(DEV)
    lp = F.log_softmax(forward(idx)[0, -1].float(), -1)
    cand = {c: float(lp[tid(str(c))]) for c in [correct] + alts}
    return max(cand, key=cand.get), {c: round(cand[c], 2) for c in cand}

ind = {}
# successor_only: single line, NO +1 demonstrated. If it still carries -> pure successor, not step-induction.
so_hits, so_cases = 0, {}
for n2 in [9, 19, 29, 49, 99]:
    w = 'cats'
    pred, cd = predict_number(f"{n2}. {w}\n", n2 + 1, [n2, (n2 // 10) * 10, n2 + 2, 1])
    so_hits += (pred == n2 + 1); so_cases[f"{n2}->{n2+1}"] = pred
ind['successor_only_single_line'] = {'acc': round(so_hits / 5, 3), 'preds': so_cases}
# broken_increment: n1 != n2-1 (unrelated first line). Successor-of-last => still n2+1.
bi_hits, bi_cases = 0, {}
for n1, n2 in [(3, 9), (2, 19), (5, 29), (1, 49), (4, 99)]:
    pred, cd = predict_number(f"{n1}. dogs\n{n2}. cats\n", n2 + 1, [n2, (n2 // 10) * 10, n1 + 1, n2 + (n2 - n1)])
    bi_hits += (pred == n2 + 1); bi_cases[f"{n1},{n2}->{n2+1}?"] = pred
ind['broken_increment_n1!=n2-1'] = {'acc': round(bi_hits / 5, 3), 'preds': bi_cases,
    'note': 'successor computes n2+1 regardless of n1; step-induction would track (n2-n1)'}
# step-2 / step-4: demonstrated step != 1. Successor => n2+1; step-induction => n2+step.
step_cases = {}
for step, n1, n2 in [(2, 7, 9), (2, 17, 19), (4, 5, 9), (2, 96, 98)]:
    succ = n2 + 1; stepped = n2 + step
    pred, cd = predict_number(f"{n1}. dogs\n{n2}. cats\n", succ, [succ, stepped, n2, (n2 // 10) * 10])
    step_cases[f"step{step}:{n1},{n2}"] = {'pred': pred, 'successor': succ, 'stepped': stepped,
                                           'logits': cd}
ind['step_gt1'] = step_cases
res['induction_context'] = ind

# --- (3) knockout ranking (drop in carry margin) ---
records = []
for li in range(NL):
    for kind, arg in ([('attn', ('attn', li))]
                      + [('head', ('head', li, h)) for h in range(NH)]
                      + [('mlp', ('mlp', li))]):
        L = run(IDS, ablate=arg, means=means)
        mg = carry_margin(L)
        drop = float(mg_base[carry_mask].mean() - mg[carry_mask].mean())
        name = f"L{li}.{'attn' if kind=='attn' else ('mlp' if kind=='mlp' else 'h'+str(arg[2]))}"
        records.append({'comp': name, 'kind': kind, 'li': li,
                        'drop_carry_margin': round(drop, 3),
                        'carry_margin_abl': round(float(mg[carry_mask].mean()), 3),
                        'carry_acc_abl': round(restricted_acc(L, carry_mask), 3)})
records.sort(key=lambda r: -r['drop_carry_margin'])
res['knockout_top15'] = records[:15]
res['knockout_bottom3'] = records[-3:]

# --- (4) HELD-OUT generalization across the number line ---
@torch.no_grad()
def sweep(pairs, is_carry):
    """pairs of (n1,n2); correct=n2+1. restricted argmax over {correct, n2, dropped/prev, n2+2}."""
    hits, cases, margins = 0, {}, []
    tested = 0
    for n1, n2 in pairs:
        cor = n2 + 1
        alt = (n2 // 10) * 10 if is_carry else n2 - 1
        if not (one_tok(n1) and one_tok(n2) and one_tok(cor) and one_tok(alt) and one_tok(n2 + 2)):
            continue
        tested += 1
        p = f"{n1}. dogs\n{n2}. cats\n"
        idx = tok(p, return_tensors='pt')['input_ids'].to(DEV)
        L = forward(idx)[0, -1].float()
        cand = {c: float(L[tid(str(c))]) for c in [cor, n2, alt, n2 + 2]}
        pred = max(cand, key=cand.get)
        hits += (pred == cor)
        margins.append(float(L[tid(str(cor))] - L[tid(str(alt))]))
        if len(cases) < 12: cases[f"{n2}->{cor}"] = pred
    return {'n': tested, 'acc': round(hits / max(tested, 1), 3),
            'margin_mean': round(float(np.mean(margins)), 3) if margins else None, 'sample_preds': cases}

# Held-out CARRY boundaries (n2 ends in 9), NOT among the 4 canonical algoverify cases (9,19,29,99):
heldout_carry = [(n2 - 1, n2) for n2 in [39, 49, 59, 69, 79, 89, 109, 119, 129, 149, 199,
                                         209, 249, 299, 349, 399, 449, 499, 599, 699, 799, 899, 999]]
# Held-out within-decade CONTROLS (no carry) at matched magnitudes:
heldout_ctrl = [(n2 - 1, n2) for n2 in [34, 42, 57, 63, 76, 88, 105, 112, 137, 168, 203,
                                        246, 321, 385, 442, 517, 638, 742, 861, 953]]
res['heldout_carry'] = sweep(heldout_carry, True)
res['heldout_control'] = sweep(heldout_ctrl, False)
# double-carry (n2 ends in 99): 199->200, 299->300, 499->500, 999->1000
res['heldout_double_carry'] = sweep([(n2 - 1, n2) for n2 in [199, 299, 399, 499, 599, 999]], True)

# --- verdict heuristic ---
sp = res['static_prior']['static_prior_fraction']
ho = res['heldout_carry']['acc']; hd = res['heldout_double_carry']['acc']
succ = ind['successor_only_single_line']['acc']
if sp > 0.6:
    verdict = 'STATIC/POSITIONAL PRIOR (carry margin survives attention ablation)'
elif ho >= 0.8 and hd >= 0.6:
    verdict = 'GENUINE CARRY COMPUTATION (generalizes to held-out + double-carry boundaries)'
elif ho >= 0.8:
    verdict = 'SUCCESSOR COMPUTATION, single-carry (held-out generalizes; double-carry weaker)'
else:
    verdict = 'LOOKUP / SUCCESSOR-TABLE (fails held-out boundaries)'
res['verdict'] = verdict
res['verdict_inputs'] = {'static_prior_fraction': sp, 'heldout_carry_acc': ho,
                         'heldout_double_carry_acc': hd, 'successor_only_acc': succ}

print(json.dumps(res, indent=2), flush=True)
json.dump(res, open(f'{QK}/qk_increment_patch.json', 'w'), indent=2)
print('DONE', flush=True)
