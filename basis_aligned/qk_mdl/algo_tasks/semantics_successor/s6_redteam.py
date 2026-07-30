"""s6: RED-TEAM. (a) longer sequences; (b) elements excluded from calibration
(holdout-fit codes); (c) two competing interleaved sequences; (d) the
no-wrap boundary (Sunday); (e) digit vs word-number.
"""
import sys

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/semantics_successor')
from semlib import (HERE, DEV, L_PAY, HEADS, FAMILIES, CYCLIC, get_model, get_tok,
                    comma_prompt, tok1, batched_run, succ, follow_ans_tok,
                    save_json, free_gb, WORDNUM)

torch.manual_seed(0)
print(f'free GPU {free_gb():.1f} GB', flush=True)
m, cfg = get_model()
NH, D = cfg['n_head'], cfg['n_embd']
HD = D // NH
tok = get_tok()

Wd = torch.load(f'{HERE}/code_W.pt')
W_A, W_A_hold = Wd['W_full'], Wd['W_hold']
WBd = torch.load(f'{HERE}/code_WB.pt')
W_B, W_B_hold = WBd['W_B'], WBd['W_B_hold']
emb = m.transformer.wte.weight.detach().double().cpu()


def codeA(W, toks):
    X = torch.cat([emb[toks.cpu()], torch.ones(len(toks), 1, dtype=torch.float64)], 1)
    out = (X @ W).float().to(DEV)
    return {h: out[:, i * HD:(i + 1) * HD] for i, h in enumerate(HEADS)}


def codeB(W, toks):
    X = torch.cat([emb[toks.cpu()], torch.ones(len(toks), 1, dtype=torch.float64)], 1)
    return (X @ W).float().to(DEV).view(len(toks), NH, HD)


def run_sub(ci, pp, lp, pA=None, vB=None):
    kw = {}
    if pA is not None:
        kw['head_sub'] = {(L_PAY, h): (pA[h], pp) for h in HEADS}
    if vB is not None:
        kw['v1_sub'] = {(li, h): (vB[:, h], lp) for li in range(1, 18) for h in range(NH)}
    lg, _ = batched_run(m, cfg, ci, bs=6, **kw)
    return lg[torch.arange(len(ci)), pp.cpu()].argmax(-1)


out = {}

# ---------------------------------------------------- (a) longer sequences ---
res = {}
for length in (6, 7):
    for fam in FAMILIES:
        n_f = len(FAMILIES[fam])
        starts = range(n_f) if CYCLIC[fam] else range(n_f - length)
        ps = [comma_prompt(fam, s, length) for s in starts]
        ps = [p for p in ps if p['ans_tok'] is not None and
              follow_ans_tok(fam, p['last_elem']) is not None]
        if not ps:
            continue
        ci = torch.tensor([p['tokens'] for p in ps], device=DEV)
        pp = torch.tensor([p['pred_pos'] for p in ps], device=DEV)
        lp = torch.tensor([p['last_pos'] for p in ps], device=DEV)
        ans = torch.tensor([p['ans_tok'] for p in ps])
        toks = ci[torch.arange(len(ps)), lp]
        base = run_sub(ci, pp, lp)
        prA = run_sub(ci, pp, lp, pA=codeA(W_A, toks))
        prB = run_sub(ci, pp, lp, vB=codeB(W_B, toks))
        res[f'{fam}_len{length}'] = {
            'n': len(ps), 'base_acc': (base == ans).float().mean().item(),
            'A_self_acc': (prA == ans).float().mean().item(),
            'B_self_acc': (prB == ans).float().mean().item()}
        print(f'len{length} {fam}: base={res[f"{fam}_len{length}"]["base_acc"]:.2f} '
              f'A={res[f"{fam}_len{length}"]["A_self_acc"]:.2f} '
              f'B={res[f"{fam}_len{length}"]["B_self_acc"]:.2f}', flush=True)
out['longer_sequences'] = res

# ------------------------------------- (b) calibration-held-out elements -----
# impose the holdout element (never seen by W_*_hold) in 2 in-family contexts
# whose true last element differs; follow = its successor.
HOLD = [('weekday', 'Thursday'), ('month', 'October'), ('alphabet', 'm'), ('digit', '7')]
res = {}
for fam, e in HOLD:
    lst = FAMILIES[fam]
    n_f = len(lst)
    starts = [s for s in (range(n_f) if CYCLIC[fam] else range(n_f - 3))
              if lst[(s + 3) % n_f if CYCLIC[fam] else s + 3] != e][:4]
    ps = [comma_prompt(fam, s, 4) for s in starts]
    ci = torch.tensor([p['tokens'] for p in ps], device=DEV)
    pp = torch.tensor([p['pred_pos'] for p in ps], device=DEV)
    lp = torch.tensor([p['last_pos'] for p in ps], device=DEV)
    it = torch.full((len(ps),), tok1(' ' + e), device=DEV)
    ft = follow_ans_tok(fam, e)
    r = {}
    for wa, wb, tag in [(W_A_hold, W_B_hold, 'holdfit'), (W_A, W_B, 'fullfit')]:
        prA = run_sub(ci, pp, lp, pA=codeA(wa, it))
        prB = run_sub(ci, pp, lp, vB=codeB(wb, it))
        r[tag] = {'A_follow': (prA == ft).float().mean().item(),
                  'B_follow': (prB == ft).float().mean().item(),
                  'B_outs': [tok.decode([t]) for t in prB.tolist()]}
    res[f'{fam}:{e}'] = r
    print(f'holdout {fam} {e}: holdfit A={r["holdfit"]["A_follow"]:.2f} '
          f'B={r["holdfit"]["B_follow"]:.2f} (fullfit B={r["fullfit"]["B_follow"]:.2f}) '
          f'B outs={r["holdfit"]["B_outs"]}', flush=True)
out['holdout_elements'] = res

# ------------------------------------------- (c) two competing sequences -----
def custom_prompt(elems):
    toks = [tok1(elems[0])]
    for e in elems[1:]:
        toks += [tok1(','), tok1(' ' + e)]
    toks += [tok1(',')]
    return toks


res = []
for elems, imps in [
        (['Monday', 'January', 'Tuesday', 'February'],
         [('weekday', 'Tuesday'), ('month', 'February'), ('weekday', 'Friday')]),
        (['January', 'Monday', 'February', 'Tuesday'],
         [('weekday', 'Tuesday'), ('month', 'February')])]:
    toks = custom_prompt(elems)
    n = 1 + len(imps)
    ci = torch.tensor([toks] * n, device=DEV)
    pp = torch.full((n,), len(toks) - 1, device=DEV)
    lp = torch.full((n,), len(toks) - 2, device=DEV)
    it = torch.tensor([ci[0, -2].item()] + [tok1(' ' + e) for _, e in imps], device=DEV)
    vB = codeB(W_B, it)
    base = run_sub(ci[:1], pp[:1], lp[:1])
    prB = run_sub(ci, pp, lp, vB=vB)
    entry = {'prompt': ', '.join(elems) + ',',
             'baseline_out': tok.decode([base[0].item()]),
             'impositions': []}
    for j, (fam, e) in enumerate(imps):
        o = tok.decode([prB[j + 1].item()])
        entry['impositions'].append({'imposed': e, 'out': o,
                                     'own_succ': ' ' + succ(fam, e)})
    res.append(entry)
    print('competing:', entry, flush=True)
out['competing_sequences'] = res

# ------------------------------------------------- (d) no-wrap boundary ------
# natural Sunday-ending prompt vs Sunday IMPOSED in a mid-week context
p_nat = comma_prompt('weekday', 4, 3)     # Friday, Saturday, Sunday,
ci = torch.tensor([p_nat['tokens']], device=DEV)
pp = torch.tensor([p_nat['pred_pos']], device=DEV)
lp = torch.tensor([p_nat['last_pos']], device=DEV)
base = run_sub(ci, pp, lp)
# impose Sunday identity on the same Sunday-ending prompt (code = what it already is)
it = torch.tensor([tok1(' Sunday')], device=DEV)
prB_same = run_sub(ci, pp, lp, vB=codeB(W_B, it))
# mid-week context, impose Sunday
p_mid = comma_prompt('weekday', 0, 4)     # Monday..Thursday,
ci2 = torch.tensor([p_mid['tokens']], device=DEV)
pp2 = torch.tensor([p_mid['pred_pos']], device=DEV)
lp2 = torch.tensor([p_mid['last_pos']], device=DEV)
prB_mid = run_sub(ci2, pp2, lp2, vB=codeB(W_B, it))
prA_mid = run_sub(ci2, pp2, lp2, pA=codeA(W_A, it))
out['boundary'] = {
    'natural_sunday_prompt': p_nat['text'],
    'natural_out': tok.decode([base[0].item()]),
    'sunday_code_on_sunday_prompt_out': tok.decode([prB_same[0].item()]),
    'sunday_imposed_midweek_B': tok.decode([prB_mid[0].item()]),
    'sunday_imposed_midweek_A': tok.decode([prA_mid[0].item()])}
print('boundary:', out['boundary'], flush=True)

# ------------------------------------------- (e) digit vs word-number --------
wn = WORDNUM[:5]                          # one..five
toks_w = custom_prompt(wn[:3])            # one, two, three,
ci = torch.tensor([toks_w] * 3, device=DEV)
pp = torch.full((3,), len(toks_w) - 1, device=DEV)
lp = torch.full((3,), len(toks_w) - 2, device=DEV)
base = run_sub(ci[:1], pp[:1], lp[:1])
# impose: digit ' 3' code in word context; word ' three' code (extrapolated, not calibrated)
it = torch.tensor([tok1(' three'), tok1(' 3'), tok1(' three')], device=DEV)
prB = run_sub(ci, pp, lp, vB=codeB(W_B, it))
# digit context with word-code imposition
p_d = comma_prompt('digit', 1, 3)         # 1, 2, 3,
ci2 = torch.tensor([p_d['tokens']] * 2, device=DEV)
pp2 = torch.full((2,), p_d['pred_pos'], device=DEV)
lp2 = torch.full((2,), p_d['last_pos'], device=DEV)
it2 = torch.tensor([tok1(' three'), tok1(' 3')], device=DEV)
prB2 = run_sub(ci2, pp2, lp2, vB=codeB(W_B, it2))
out['word_number'] = {
    'word_prompt': 'one, two, three,',
    'word_baseline_out': tok.decode([base[0].item()]),
    'word_ctx_imposed_three_code': tok.decode([prB[0].item()]),
    'word_ctx_imposed_digit3_code': tok.decode([prB[1].item()]),
    'digit_ctx_imposed_word_three': tok.decode([prB2[0].item()]),
    'digit_ctx_imposed_digit3': tok.decode([prB2[1].item()])}
print('word_number:', out['word_number'], flush=True)

save_json('s6_redteam.json', out)
print('done', flush=True)
