"""s5: CROSS-FAMILY imposition. Impose a weekday payload in a month context and
vice versa (plus digit->weekday, letter->month): does the model output the
IMPOSED element's own-family successor (payload dominates -> pure pointer +
identity-keyed tables) or the CONTEXT's successor (family-tagged channel)?

Imposition: Code-B (full identity channel, v1@last, layers 1-17) primary;
Code-A (L8 H3+H7 @ pred) secondary. Every in-family element with a defined
successor is imposed in 2 contexts of the other family.
Output classes: own_succ (successor of imposed element in ITS family),
context_cont (the context's own next element), imposed_echo (the imposed
element itself), and_tok (' and'), other.
"""
import sys
from collections import Counter

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/semantics_successor')
from semlib import (HERE, DEV, L_PAY, HEADS, FAMILIES, get_model, get_tok,
                    comma_prompt, tok1, batched_run, coded_payload, succ,
                    save_json, free_gb)

torch.manual_seed(0)
print(f'free GPU {free_gb():.1f} GB', flush=True)
m, cfg = get_model()
NH, D = cfg['n_head'], cfg['n_embd']
HD = D // NH
tok = get_tok()

W_A = torch.load(f'{HERE}/code_W.pt')['W_full']
W_B = torch.load(f'{HERE}/code_WB.pt')['W_B']
emb = m.transformer.wte.weight.detach().double().cpu()


def coded_v1(toks):
    X = torch.cat([emb[toks.cpu()], torch.ones(len(toks), 1, dtype=torch.float64)], 1)
    return (X @ W_B).float().to(DEV).view(len(toks), NH, HD)


CROSSES = [('month', 'weekday'), ('weekday', 'month'),
           ('weekday', 'digit'), ('month', 'alphabet')]
CTX_STARTS = {'weekday': (0, 3), 'month': (2, 6), 'alphabet': (4, 10), 'digit': (2, 4)}

out = {}
for ctx_fam, pay_fam in CROSSES:
    # imposed elements: all with a defined non-wrap successor in their own family
    lst = FAMILIES[pay_fam]
    elems = [e for e in lst if lst.index(e) + 1 < len(lst)]
    toks = torch.tensor([tok1(' ' + e) for e in elems], device=DEV)
    pA = coded_payload(W_A, m, toks)
    vB = coded_v1(toks)
    contexts = [comma_prompt(ctx_fam, s, 4) for s in CTX_STARTS[ctx_fam]]
    rows = []
    for c in contexts:
        n = len(elems)
        ci = torch.tensor([c['tokens']] * n, device=DEV)
        pp = torch.full((n,), c['pred_pos'], device=DEV)
        lp = torch.full((n,), c['last_pos'], device=DEV)
        lgA, _ = batched_run(m, cfg, ci, bs=6,
                             head_sub={(L_PAY, h): (pA[h], pp) for h in HEADS})
        lgB, _ = batched_run(m, cfg, ci, bs=6,
                             v1_sub={(li, h): (vB[:, h], lp)
                                     for li in range(1, 18) for h in range(NH)})
        for key, lg in (('A', lgA), ('B', lgB)):
            pr = lg[torch.arange(n), c['pred_pos']].argmax(-1)
            for j, e in enumerate(elems):
                o = tok.decode([pr[j].item()])
                own = ' ' + succ(pay_fam, e)
                ctx_next = ' ' + c['succ_elem'] if c['succ_elem'] else None
                cls = ('own_succ' if o == own else
                       'context_cont' if o == ctx_next else
                       'imposed_echo' if o.strip() == e else
                       'and_tok' if o == ' and' else 'other')
                rows.append({'code': key, 'context': c['text'], 'imposed': e,
                             'out': o, 'class': cls})
    res = {}
    for key in ('A', 'B'):
        cnt = Counter(r['class'] for r in rows if r['code'] == key)
        tot = sum(cnt.values())
        res[key] = {k: v / tot for k, v in cnt.items()}
        res[key]['n'] = tot
    out[f'{pay_fam}_payload_in_{ctx_fam}_context'] = {'classes': res, 'rows': rows}
    print(f'{pay_fam} payload in {ctx_fam} ctx: '
          f'A={ {k: round(v, 2) for k, v in res["A"].items()} } '
          f'B={ {k: round(v, 2) for k, v in res["B"].items()} }', flush=True)

save_json('s5_crossfam.json', out)
print('done', flush=True)
