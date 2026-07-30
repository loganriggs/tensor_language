"""s3: TABLE EXTRACTION. For each family, impose element e's CODED payload in a
fixed family context and read the argmax prediction -> table(e). This pulls the
successor function out of the model as an explicit standalone lookup table.

Imposition mechanisms:
  Code-A: W_A emb(e) -> L8 H3+H7 at pred pos (spec channel; partial carrier for
          name families per s1b).
  Code-B: W_B emb(e) -> v1 cache slice at last pos, all layers 1-17 (full
          identity channel).
3 contexts per family; majority vote + per-context tables. Family-end elements
(Sunday / December / z / 9) included as wrap probes.
"""
import sys
from collections import Counter

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/semantics_successor')
from semlib import (HERE, DEV, L_PAY, HEADS, FAMILIES, CYCLIC, get_model, get_tok,
                    comma_prompt, numlist_prompt, tok1, batched_run,
                    coded_payload, save_json, free_gb)

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


CONTEXTS = {
    'weekday': [comma_prompt('weekday', s, 4) for s in (0, 2, 4)],
    'month': [comma_prompt('month', s, 4) for s in (0, 3, 6)],
    'alphabet': [comma_prompt('alphabet', s, 4) for s in (0, 8, 16)],
    'digit': [comma_prompt('digit', s, 4) for s in (0, 3, 5)],
    'numlist': [numlist_prompt(k, w1, w2) for k, w1, w2 in
                [(1, 'dogs', 'cats'), (3, 'books', 'trees'), (5, 'cars', 'fish')]],
}
ELEMS = {fam: list(lst) for fam, lst in FAMILIES.items()}
ELEMS['numlist'] = [str(i) for i in range(1, 10)]


def imp_tok(fam, e):
    return tok1(e if fam == 'numlist' else ' ' + e)


def truth(fam, e):
    """Ground-truth successor STRING (leading space stripped) or None (no succ /
    known no-wrap end)."""
    if fam == 'numlist':
        return str(int(e) + 1) if int(e) < 9 else None
    lst = FAMILIES[fam]
    i = lst.index(e)
    if i + 1 < len(lst):
        return lst[i + 1]
    return None   # family end: wrap probe, scored separately


results = {}
for fam in CONTEXTS:
    elems = ELEMS[fam]
    toks = torch.tensor([imp_tok(fam, e) for e in elems], device=DEV)
    pA = coded_payload(W_A, m, toks)
    vB = coded_v1(toks)
    fam_res = {'elements': elems, 'contexts': [c['text'] for c in CONTEXTS[fam]],
               'tables': {'A': [], 'B': []}}
    for c in CONTEXTS[fam]:
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
            fam_res['tables'][key].append([tok.decode([t]) for t in pr.tolist()])
    # majority-vote table + accuracy
    summary = {}
    for key in ('A', 'B'):
        maj, acc_n, acc_ok, wrap_out = [], 0, 0, {}
        for j, e in enumerate(elems):
            votes = Counter(fam_res['tables'][key][ctx][j] for ctx in range(3))
            top = votes.most_common(1)[0][0]
            maj.append(top)
            t = truth(fam, e)
            if t is None:
                wrap_out[e] = top
            else:
                acc_n += 1
                acc_ok += (top.strip() == t)
        summary[key] = {'majority_table': dict(zip(elems, maj)),
                        'acc': acc_ok / acc_n, 'n_scored': acc_n,
                        'end_element_outputs': wrap_out}
    fam_res['summary'] = summary
    results[fam] = fam_res
    print(f"{fam}: A acc={summary['A']['acc']:.2f} B acc={summary['B']['acc']:.2f} "
          f"ends A={summary['A']['end_element_outputs']} B={summary['B']['end_element_outputs']}",
          flush=True)
    print('  B table:', summary['B']['majority_table'], flush=True)

save_json('s3_table.json', results)
print('done', flush=True)
