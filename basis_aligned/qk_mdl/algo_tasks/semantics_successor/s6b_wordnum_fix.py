"""s6b: unconfounded digit vs word-number cross-format imposition (the s6
version imposed the element the context already continued to). Impose numbers
MISMATCHED to the context via Code-B."""
import sys, torch
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/semantics_successor')
from semlib import (HERE, DEV, get_model, get_tok, comma_prompt, tok1,
                    batched_run, save_json)
m, cfg = get_model()
NH, D = cfg['n_head'], cfg['n_embd']; HD = D // NH
tok = get_tok()
W_B = torch.load(f'{HERE}/code_WB.pt')['W_B']
emb = m.transformer.wte.weight.detach().double().cpu()
def codeB(toks):
    X = torch.cat([emb[toks.cpu()], torch.ones(len(toks), 1, dtype=torch.float64)], 1)
    return (X @ W_B).float().to(DEV).view(len(toks), NH, HD)
def custom(elems):
    t = [tok1(elems[0])]
    for e in elems[1:]:
        t += [tok1(','), tok1(' ' + e)]
    return t + [tok1(',')]
out = {}
toks_w = custom(['one', 'two', 'three'])
imps = [' 7', ' seven', ' five', ' 5']
ci = torch.tensor([toks_w] * len(imps), device=DEV)
lp = torch.full((len(imps),), len(toks_w) - 2, device=DEV)
it = torch.tensor([tok1(s) for s in imps], device=DEV)
vB = codeB(it)
lg, _ = batched_run(m, cfg, ci, bs=6, v1_sub={(li, h): (vB[:, h], lp)
                    for li in range(1, 18) for h in range(NH)})
pr = lg[torch.arange(len(imps)), len(toks_w) - 1].argmax(-1)
out['word_ctx'] = {s: tok.decode([p.item()]) for s, p in zip(imps, pr)}
p_d = comma_prompt('digit', 1, 3)
imps2 = [' seven', ' 7', ' five']
ci2 = torch.tensor([p_d['tokens']] * len(imps2), device=DEV)
lp2 = torch.full((len(imps2),), p_d['last_pos'], device=DEV)
vB2 = codeB(torch.tensor([tok1(s) for s in imps2], device=DEV))
lg2, _ = batched_run(m, cfg, ci2, bs=6, v1_sub={(li, h): (vB2[:, h], lp2)
                     for li in range(1, 18) for h in range(NH)})
pr2 = lg2[torch.arange(len(imps2)), p_d['pred_pos']].argmax(-1)
out['digit_ctx'] = {s: tok.decode([p.item()]) for s, p in zip(imps2, pr2)}
print(out); save_json('s6b_wordnum_fix.json', out)
