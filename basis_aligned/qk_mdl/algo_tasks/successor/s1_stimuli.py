"""Step 1: build 60 clean/corrupted stimulus pairs across three families and
VERIFY (a) clean successor accuracy, (b) whether on corrupted prompts the model
follows the LAST ELEMENT (successor lookup) or the POSITION (= clean answer)."""
import json
import random
import sys

import torch
from transformers import AutoTokenizer

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/successor')
from successor_lib import (HERE, DEV, FAMILIES, CYCLIC, LAST_POS, PRED_POS,
                           load_model, run)

random.seed(0)
tok = AutoTokenizer.from_pretrained('gpt2')
COMMA = tok(',')['input_ids'][0]


def tid(word, space):
    ids = tok((' ' if space else '') + word)['input_ids']
    assert len(ids) == 1, word
    return ids[0]


def succ_idx(fam_name, i):
    fam = FAMILIES[fam_name]
    if CYCLIC[fam_name]:
        return (i + 1) % len(fam)
    return i + 1 if i + 1 < len(fam) else None


pairs = []
for fam_name, fam in FAMILIES.items():
    n = len(fam)
    # 20 starts, cyclic over what's valid for the family
    if fam_name == 'alphabet':
        starts = [i % 20 for i in range(20)]      # s+3 <= 22, succ always exists
    else:
        starts = [i % n for i in range(20)]
    for j, s in enumerate(starts):
        e = [(s + k) % n for k in range(3)]
        clean_ans_i = succ_idx(fam_name, e[2])
        # corrupted last element: different member, different successor, valid succ,
        # not equal to the earlier elements (avoid pure-repeat/induction confound)
        valid = [c for c in range(n)
                 if c not in e and succ_idx(fam_name, c) is not None
                 and succ_idx(fam_name, c) != clean_ans_i
                 and c != clean_ans_i]
        c = random.choice(valid)
        corr_ans_i = succ_idx(fam_name, c)
        ct = [tid(fam[e[0]], False), COMMA, tid(fam[e[1]], True), COMMA,
              tid(fam[e[2]], True), COMMA]
        xt = list(ct)
        xt[LAST_POS] = tid(fam[c], True)
        pairs.append({
            'family': fam_name, 'start': s,
            'clean_str': f'{fam[e[0]]}, {fam[e[1]]}, {fam[e[2]]},',
            'corr_str': f'{fam[e[0]]}, {fam[e[1]]}, {fam[c]},',
            'clean_tokens': ct, 'corr_tokens': xt,
            'clean_ans': tid(fam[clean_ans_i], True),
            'corr_ans': tid(fam[corr_ans_i], True),
            'clean_ans_str': ' ' + fam[clean_ans_i],
            'corr_ans_str': ' ' + fam[corr_ans_i],
            'split': 'analysis' if j < 15 else 'heldout'})

m, cfg = load_model()

ver = {}
for fam_name in FAMILIES:
    rows = [r for r in pairs if r['family'] == fam_name]
    ci = torch.tensor([r['clean_tokens'] for r in rows], device=DEV)
    xi = torch.tensor([r['corr_tokens'] for r in rows], device=DEV)
    ca = torch.tensor([r['clean_ans'] for r in rows], device=DEV)
    xa = torch.tensor([r['corr_ans'] for r in rows], device=DEV)
    lg_c = torch.cat([run(m, cfg, ci[i:i + 8])[0] for i in range(0, len(ci), 8)])
    lg_x = torch.cat([run(m, cfg, xi[i:i + 8])[0] for i in range(0, len(xi), 8)])
    pc, px = lg_c[:, PRED_POS].argmax(-1), lg_x[:, PRED_POS].argmax(-1)
    n = len(rows)
    marg_c = (lg_c[range(n), PRED_POS, ca] - lg_c[range(n), PRED_POS, xa])
    marg_x = (lg_x[range(n), PRED_POS, ca] - lg_x[range(n), PRED_POS, xa])
    ver[fam_name] = {
        'n': n,
        'clean_acc': (pc == ca).float().mean().item(),
        'corr_follows_last_element': (px == xa).float().mean().item(),
        'corr_follows_position': (px == ca).float().mean().item(),
        'corr_other': (~((px == xa) | (px == ca))).float().mean().item(),
        'mean_margin_clean': marg_c.mean().item(),
        'mean_margin_corr': marg_x.mean().item(),
        'corr_top_preds': [tok.decode([t]) for t in px[:8].tolist()],
    }
    for i, r in enumerate(rows):
        r['clean_pred'] = tok.decode([pc[i].item()])
        r['corr_pred'] = tok.decode([px[i].item()])
        r['margin_clean'] = marg_c[i].item()
        r['margin_corr'] = marg_x[i].item()

print(json.dumps(ver, indent=2))
json.dump({'pairs': pairs, 'verification': ver},
          open(f'{HERE}/stimuli.json', 'w'), indent=1)
print('saved stimuli.json;',
      sum(r['split'] == 'analysis' for r in pairs), 'analysis /',
      sum(r['split'] == 'heldout' for r in pairs), 'heldout')
