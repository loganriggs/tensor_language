"""Step 4: DIAL demo — funcword vs punct. At AMBIGUOUS held-back positions (funcword and
punct are the two largest predicted category masses, both >= 0.15), steering with the named
direction should flip the model's category argmax in the named direction. Controls: zero
(alpha=0), random direction at the same alpha, and PLACEBO (a different named direction,
digit). Reports flip rates + decoded examples.
"""
import json, sys
import torch
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/semantics_category')
from common import (FINEWEB, HELD_ROWS, CATNAMES, tok, forward, batches, oom_retry,
                    cat_mass, load_probe, OUT, DEV, D, T_CTX)

P = load_probe()
d_unit = P['d_unit'].to(DEV); r_med = P['r_med']
ALPHA = 1.0 * r_med
g = torch.Generator().manual_seed(321)
rvec = torch.randn(D, generator=g); rvec = (rvec / rvec.norm()).to(DEV)
FW, PU, DG = CATNAMES.index('funcword'), CATNAMES.index('punct'), CATNAMES.index('digit')
ROWS = list(HELD_ROWS)

# baseline masses over held set
base = []
for idx in batches(FINEWEB, ROWS):
    lg, _ = oom_retry(forward, idx[:, :-1])
    base.append(cat_mass(lg).cpu())
base = torch.cat(base)                      # (NR, 127, 6)
top2v, top2i = base.topk(2, dim=-1)
amb = (((top2i[..., 0] == FW) & (top2i[..., 1] == PU)) |
       ((top2i[..., 0] == PU) & (top2i[..., 1] == FW))) & (top2v[..., 1] >= 0.15)
print(f"ambiguous funcword/punct positions: {int(amb.sum())} / {amb.numel()}", flush=True)

def run_masses(dvec, alpha):
    out = []
    edit = None if dvec is None else (lambda x: x + alpha*dvec)
    for idx in batches(FINEWEB, ROWS):
        lg, _ = oom_retry(forward, idx[:, :-1], edit_fn=edit)
        out.append(cat_mass(lg).cpu())
    return torch.cat(out)

conds = {'steer_punct': (d_unit[:, PU], ALPHA), 'steer_funcword': (d_unit[:, FW], ALPHA),
         'zero': (None, 0.0), 'random_dir': (rvec, ALPHA), 'placebo_digit': (d_unit[:, DG], ALPHA)}
res = {'alpha_rel': 1.0, 'n_ambiguous': int(amb.sum()), 'conds': {}}
masses = {}
for name, (dv, al) in conds.items():
    mm = run_masses(dv, al); masses[name] = mm
    b_arg = base[amb].argmax(-1); s_arg = mm[amb].argmax(-1)
    to_punct = float(((b_arg == FW) & (s_arg == PU)).sum() / (b_arg == FW).sum().clamp(min=1))
    to_func = float(((b_arg == PU) & (s_arg == FW)).sum() / (b_arg == PU).sum().clamp(min=1))
    res['conds'][name] = {
        'flip_funcword_to_punct': round(to_punct, 4), 'n_funcword_top': int((b_arg == FW).sum()),
        'flip_punct_to_funcword': round(to_func, 4), 'n_punct_top': int((b_arg == PU).sum()),
        'mean_d_punct_mass_at_amb': round(float((mm[amb][:, PU] - base[amb][:, PU]).mean()), 4),
        'mean_d_funcword_mass_at_amb': round(float((mm[amb][:, FW] - base[amb][:, FW]).mean()), 4)}
    print(name, res['conds'][name], flush=True)

# decoded examples: strongest punct-direction flips (funcword-top -> punct-top)
ex = []
mm = masses['steer_punct']
b_arg = base.argmax(-1)
flip_mask = amb & (b_arg == FW) & (mm.argmax(-1) == PU)
ridx, tidx = flip_mask.nonzero(as_tuple=True)
gain = mm[ridx, tidx, PU] - base[ridx, tidx, PU]
order = gain.argsort(descending=True)[:8]
for o in order:
    r, t = int(ridx[o]), int(tidx[o])
    row = FINEWEB[ROWS[r], :T_CTX]
    ctx = tok.decode(row[max(0, t-11):t+1].tolist())
    ex.append({'context_tail': ctx,
               'base_mass_funcword_punct': [round(float(base[r, t, FW]), 3), round(float(base[r, t, PU]), 3)],
               'steered_mass_funcword_punct': [round(float(mm[r, t, FW]), 3), round(float(mm[r, t, PU]), 3)],
               'actual_next': tok.decode([int(row[t+1])])})
res['examples_funcword_to_punct_under_steer_punct'] = ex
for e in ex:
    print(f"  ...{e['context_tail']!r} | fw/punct {e['base_mass_funcword_punct']} -> "
          f"{e['steered_mass_funcword_punct']} (actual next {e['actual_next']!r})", flush=True)
json.dump(res, open(f'{OUT}/s4_dial.json', 'w'), indent=2)
print("S4 DONE", flush=True)
