"""Step 7 addenda:
(A) Dial demo at alpha=0.5 (collateral dCE ~0.06-0.09 there vs ~0.4-0.6 at alpha=1.0):
    same funcword/punct flip design + controls as s4.
(B) Concentration context for s6(a): the top-5 gain share must be compared to the share of
    BASELINE category mass those same tokens already hold (',' '.' dominate punct naturally).
"""
import json, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/semantics_category')
from common import (FINEWEB, HELD_ROWS, CATNAMES, CATM, tok, forward, batches, oom_retry,
                    cat_mass, load_probe, OUT, DEV, D, V)

P = load_probe(); d_unit = P['d_unit'].to(DEV); r_med = P['r_med']
FW, PU, DG = CATNAMES.index('funcword'), CATNAMES.index('punct'), CATNAMES.index('digit')
ROWS = list(HELD_ROWS)
g = torch.Generator().manual_seed(321)
rvec = torch.randn(D, generator=g); rvec = (rvec / rvec.norm()).to(DEV)

base = []
for idx in batches(FINEWEB, ROWS):
    lg, _ = oom_retry(forward, idx[:, :-1])
    base.append(cat_mass(lg).cpu())
base = torch.cat(base)
top2v, top2i = base.topk(2, dim=-1)
amb = (((top2i[..., 0] == FW) & (top2i[..., 1] == PU)) |
       ((top2i[..., 0] == PU) & (top2i[..., 1] == FW))) & (top2v[..., 1] >= 0.15)

def run_masses(dvec, alpha):
    out = []
    edit = None if dvec is None else (lambda x: x + alpha*dvec)
    for idx in batches(FINEWEB, ROWS):
        lg, _ = oom_retry(forward, idx[:, :-1], edit_fn=edit)
        out.append(cat_mass(lg).cpu())
    return torch.cat(out)

res = {'alpha_rel': 0.5, 'n_ambiguous': int(amb.sum()), 'conds': {}}
ALPHA = 0.5 * r_med
for name, (dv, al) in {'steer_punct': (d_unit[:, PU], ALPHA),
                       'steer_funcword': (d_unit[:, FW], ALPHA),
                       'random_dir': (rvec, ALPHA),
                       'placebo_digit': (d_unit[:, DG], ALPHA)}.items():
    mm = run_masses(dv, al)
    b_arg = base[amb].argmax(-1); s_arg = mm[amb].argmax(-1)
    to_punct = float(((b_arg == FW) & (s_arg == PU)).sum() / (b_arg == FW).sum().clamp(min=1))
    to_func = float(((b_arg == PU) & (s_arg == FW)).sum() / (b_arg == PU).sum().clamp(min=1))
    res['conds'][name] = {
        'flip_funcword_to_punct': round(to_punct, 4), 'flip_punct_to_funcword': round(to_func, 4),
        'mean_d_punct_mass_at_amb': round(float((mm[amb][:, PU] - base[amb][:, PU]).mean()), 4),
        'mean_d_funcword_mass_at_amb': round(float((mm[amb][:, FW] - base[amb][:, FW]).mean()), 4)}
    print(name, res['conds'][name], flush=True)

# (B) baseline concentration within each category, on the same 36-row subset as s6(a)
SUB = ROWS[:36]
acc = torch.zeros(V, dtype=torch.float64); n = 0
for idx in batches(FINEWEB, SUB):
    lg, _ = oom_retry(forward, idx[:, :-1])
    p = F.softmax(lg, -1).double().reshape(-1, V)
    acc += p.sum(0).cpu(); n += p.shape[0]
p0 = acc / n
s6 = json.load(open(f'{OUT}/s6_redteam.json'))
res['baseline_concentration'] = {}
for k in range(6):
    pk = p0.clone(); pk[~CATM[k].cpu()] = 0
    tot = float(pk.sum())
    top5v, top5i = pk.topk(5)
    same5 = s6['concentration_alpha1.0'][CATNAMES[k]]['top5_tokens']
    res['baseline_concentration'][CATNAMES[k]] = {
        'baseline_cat_mass': round(tot, 4),
        'baseline_top5_share': round(float(top5v.sum())/max(tot, 1e-9), 4),
        'baseline_top5_tokens': [tok.decode([int(i)]) for i in top5i],
        'gain_top5_tokens_from_s6': same5}
    print(CATNAMES[k], res['baseline_concentration'][CATNAMES[k]], flush=True)
json.dump(res, open(f'{OUT}/s7_addenda.json', 'w'), indent=2)
print("S7 DONE", flush=True)
