"""s2b: natural-text audit of the coded channel. Paired per-token dCE (with
row-clustered SE) for model-wide versions of the channel edits:

  A_zero / A_scale_s : L8 H3+H7 output zeroed / scaled by s at all positions.
  A_coded_pointer    : L8 H3+H7 output at every position q replaced by
                       W_A emb(token at k*), k* = argmax_k |pattern(q,k)| per
                       head -- H made global: 'the channel transmits the
                       identity of the token it points at'.
  A_coded_restricted : same, only where the pointed token is in the calibrated
                       element vocabulary (elsewhere real output kept).
  B_coded_all        : v1 cache replaced by W_B emb(token) at EVERY position
                       (i.e. layer-0 c_v swapped for the fitted linear code),
                       read by layers 1-17.
  B_coded_restricted : same, only at positions holding calibrated element tokens.
  B_zero_restricted  : v1 zeroed at element-token positions (damage control).

usage: s2b_natural.py cooc|fineweb
  cooc    = exploration (cooc rows 0:2400:20, 120 rows)
  fineweb = HELD-BACK audit (rows 448:600)
"""
import sys

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/semantics_successor')
from semlib import (HERE, DEV, L_PAY, HEADS, get_model, run, retry_oom,
                    load_rows, dce_stats, save_json, free_gb)

which = sys.argv[1] if len(sys.argv) > 1 else 'cooc'
rows = load_rows('cooc', slice(0, 2400, 20)) if which == 'cooc' \
    else load_rows('fineweb', slice(448, 600))
print(f'{which}: {rows.shape[0]} rows, free GPU {free_gb():.1f} GB', flush=True)

m, cfg = get_model()
NH, D = cfg['n_head'], cfg['n_embd']
HD = D // NH

Wd = torch.load(f'{HERE}/code_W.pt')
W_A = Wd['W_full']                                         # (1153, 256) float64
HOLDOUT = set(Wd['holdout_elems'])
WBd = torch.load(f'{HERE}/code_WB.pt')
W_B, cal_toks = WBd['W_B'], WBd['cal_toks']                # (1153, 1152)
W_B_hold = WBd['W_B_hold']
emb_all = m.transformer.wte.weight.detach().double().cpu()
V = emb_all.shape[0]
vocab_mask = torch.zeros(V, dtype=torch.bool)
vocab_mask[torch.tensor(cal_toks)] = True
# holdout element tokens (never seen by W_*_hold fits)
from semlib import get_tok
tokz = get_tok()
hold_tok_ids = []
for e in HOLDOUT:
    for form in (e, ' ' + e):
        ids = tokz(form)['input_ids']
        if len(ids) == 1:
            hold_tok_ids.append(ids[0])
hold_mask = torch.zeros(V, dtype=torch.bool)
hold_mask[torch.tensor(hold_tok_ids)] = True

# full-vocab code tables (float32)
X_all = torch.cat([emb_all, torch.ones(V, 1, dtype=torch.float64)], 1)
tab_A = (X_all @ W_A).float()          # (V, 256)
tab_B = (X_all @ W_B).float()          # (V, 1152)
BS = 4


@torch.no_grad()
def ce_cond(name, mode, s=None):
    outs = []
    for i in range(0, len(rows), BS):
        idx = rows[i:i + BS].to(DEV)
        inp, tgt = idx[:, :-1], idx[:, 1:]
        B, T = inp.shape
        kw = {}
        if mode == 'scale':
            kw = {'head_scale': {(L_PAY, h): s for h in HEADS}}
        elif mode in ('A_coded', 'A_coded_r'):
            _, c = retry_oom(run, m, cfg, inp, want_pat=(L_PAY,), want_head=(L_PAY,))
            pat = c[('pat', L_PAY)]
            yh = c[('h', L_PAY)]
            sub = {}
            for j, h in enumerate(HEADS):
                kstar = pat[:, h].abs().argmax(-1)                     # (B,T)
                ptok = torch.gather(inp, 1, kstar)                     # (B,T)
                vals = tab_A[:, j * HD:(j + 1) * HD].to(DEV)[ptok]     # (B,T,HD)
                if mode == 'A_coded_r':
                    keep = ~vocab_mask.to(DEV)[ptok]                   # (B,T)
                    vals = torch.where(keep[..., None], yh[:, :, h, :], vals)
                sub[(L_PAY, h)] = (vals, None)
            del c, pat, yh
            kw = {'head_sub': sub}
        elif mode in ('B_coded', 'B_coded_r', 'B_zero_r', 'B_hold_r'):
            if mode == 'B_zero_r':
                vals = torch.zeros(B, T, D, device=DEV)
            elif mode == 'B_hold_r':
                X_h = torch.cat([emb_all, torch.ones(V, 1, dtype=torch.float64)], 1) @ W_B_hold
                vals = X_h.float().to(DEV)[inp]
            else:
                vals = tab_B.to(DEV)[inp]                              # (B,T,1152)
            if mode != 'B_coded':
                from semlib import v1_of_tokens
                real = v1_of_tokens(m, cfg, inp.reshape(-1)).view(B, T, D)
                msk = hold_mask if mode == 'B_hold_r' else vocab_mask
                keep = ~msk.to(DEV)[inp]
                vals = torch.where(keep[..., None], real, vals)
            vals = vals.view(B, T, NH, HD)
            kw = {'v1_sub': {(li, h): (vals[:, :, h], None)
                             for li in range(1, 18) for h in range(NH)}}
        lg, _ = retry_oom(run, m, cfg, inp, **kw)
        ls = torch.nn.functional.cross_entropy(
            lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction='none').view(B, T)
        outs.append(ls.float().cpu())
        del lg
        torch.cuda.empty_cache()
    ce = torch.cat(outs, 0)
    print(f'{name}: CE={ce.mean():.4f}', flush=True)
    return ce


base = ce_cond('base', 'base')
res = {'which': which, 'n_rows': len(rows), 'base_ce': base.mean().item()}
for name, mode, s in [('A_zero', 'scale', 0.0), ('A_scale_0.5', 'scale', 0.5),
                      ('A_scale_1.5', 'scale', 1.5), ('A_scale_2.0', 'scale', 2.0),
                      ('A_coded_pointer', 'A_coded', None),
                      ('A_coded_restricted', 'A_coded_r', None),
                      ('B_coded_all', 'B_coded', None),
                      ('B_coded_restricted', 'B_coded_r', None),
                      ('B_zero_restricted', 'B_zero_r', None),
                      ('B_codedhold_restricted', 'B_hold_r', None)]:
    ce = ce_cond(name, mode, s)
    st = dce_stats(base, ce)
    st['ce'] = ce.mean().item()
    res[name] = st
    print(f'   dCE={st["dce"]:+.4f} +- {st["se_row"]:.4f} (row SE)', flush=True)

save_json(f's2b_natural_{which}.json', res)
print('done', flush=True)
