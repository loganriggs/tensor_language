"""Generate figures + example tables for qk_mdl/results/*.md.

Assets:
  results/fig_conjunction_causal.png   pos-avg intervention bars (ticks 6/8)
  results/fig_conditioned_G.png        identity diagonal in conditioned G (H0.b1 via L0H1)
  results/conjunction_examples.txt     token -> top matched prev-tokens (decoded, tiny vocab)
  results/fig_tiny_frontier.png        tiny-model per-codebook dCE vs DL (tier 1.1)
  results/vq16_exemplars.txt           546M H3/H6 vq16 cluster exemplars (decoded)
  results/fig_tier2_frontier.png       copied from ../fig_tier2_frontier.png
"""

import json
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, '/workspace/tensor_language')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from palette import INK, SECONDARY, MUTED, GRID, DIVERGING

BASE = '/workspace/tensor_language/basis_aligned/qk_mdl'
RES = f'{BASE}/results'
shutil.copy(f'{BASE}/fig_tier2_frontier.png', f'{RES}/fig_tier2_frontier.png')


def style(ax):
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=SECONDARY, labelsize=9)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7, axis='y')
    ax.set_axisbelow(True)


# ---------------- conjunction causal bars
combos = json.load(open(f'{BASE}/tier12b_combos.json'))
labels = ['posavg H0.b1 (L0H1-fed)', 'posavg H0 both branches',
          'posavg H3 both branches', 'posavg H0.b2 + H3.b1 (both diffuse)',
          'posavg H0.b1 + H3.b2 (both L0H1-fed)', 'posavg all four']
short = ['H0.b1 only', 'H0 both', 'H3 both', 'diffuse pair', 'identity pair', 'all four']
vals = [combos[l] for l in labels]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
colors = ['#3987e5', '#3987e5', '#3987e5', '#2f9e63', '#e34948', '#9a6ae1']
ax.bar(range(len(vals)), vals, color=colors, width=0.62)
for i, v in enumerate(vals):
    ax.text(i, v - 0.02 if v < 0 else v + 0.005, f'{v:+.3f}', ha='center',
            fontsize=9, color=INK, va='top' if v < 0 else 'bottom')
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(short, fontsize=9, color=INK)
ax.axhline(0, color=GRID, lw=1)
ax.set_ylabel('ΔP(copy) (base 0.747)', color=INK)
ax.set_title('Destroying token identity (per-Δ score replacement):\n'
             'single heads are free (redundancy); the cross-head identity pair collapses the circuit',
             fontsize=10.5, color=INK)
style(ax)
fig.tight_layout()
fig.savefig(f'{RES}/fig_conjunction_causal.png', dpi=160)

# ---------------- conditioned-G heatmap + examples (recompute accumulators)
exec(open(f'{BASE}/tier12c_conditioned.py').read().split('for seed in range(NBATCH):')[0])
for seed in range(6):   # 6 batches suffice for the visualization
    accumulate(seed)
S = ((q_cnt >= 3) & (k_cnt >= 3)).nonzero().flatten()
Q = (q_sum[(0, 1)][S] / q_cnt[S, None])
K = (k_sum[(0, 1, 'L0H1')][S] / k_cnt[S, None])
g = torch.Generator(); g.manual_seed(1)
sub = torch.randperm(len(S), generator=g)[:60].sort().values.to(DEV)
G = (Q[sub] @ K[sub].T).cpu().numpy()
fig, ax = plt.subplots(figsize=(5.6, 5.2))
vmax = abs(G).max()
ax.imshow(G, cmap=DIVERGING, vmin=-vmax, vmax=vmax)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel('previous-token identity of key (60 random tokens)', color=INK, fontsize=9)
ax.set_ylabel('query token (same 60)', color=INK, fontsize=9)
ax.set_title('Data-conditioned G: L1H0 branch-1 via L0H1\n'
             '(the identity conjunct — diagonal = token match)', fontsize=10.5, color=INK)
fig.tight_layout()
fig.savefig(f'{RES}/fig_conditioned_G.png', dpi=160)

# example matches, decoded with the tiny-model tokenizer
from tokenizers import Tokenizer
tok = Tokenizer.from_file('/workspace/tensor_language/data_owt/tokenizer.json')
Gfull = Q @ K.T
top3 = Gfull.topk(3, dim=1).indices
g = torch.Generator(); g.manual_seed(2)
rows = torch.randperm(len(S), generator=g)[:18]
lines = ['query token -> top-3 matched prev-key tokens (conditioned G, H0.b1 via L0H1)',
         'match position marked *; hit rate over full covered vocab: 0.444', '']
for r in rows.tolist():
    qt = tok.decode([int(S[r])])
    cands = [tok.decode([int(S[c])]) + ('*' if int(c) == r else '')
             for c in top3[r].tolist()]
    lines.append(f'{qt!r:>18} -> ' + ', '.join(repr(c) for c in cands))
open(f'{RES}/conjunction_examples.txt', 'w').write('\n'.join(lines))
print('conjunction assets done')

# ---------------- tiny-model frontier figure
t1 = json.load(open(f'{BASE}/tier1_mdl_attn2-mix10-seed0.json'))
FULL = t1['full_dl_bits_per_headbranch']
fig, ax = plt.subplots(figsize=(7, 4.6))
fam_color = {'svd': '#3987e5', 'vq': '#e34948', 'band': '#2f9e63'}
for row in t1['rows']:
    for fam, c in fam_color.items():
        pts = sorted([(cc['dl_bits'] / FULL, max(cc['dce'], 1e-4))
                      for cc in row['cands'] if cc['name'].startswith(fam)
                      and not cc['name'].startswith('bandband')])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], '-', color=c, lw=1,
                alpha=0.35)
for fam, c in fam_color.items():   # legend proxies
    ax.plot([], [], '-', color=c, lw=2, label={'svd': 'svd-r', 'vq': 'vq-k',
                                               'band': 'band-m'}[fam])
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('DL ratio (per head-branch)', color=INK)
ax.set_ylabel('ΔCE (nats, clipped at 1e-4)', color=INK)
ax.set_title('Tiny model (attn2-mix10) layer-0: 8 head-branches × 3 codebook families\n'
             'rank (svd) dominates; token clustering (vq) fails — opposite of the 546M',
             fontsize=10.5, color=INK)
ax.legend(frameon=False, fontsize=9)
style(ax)
fig.tight_layout()
fig.savefig(f'{RES}/fig_tiny_frontier.png', dpi=160)
print('tiny frontier done')

# ---------------- 546M vq16 exemplars (H3, H6, both branches)
del model
torch.cuda.empty_cache()
from tier2_model import load_elriggs
from tier2_folding import branch_factors as bf2
m2, cfg2 = load_elriggs('bilin18')
HD2 = cfg2['n_embd'] // cfg2['n_head']
from transformers import AutoTokenizer
gtok = AutoTokenizer.from_pretrained('gpt2')

@torch.no_grad()
def km(X, k, iters=12, seed=0):
    gg = torch.Generator(device='cpu'); gg.manual_seed(seed)
    C = X[torch.randperm(len(X), generator=gg)[:k]].clone()
    for _ in range(iters):
        assign = torch.empty(len(X), dtype=torch.long, device=X.device)
        for i in range(0, len(X), 8192):
            x = X[i:i + 8192]
            assign[i:i + 8192] = ((x ** 2).sum(1, keepdim=True) - 2 * x @ C.T
                                  + (C ** 2).sum(1)[None]).argmin(1)
        Cn = torch.zeros_like(C); cnt = torch.zeros(k, device=X.device)
        Cn.index_add_(0, assign, X); cnt.index_add_(0, assign, torch.ones(len(X), device=X.device))
        C = torch.where((cnt == 0)[:, None], C, Cn / cnt.clamp(min=1)[:, None])
    return C, assign

lines = ['546M bilin18 layer-0 vq16 token classes (nearest-to-centroid exemplars)', '']
for hh in (3, 6):
    for br in (1, 2):
        qh, kh = bf2(m2, br, dtype=torch.float32)
        X = torch.cat([qh[:, hh], kh[:, hh]], 1)
        C, assign = km(X, 16)
        lines.append(f'--- L0H{hh} branch {br} ---')
        for cl in range(16):
            members = (assign == cl).nonzero().flatten()
            if not len(members):
                continue
            d = ((X[members] - C[cl]) ** 2).sum(1)
            ex = members[d.argsort()[:10]].tolist()
            lines.append(f'  c{cl:2d} ({len(members):5d} toks): '
                         + ' | '.join(gtok.decode([t]) for t in ex))
        lines.append('')
open(f'{RES}/vq16_exemplars.txt', 'w').write('\n'.join(lines))
print('exemplars done')
