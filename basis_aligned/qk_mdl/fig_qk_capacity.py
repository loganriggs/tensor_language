"""Figure for tick 181: per-head capacity frontier of the mechanism ledger."""
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
r = json.load(open(f'{QK}/qk_capacity_frontier.json'))
GATE = 0.05
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
cmap = plt.get_cmap('tab10')

ax = axes[0]
for h in range(9):
    pts = r['ladder'].get(f'h{h}_k8', [])
    if pts:
        ax.plot([p['m'] for p in pts], [p['res'] for p in pts], 'o-',
                color=cmap(h), label=f'head {h}', ms=4)
ax.axhline(GATE, color='k', ls='--', lw=1, label='gate 0.05')
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.set_xlabel('dictionary atoms m (k=8)')
ax.set_ylabel('sketched third-moment residual')
ax.set_title('Ladders at 8 features/token')
ax.legend(fontsize=7, ncol=2)

ax = axes[1]
for h in range(9):
    ks, ms = [], []
    for k in (1, 2, 4, 8):
        pts = r['ladder'].get(f'h{h}_k{k}', [])
        p = next((p for p in pts if p['res'] < GATE), None)
        if p:
            ks.append(k)
            ms.append(p['m'])
    if ks:
        ax.plot(ks, ms, 'o-', color=cmap(h), label=f'head {h}', ms=5)
ax.plot([8], [4096], '*', color=cmap(4), ms=14,
        label='head 4 (tick-180 direct)')
ax.set_xscale('log', base=2)
ax.set_yscale('log', base=2)
ax.set_xlabel('features per token k')
ax.set_ylabel('minimal atoms m passing gate')
ax.set_title('Capacity frontier per head')
ax.legend(fontsize=7, ncol=2)

ax = axes[2]
for h in range(9):
    c = r['prune'][f'h{h}']['curve']
    ax.plot([p['n'] for p in c], [max(p['res'], 1e-4) for p in c], 'o-',
            color=cmap(h), label=f'head {h}', ms=4)
ax.axhline(GATE, color='k', ls='--', lw=1)
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.set_xlabel('atoms kept (usage-ranked, no retrain)')
ax.set_ylabel('moment residual')
ax.set_title('Pruning one big dictionary (vs retrain: far worse)')
ax.legend(fontsize=7, ncol=2)

plt.tight_layout()
plt.savefig(f'{QK}/fig_qk_capacity.png', dpi=130)
print('saved fig_qk_capacity.png')
