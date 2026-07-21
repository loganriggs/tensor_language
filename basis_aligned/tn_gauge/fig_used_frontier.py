import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
d = json.load(open('/workspace/tensor_language/basis_aligned/tn_gauge/bilin18_used_frontier.json'))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
for ax, L in zip(axes, [1, 9]):
    u = sorted(d['layers'][str(L)]['used'].items(), key=lambda kv: int(kv[0]))
    g = sorted(d['layers'][str(L)]['lowrank'].items(), key=lambda kv: int(kv[0]))
    ax.plot([v['Mbit'] for _, v in u], [v['dce'] for _, v in u], 'D-', color='#1a9850', lw=2.2, label='used-subspace (held-out)')
    ax.plot([v['Mbit'] for _, v in g], [v['dce'] for _, v in g], 'o-', color='#3b7dd8', label='generic low-rank')
    ax.axvline(d['raw_Mbit'], ls='--', color='gray', lw=1, label=f'raw ({d["raw_Mbit"]:.0f} Mbit)')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xscale('log'); ax.set_xlabel('description length (Mbit)')
    ax.set_title(f'layer {L}  ({"single-source" if L==1 else "distributed"})')
    ax.grid(alpha=0.3, which='both')
    if L == 1:
        ax.set_ylabel('ΔCE on HELD-OUT tokens (nats)'); ax.legend(fontsize=8)
    ax.set_ylim(-0.02, 0.35)
fig.suptitle('Activation-aware used-subspace QK compression dominates generic low-rank (held-out, bilin18)', fontsize=11)
plt.tight_layout()
plt.savefig('/workspace/tensor_language/basis_aligned/tn_gauge/fig_used_frontier.png', dpi=120)
print('saved fig_used_frontier.png')
