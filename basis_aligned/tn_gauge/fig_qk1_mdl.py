import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
d = json.load(open('/workspace/tensor_language/basis_aligned/tn_gauge/bilin18_qk1_mdl_frontier.json'))
raw = d['raw_Mbit']
lr = sorted(d['lowrank'].items(), key=lambda kv: int(kv[0]))
pr = sorted(d['prune'].items(), key=lambda kv: -float(kv[0]))
fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.plot([v['Mbit'] for _, v in lr], [v['dce'] for _, v in lr], 'o-', color='#3b7dd8', label='low-rank (generic)')
ax.plot([v['total_Mbit'] for _, v in pr], [v['dce'] for _, v in pr], 's-', color='#c0392b', label='magnitude-prune')
try:
    ms = json.load(open('/workspace/tensor_language/basis_aligned/tn_gauge/bilin18_qk1_msubspace.json'))
    mm = sorted(ms['msub'].items(), key=lambda kv: int(kv[0]))
    rc = sorted(ms['resid_ctrl'].items(), key=lambda kv: int(kv[0]))
    ax.plot([v['Mbit'] for _, v in mm], [v['dce'] for _, v in mm], 'D-', color='#1a9850', lw=2.2, label='M-subspace (interpretive structure)')
    ax.plot([v['Mbit'] for _, v in rc], [v['dce'] for _, v in rc], '^--', color='#999', label='residual-PCA (control)')
except Exception:
    pass
ax.axvline(raw, ls='--', color='gray', lw=1, label=f'raw / regime-1 rotation ({raw:.0f} Mbit, 0 compression)')
ax.axhline(0, color='k', lw=0.6)
ax.set_xlabel('description length (Mbit)'); ax.set_ylabel('ΔCE (nats)')
ax.set_title('Layer-1 QK MDL frontier (bilin18): what future methods must beat\nbaseline CE %.3f; rotation buys 0' % d['baseline_ce'])
ax.set_xscale('log'); ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')
ax.set_ylim(-0.05, 0.6)
plt.tight_layout()
plt.savefig('/workspace/tensor_language/basis_aligned/tn_gauge/fig_qk1_mdl.png', dpi=120)
print('saved fig_qk1_mdl.png')
