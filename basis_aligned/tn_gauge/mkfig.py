import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
d = json.load(open('/workspace/tensor_language/basis_aligned/tn_gauge/toy_code_propagation.json'))
ks = [4, 8, 16, 32, 64]
spec = d['spec']
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
# G1 FVU vs k per bond
for li in range(len(spec)):
    ys = [d['G1_fvu_per_bond'][f'k={k}'][li] for k in ks]
    ax[0].plot(ks, ys, 'o-', label=f'bond {li} ({spec[li]})')
ax[0].set_xscale('log', base=2); ax[0].set_xticks(ks); ax[0].set_xticklabels(ks)
ax[0].set_xlabel('sparsity k (atoms/token)'); ax[0].set_ylabel('FVU')
ax[0].set_title('G1: shared-Φ faithfulness rises with depth'); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
# G2 end-to-end dCE vs k
dce = [d['G2_end_to_end_dce'][f'k={k}'] for k in ks]
ax[1].plot(ks, dce, 's-', color='crimson')
ax[1].axhline(0.05, ls='--', color='gray', lw=1, label='ΔCE=0.05 target')
ax[1].set_xscale('log', base=2); ax[1].set_xticks(ks); ax[1].set_xticklabels(ks)
ax[1].set_xlabel('sparsity k (atoms/token, every bond coded)'); ax[1].set_ylabel('ΔCE (nats)')
ax[1].set_title(f'G2: coding every bond, baseline CE {d["baseline_ce"]}'); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
plt.tight_layout()
plt.savefig('/workspace/tensor_language/basis_aligned/tn_gauge/fig_code_propagation.png', dpi=120)
print('saved fig_code_propagation.png')
