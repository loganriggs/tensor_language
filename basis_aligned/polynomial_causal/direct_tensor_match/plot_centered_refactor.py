import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
prune=json.load(open(P/'CENTERED_PRODUCT_PRUNE_V1.json'))['records'];refit=json.load(open(P/'CENTERED_CONTINUOUS_REFACTOR_V1.json'))['winners'];bound=json.load(open(P/'CENTERED_QUADRATIC_CAPACITY_V1.json'))['output_rank_retention_upper_bounds']
fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
axes[0].plot([r['width'] for r in prune],[100*r['quadratic_retained_energy'] for r in prune],'o-',label='Delete products; refit output',color='#b55d3b')
axes[0].plot([r['width'] for r in refit],[100*r['retained_quadratic_energy'] for r in refit],'s-',label='Refit input directions too',color='#1d6b98')
axes[0].plot(range(1,9),[100*v for v in bound],'--',label='Output-rank upper bound',color='gray');axes[0].set(xlabel='Quadratic products',ylabel='Quadratic coefficient energy retained (%)',ylim=(0,102));axes[0].legend(fontsize=8)
axes[1].plot([r['width'] for r in prune],[100*r['diagnostics'][1]['full_quartic_error'] for r in prune],'o-',color='#b55d3b')
axes[1].plot([r['width'] for r in refit],[100*r['diagnostics'][1]['full_quartic_error'] for r in refit],'s-',color='#1d6b98')
axes[1].axhline(23.1228871375852,linestyle=':',color='gray',label='Frozen 8-product student');axes[1].set(xlabel='Quadratic products',ylabel='Native quartic prediction error (%)');axes[1].legend(fontsize=8)
for ax in axes:ax.grid(alpha=.2);ax.set_xticks(range(0,9,2))
fig.suptitle('A native graph refactor: changing features beats deleting them',fontsize=12)
fig.savefig(P/'CENTERED_REFACTOR_V1.png',dpi=180);fig.savefig(P/'CENTERED_REFACTOR_V1.pdf');plt.close(fig)
