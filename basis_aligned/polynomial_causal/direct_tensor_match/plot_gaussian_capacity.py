"""Report figure from fixed result artifacts."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
capacity=json.loads((p/'GAUSSIAN_SOURCE_CAPACITY_V1.json').read_text())['records']
shrink=json.loads((p/'GAUSSIAN_SOURCE_SHRINKAGE_V1.json').read_text())['records']
colors={'gaussian_numerator_variation_error':'tab:blue','train_normalized_error':'tab:orange','opened_distinct_normalized_error':'tab:green'}
labels={'gaussian_numerator_variation_error':'Gaussian numerator','train_normalized_error':'Native training','opened_distinct_normalized_error':'Native evaluation (7 prefixes)'}
fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
a=axes[0]
for key in colors:
 a.plot([r['rank_per_source'] for r in capacity],[100*r[key] for r in capacity],marker='o',color=colors[key],label=labels[key])
a.set(xscale='log',xlabel='Rank per source',ylabel='Relative variation error (%)',title='Capacity changes the metrics differently',ylim=(0,50))
a.set_xticks([8,16,32,64,128],labels=['8','16','32','64','128']);a.legend(fontsize=8);a.grid(alpha=.2)
a=axes[1]
for key in ['train_normalized_error','opened_distinct_normalized_error']:
 a.plot([r['covariance_shrinkage'] for r in shrink],[100*r[key] for r in shrink],marker='o',color=colors[key],label=labels[key])
a.axhline(15,color='grey',ls='--',label='Registered fidelity threshold')
a.set(xlabel='Covariance shrinkage toward isotropic',ylabel='Relative variation error (%)',title='Same rank 64, different input metric',ylim=(0,50))
a.legend(fontsize=8);a.grid(alpha=.2)
fig.savefig(p.parent/'explanations/for_logan/assets/gaussian_capacity_and_metric_2026-09-21_0554.png',dpi=160)
