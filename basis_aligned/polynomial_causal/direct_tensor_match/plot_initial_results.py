from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
r=json.loads((p/'NATIVE_QUARTIC_AUDIT_V1.json').read_text());c=json.loads((p/'TOY_COVARIANCE_SWEEP_V1.json').read_text())
fig,ax=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
for kind in ['tree','dag']:
 rows=[x for x in r['optimized'] if x['kind']==kind];ax[0].loglog([x['values'] for x in rows],[100*x['optimized_frobenius_error'] for x in rows],'o-',label='Optimized '+kind)
b=[x for x in r['same_target_spectral'] if x['representative']=='raw'];ax[0].loglog([30*x['rank']+4*x['rank']**2 for x in b],[100*x['error'] for x in b],'s--',label='Spectral tree, symmetric packed storage')
ax[0].axhline(10,color='gray',ls=':');ax[0].axvline(280,color='gray',ls='--');ax[0].set(xlabel='Stored parameters (local program)',ylabel='Symmetric coefficient error (%)',title='Native quartic: one fixed context');ax[0].legend(fontsize=8)
for name in dict.fromkeys(x['case'] for x in c['records']):
 rows=[x for x in c['records'] if x['case']==name and x['objective']=='empirical_covariance_gaussian'];best=min(rows,key=lambda x:x['relative_error']);ax[1].loglog(100*best['relative_error'],100*best['isotropic_gaussian_error'],'o',label=name)
ax[1].plot([.001,200],[.001,200],'k:',alpha=.4);ax[1].set(xlabel='Covariance-weighted error (%)',ylabel='Same student: isotropic error (%)',title='Toy covariance: best weighted-loss fits');ax[1].legend(fontsize=8)
fig.savefig(p/'INITIAL_RESULTS_V1.png',dpi=180);fig.savefig(p/'INITIAL_RESULTS_V1.pdf')
