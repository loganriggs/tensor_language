from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).parent;reports=P.parent/'explanations/for_logan';fig,axes=plt.subplots(1,2,figsize=(11,4.4),sharey=True)
colors={('adam',.01):'#8fa9ce',('adam',.03):'#225ea8',('muon',.01):'#ebb87d',('muon',.03):'#c76b14'}
for ax,version,steps in zip(axes,['V1','V2'],[600,1800]):
 data=json.loads((P/f'TOY_OVERLAP_OPTIMIZER_{version}.json').read_text())
 for summary in data['summaries']:
  opt,rate=summary['optimizer'],summary['rate'];group=[r for r in data['records'] if r['optimizer']==opt and r['rate']==rate];small=[];large=[]
  for case in range(5):
   values=[100*r['regularized_root_error'] for r in group if r['case']==case];small.append(min(values));large.append(max(values))
  offset={('adam',.01):-.12,('adam',.03):-.04,('muon',.01):.04,('muon',.03):.12}[(opt,rate)];x=np.arange(1,6)+offset;color=colors[(opt,rate)];ax.vlines(x,small,large,color=color,alpha=.45,lw=2);ax.plot(x,small,'o-',color=color,label=f'{opt.title()}, lr {rate}',markersize=4,lw=1)
 ax.axhline(1,color='#555555',linestyle='--',lw=1);ax.set_yscale('log');ax.set_xticks(range(1,6));ax.set_xlabel('Planted graph fixture');ax.set_title(f'{steps:,} optimization steps');ax.grid(True,axis='y',alpha=.16);ax.legend(fontsize=8,loc='upper right');ax.set_ylim(.0007,50)
axes[0].set_ylabel('Root regularized coefficient error (%)');fig.suptitle('Longer fitting changes the optimizer verdict',fontsize=14);fig.text(.5,.01,'Dots: best of two random starts. Vertical lines: full two-start range, not confidence intervals. Dashed line: 1% recovery gate.',ha='center',fontsize=8);fig.tight_layout(rect=[0,.06,1,.94]);path=reports/'overlap_optimizer_recovery_2026-09-21.png';fig.savefig(path,dpi=180);print(path)
