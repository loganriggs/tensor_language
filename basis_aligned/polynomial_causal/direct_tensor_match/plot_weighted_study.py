"""Static scientific summary; points are configurations, not uncertainty samples."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
P=Path(__file__).resolve().parent
R=P.parent/'explanations/for_logan'
d=json.load(open(P/'NATIVE_WEIGHTED_BANK_V1.json'))
metrics=d['plan']['metrics'];labels=['Isotropic','Centered, floor .01','Centered, floor .1','Second moment, floor .01'];colors=['#405e99','#4a9b7a','#d69831','#b64858'];markers={'adam':'o','muon':'s'}
fig,axes=plt.subplots(1,2,figsize=(11.4,4.4),layout='constrained')
for m,(metric,label,color) in enumerate(zip(metrics,labels,colors)):
 rows=[r for r in d['records'] if r['metric']==metric];best=max(rows,key=lambda r:r['weighted_gain'])
 for r in rows:
  offset=(-.12 if r['optimizer']=='adam' else .12)+(.025 if r['seed'] else -.025)
  axes[0].scatter(m+offset,100*r['empirical_evaluation_error'],color=color,marker=markers[r['optimizer']],s=45,zorder=3)
  axes[1].scatter(100*r['coefficient_error'],100*r['empirical_evaluation_error'],color=color,marker=markers[r['optimizer']],s=45,zorder=3)
 axes[0].scatter(m+(-.12 if best['optimizer']=='adam' else .12)+(.025 if best['seed'] else -.025),100*best['empirical_evaluation_error'],facecolors='none',edgecolors='black',s=125,linewidths=1,zorder=4)
axes[0].set_xticks(range(4),['Isotropic','Centered\nfloor .01','Centered\nfloor .1','Second moment\nfloor .01']);axes[0].set_ylabel('Second-panel relative quartic error (%)');axes[0].set_ylim(0,75);axes[0].set_title('Same architecture and original reader coordinates')
axes[1].set_xlabel('Isotropic coefficient-query error (%)');axes[1].set_ylabel('Second-panel relative quartic error (%)');axes[1].set_title('Empirical improvement trades off global accuracy');axes[1].set_ylim(0,75)
for ax in axes:ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
legend=[Line2D([],[],marker=markers[o],linestyle='none',color='black',label=o.title()) for o in markers]+[Line2D([],[],marker='o',markerfacecolor='none',linestyle='none',color='black',markersize=10,label='Training-selected per metric')]
axes[0].legend(handles=legend,loc='lower left',fontsize=8)
axes[1].legend(handles=[Line2D([],[],marker='o',linestyle='none',color=c,label=l) for c,l in zip(colors,labels)],loc='lower right',fontsize=8)
fig.suptitle('Folded MLP16 → MLP17 quartic: 16 fixed-budget fits',fontsize=13)
fig.savefig(R/'weighted_quartic_metrics_2026-09-20.png',dpi=180);fig.savefig(R/'weighted_quartic_metrics_2026-09-20.pdf');print('saved PNG and PDF; four configurations per metric, no error-bar inference')
