import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
x=json.load(open(P/'QUARTIC_CONTEXT_BASIS_V1.json'))['records'];before=[r['baseline'][1]['error'] for r in x];after=[min(z['refits'][1]['error'] for z in r['runs']) for r in x]
axes[0].scatter(before,after);axes[0].plot([.005,1],[.005,1],'k--',alpha=.4);axes[0].set(xscale='log',yscale='log',xlabel='Before basis search: relative error',ylabel='After basis search: relative error',title='Four shared products, 16 local quartics')
x=json.load(open(P/'NATIVE_METRIC_SWEEP_V1.json'))['records'];colors={'isotropic':'#555555','centered_covariance':'#0072B2','second_moment':'#009E73','noncentral':'#D55E00'}
for metric,col in colors.items():
 rows=[min((r for r in x if r['metric']==metric and r['width']==w),key=lambda r:r['training_relative_error']) for w in [128,512]]
 axes[1].plot([r['isotropic_relative_error'] for r in rows],[r['empirical_evaluation_error'] for r in rows],'-o',color=col,label=metric.replace('_',' '))
 for r in rows:axes[1].annotate(str(r['width']),(r['isotropic_relative_error'],r['empirical_evaluation_error']),fontsize=7,xytext=(3,3),textcoords='offset points')
axes[1].set(xlabel='Isotropic Gaussian relative error',ylabel='Evaluation-panel empirical error',title='Metric tradeoff (lower is better)');axes[1].legend(fontsize=7)
x=json.load(open(P/'NATIVE_CHANNEL_BASELINE_V1.json'))['records'];old=json.load(open(P/'NATIVE_FULL_QUADRATIC_V2.json'))['records'];widths=[128,512,1024]
for label,values in [('Random-start joint fit',[min(r['frobenius_relative_error'] for r in old if r['width']==w and r['objective']=='frobenius') for w in widths]),('Teacher channels + output refit',[next(r['relative_errors']['frobenius'] for r in x if r['width']==w and r['metric']=='frobenius' and r['policy']=='energy') for w in widths])]:axes[2].plot(widths,values,'-o',label=label)
axes[2].set(xlabel='Bilinear channels',ylabel='Coefficient Frobenius relative error',title='Optimization gap at matched width');axes[2].legend(fontsize=7)
for ax in axes:ax.grid(alpha=.2)
fig.savefig(P/'REPLICATION_AND_METRICS_V1.png',dpi=160);fig.savefig(P/'REPLICATION_AND_METRICS_V1.pdf')
