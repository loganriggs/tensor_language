from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).parent;m=json.loads((P/'DUAL_GEOMETRY_SOURCE_V1.json').read_text());fig,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
for width,color in [(367,'#3478ad'),(560,'#c65a37')]:
 selected=[next(r for r in m['records'] if r['key']==m['winners'][f'{width}_{a}']) for a in [0,.5,1]]
 for axis,key,label in zip(axes,['native_isotropic_equal_pair_error','calibration_shaped_error','per_mode_errors'],['Native coefficient error (%)','Calibration-shaped coefficient error (%)','Component 3 error on opened states (%)']):
  values=[100*(r[key][2] if key=='per_mode_errors' else r[key]) for r in selected]
  axis.plot([0,.5,1],values,'o-',color=color,label=f'{width+32} products')
  axis.set(xlabel='Native-metric weight α',ylabel=label,xticks=[0,.5,1]);axis.grid(alpha=.2)
axes[2].axhline(8.8052027,color='#444444',linestyle='--',label='Same-storage covariance pair baseline')
axes[0].legend(frameon=False);axes[2].legend(frameon=False,fontsize=8,loc='upper right');fig.suptitle('Two graph widths, two coefficient geometries: selected fits')
fig.savefig(P.parent/'explanations/for_logan/dual_geometry_source_2026-09-21.png',dpi=160)
