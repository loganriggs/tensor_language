"""Check exact source-branch write composition and nonlinear behavioral interaction."""
from pathlib import Path
import json
import torch
P=Path(__file__).resolve().parent
out=P/'REGIONAL_VALUE_COMPOSITION_V1_RESULT.json';assert not out.exists()
results={}
for suffix,rowname in [('V1','V2'),('OOD_V1','OOD_V1')]:
    a=torch.load(P/f'REGIONAL_PRODUCER_VALUE_STREAM_{suffix}_ARTIFACT.pt',weights_only=True,map_location='cpu');w=a['write_vertices'];m=a['arm_margins'];rows=json.loads((P/f'REGIONAL_SOURCE_BLOCK_{rowname}_ROWS.json').read_text())['rows']
    changes=w-w[:,0,None];residual=changes[:,3]-changes[:,1]-changes[:,2]
    err=float(residual.norm()/changes[:,3].norm());assert err<1e-10
    effects=m[0,None]-m;interaction=effects[3]-effects[1]-effects[2];cells=[]
    for family in sorted(set(r['family'] for r in rows)):
        uk=[i for i,r in enumerate(rows) if r['family']==family and r['cue']=='British'];us=[i+1 for i in uk];ids=uk+us
        paired=(effects[:,uk]-effects[:,us])/2;both=paired[3,:,0];current=paired[1,:,0];first=paired[2,:,0]
        cells.append(dict(family=family,prefix_nonlinear_interaction_relative_norm=float(interaction[ids,0].norm()/effects[3,ids,0].norm()),paired_nonlinear_interaction_relative_norm=float((both-current-first).norm()/both.norm()),first_fraction_min=float((first/both).min()),first_fraction_max=float((first/both).max()),current_positive=int((current>0).sum()),first_positive=int((first>0).sum()),pairs=len(uk)))
    results[suffix]=dict(relative_write_composition_error=err,cells=cells)
result=dict(panels=results,scope='Exact additivity of upstream stream contributions with recipient routing/RMS/downstream ports fixed; nonlinear suffix interaction reported, not assumed absent. These two branches serve one regional behavior, not demonstrated multi-task reuse.')
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
