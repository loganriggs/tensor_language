"""Diagnostic cancellation audit; no new behavioral threshold or refitting."""
from pathlib import Path
import json,sys
import torch
P=Path(__file__).resolve().parent
out=P/('REGIONAL_ROUTE_VALUE_ROLE_AUDIT_'+sys.argv[1]+'.json');assert not out.exists()
rows=json.loads((P/('REGIONAL_SOURCE_BLOCK_'+('OOD_V1' if sys.argv[1]=='OOD' else 'V2')+'_ROWS.json')).read_text())['rows']
stem='REGIONAL_PRODUCER_ROUTE_VALUE_'+('OOD_V1' if sys.argv[1]=='OOD' else 'V1')
a=torch.load(P/(stem+'_ARTIFACT.pt'),weights_only=True,map_location='cpu')['arm_margins']
e=a[0,None]-a;interaction=e[3]-e[1]-e[2];cells=[]
for family in sorted(set(r['family'] for r in rows)):
    uk=[i for i,r in enumerate(rows) if r['family']==family and r['cue']=='British'];us=[i+1 for i in uk];allids=uk+us
    full=(e[3,uk,0]-e[3,us,0])/2;values=(e[2,uk,0]-e[2,us,0])/2
    ratios=values/full
    cells.append(dict(family=family,paired_value_fraction_min=float(ratios.min()),paired_value_fraction_max=float(ratios.max()),positive_value_pairs=int((values>0).sum()),pair_count=len(uk),prefix_routing_relative_norm=float(e[1,allids,0].norm()/e[3,allids,0].norm()),prefix_value_error_relative_norm=float((e[2,allids,0]-e[3,allids,0]).norm()/e[3,allids,0].norm()),prefix_interaction_relative_norm=float(interaction[allids,0].norm()/e[3,allids,0].norm())))
result=dict(cells=cells,scope='Post-result diagnostic of cancellation and heterogeneity; original family-level verdicts unchanged. Contrast-specific conditional effects, not full-vocabulary preservation or static-routing adoption.')
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
