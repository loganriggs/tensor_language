"""Descriptive fresh-panel heterogeneity and explicit matched scalar-interchange map."""
import json,torch
from pathlib import Path
P=Path(__file__).resolve().parent
def main():
 rows=json.loads((P/'SCALAR_PRODUCERS_CONTEXT_TRANSFER_V1_ROWS.json').read_text())['rows']
 m=torch.load(P/'SCALAR_PRODUCERS_CONTEXT_TRANSFER_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['margins'][:,:,0];records=[];donors=[]
 for i,r in enumerate(rows):
  s=rows[i^1];assert all(r[k]==s[k] for k in ('family','city_pair','concept','uk_id','us_id'));assert len(r['ids'])==len(s['ids']) and sum(x!=y for x,y in zip(r['ids'],s['ids']))==1
  donors.append(dict(recipient=i,donor=i^1,self_donor=i,desired_margin_sign=-1 if r['cue']=='British' else 1))
 for family in range(2):
  for city in range(2):
   ix=[i for i,r in enumerate(rows) if r['family']==family and r['city_pair']==city];contrast=m[ix][::2]-m[ix][1::2];reduction=contrast[:,0]-contrast[:,3]
   records.append(dict(family=family,city_pair=city,pairs=len(contrast),mean_native_contrast=float(contrast[:,0].mean()),joint_fraction=float(reduction.mean()/contrast[:,0].mean()),positive_reductions=int((reduction>0).sum()),per_concept_fraction=(reduction/contrast[:,0]).tolist()))
 out=P/'SCALAR_PRODUCERS_CONTEXT_AUDIT_V1_RESULT.json';assert not out.exists();out.write_text(json.dumps(dict(records=records,scope='Post-result descriptive subgroup audit, no new preregistered pass claim or filtering.'),indent=2)+'\n')
 (P/'SCALAR_PRODUCERS_INTERCHANGE_V1_DONORS.json').write_text(json.dumps(dict(donors=donors,source_rows='SCALAR_PRODUCERS_CONTEXT_TRANSFER_V1_ROWS.json',scope='All48rows, opposite-city-cue scalar donor with matched length/positions; no score selection.'),indent=2)+'\n')
 print(json.dumps(records))
if __name__=='__main__':main()
