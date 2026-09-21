"""Independent prediction for consistent source generation, using prior signed Grams."""
import json,math
from pathlib import Path
P=Path(__file__).resolve().parent

def main():
 data=json.loads((P/'FRONTIER_READ_ERROR_V1.json').read_text());v=(-1,1,-1);rows=[]
 for r in data['summaries']:
  predicted={}
  for name in ('graph','separate'):
   gram=r[name]['gram'];predicted[name]=sum(v[i]*gram[i][j]*v[j] for i in range(3) for j in range(3))
  rows.append(dict(family=r['family'],cohort=r['cohort'],energy=predicted,ratio=math.sqrt(predicted['graph']/predicted['separate']),original_ratio=math.sqrt(r['graph']['total_energy']/r['separate']['total_energy'])))
 out=dict(rows=rows,derivation='Boundary error terms (t1,t2,t3) become (-t1,t2,-t3) when the first source read also generates its residual contribution. This predicts exact-carry mode3 energies on pooled opened code sites, before adding the small native RMS16 reconstruction discrepancy.',scope='Reanalysis, not independent fresh evidence. New native executor must replay this prediction; allcomponents and otherdomains still need checks.')
 (P/'GENERATED_ERROR_GRAM_PREDICTION_V1.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
