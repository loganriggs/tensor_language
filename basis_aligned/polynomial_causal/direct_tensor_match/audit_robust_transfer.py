"""Same-panel comparison with the previous graph and the strengthened baselines."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent

def main():
 new=json.loads((P/'ROBUST_TRANSFER_V1.json').read_text());old=json.loads((P/'GENERATED_RESIDUAL_V1.json').read_text());key=lambda c:tuple(c[k] for k in ('panel','domain','family','cohort'));old={key(c):c for c in old['summaries']};rows=[];counts={mode:{which:dict(absolute=0,covariance_comparisons=0,isotropic_comparisons=0,unique_relative_cells=0,total=0) for which in ('old','new')} for mode in ('boundary','generated')}
 for cell in new['summaries']:
  previous=old[key(cell)]
  for mode in ('boundary','generated'):
   for j in range(4):
    a=previous['candidates']['graph_'+mode]['relative_error'][j];b=cell['candidates']['graph_'+mode]['relative_error'][j];record=dict(**{k:cell[k] for k in ('panel','domain','family','cohort')},interface=mode,component=j+1 if j<3 else 'combined',old_error=a,new_error=b,new_over_old=b/a,baselines={})
    for which,error in [('old',a),('new',b)]:
     tally=counts[mode][which];tally['total']+=1;tally['absolute']+=error>(.20 if cell['family']=='change' else .15);any_failure=False
     for name,value in cell['candidates'].items():
      if name.startswith('graph') or not name.endswith('_'+mode):continue
      baseline=value['relative_error'][j];failure=error>1.1*baseline;record['baselines'].setdefault(name,{})[which+'_ratio']=error/baseline;tally['covariance_comparisons' if name.startswith('calibration') else 'isotropic_comparisons']+=failure;any_failure |= failure
     tally['unique_relative_cells']+=any_failure
    rows.append(record)
 impacted=[r for r in rows if r['component']==3];out=dict(counts=counts,rows=rows,component3_new_over_old_range=[min(r['new_over_old'] for r in impacted),max(r['new_over_old'] for r in impacted)],scope='Both graphs compared against same strengthened baselines on same opened panels. Scalar metrics only; no fresh or endpoint claim.')
 (P/'ROBUST_TRANSFER_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(counts,indent=2));print(out['component3_new_over_old_range']);print(json.dumps([r for r in rows if r['interface']=='generated' and r['component']==3 and r['new_error']>.15],indent=2))
if __name__=='__main__':main()
