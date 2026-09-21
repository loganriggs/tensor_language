"""Signed accounting and diagnostic exact-read replacements; no fitted candidate."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent

def main():
 data=json.loads((P/'FRONTIER_READ_ERROR_V1.json').read_text())
 assert data['predictions']['pred_a_replay'] and data['source_replay']<1e-4
 rows=[]
 for cell in data['summaries']:
  g,b=cell['graph']['gram'],cell['separate']['gram']
  excess=cell['graph']['total_energy']-cell['separate']['total_energy']
  diagonal=[g[i][i]-b[i][i] for i in range(3)]
  cross=[dict(pair=[i,j],excess=2*(g[i][j]-b[i][j])) for i,j in ((0,1),(0,2),(1,2))]
  assert abs(sum(diagonal)+sum(x['excess'] for x in cross)-excess)<1e-8*max(abs(excess),1)
  row=dict(family=cell['family'],cohort=cell['cohort'],total_excess_energy=excess,diagonal_excess=diagonal,cross_excess=cross,candidates={})
  for candidate in ('graph','separate'):
   c=cell[candidate];gram=c['gram'];total=c['total_energy']
   records=[r for r in data['records'] if r['candidate']==candidate and r['family']==cell['family'] and r['cohort']==cell['cohort']]
   n=sum(r['sites'] for r in records);error_sum=sum(sum(r['term_sums']) for r in records)
   row['candidates'][candidate]=dict(
    exact_first_read_error_ratio=(gram[1][1]/total)**.5,
    exact_second_read_error_ratio=(gram[0][0]/total)**.5,
    product_error_norm_ratio=(gram[2][2]/total)**.5,
    signed_cross_energy_fraction=c['cross_energy']/total,
    squared_mean_error_fraction=error_sum**2/n/total,
    note='Ratios compare with this same candidate total error; exact read replacements are diagnostic oracles, not executable compression gains.')
  rows.append(row)
 out=dict(rows=rows,scope='Opened-data exact signed energy accounting; terms and cross terms may exceed total or be negative. No independent validation, refitting, or additive causal attribution.')
 (P/'FRONTIER_READ_ERROR_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n')
 for r in rows:
  if r['cohort']=='spaced_word':print(json.dumps(r))
if __name__=='__main__':main()
