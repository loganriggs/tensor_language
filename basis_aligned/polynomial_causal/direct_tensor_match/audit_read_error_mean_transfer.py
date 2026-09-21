"""Descriptive cross-panel mean-error transfer, no changes to frozen programs."""
import json,math
from pathlib import Path
P=Path(__file__).resolve().parent

def stats(rows):
 return dict(n=sum(r['sites'] for r in rows),e=sum(r['error_energy'] for r in rows),s=sum(sum(r['term_sums']) for r in rows))

def main():
 data=json.loads((P/'FRONTIER_READ_ERROR_V1.json').read_text());out=[]
 for train,test in ((1,2),(2,1)):
  for training_cohort in ('all','spaced_word'):
   for family in ('natural','hybrid','change'):
    for cohort in ('all','continuation','spaced_word'):
     result=dict(train_panel=train,test_panel=test,training_cohort=training_cohort,family=family,cohort=cohort,candidates={})
     for candidate in ('graph','separate'):
      fit=stats([r for r in data['records'] if r['panel']==train and r['family']=='natural' and r['cohort']==training_cohort and r['candidate']==candidate])
      test_stats=stats([r for r in data['records'] if r['panel']==test and r['family']==family and r['cohort']==cohort and r['candidate']==candidate])
      offset=fit['s']/fit['n'] if family!='change' else 0
      energy=test_stats['e']-2*offset*test_stats['s']+test_stats['n']*offset**2
      assert energy>=0
      result['candidates'][candidate]=dict(offset=offset,original_energy=test_stats['e'],adjusted_energy=energy,test_mean=test_stats['s']/test_stats['n'])
     g,b=result['candidates']['graph'],result['candidates']['separate']
     result['original_ratio']=math.sqrt(g['original_energy']/b['original_energy']);result['both_adjusted_ratio']=math.sqrt(g['adjusted_energy']/b['adjusted_energy']);result['graph_adjusted_vs_original_baseline']=math.sqrt(g['adjusted_energy']/b['original_energy']);out.append(result)
 (P/'FRONTIER_READ_ERROR_MEAN_TRANSFER_V1.json').write_text(json.dumps(dict(rows=out,scope='Post-hoc descriptive analysis of two already-opened panels. Offsets estimated from natural errors only; donor-change offsets cancel. Both baselines treated equally. No frozen artifact changed; no fresh or circuit-identification claim.'),indent=2)+'\n')
 for r in out:
  if r['cohort']=='spaced_word' and r['family']!='change':print(json.dumps(r))
if __name__=='__main__':main()
