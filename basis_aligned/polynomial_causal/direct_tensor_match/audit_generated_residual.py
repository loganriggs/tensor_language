import json,math
from pathlib import Path
P=Path(__file__).resolve().parent

def main():
 d=json.loads((P/'GENERATED_RESIDUAL_V1.json').read_text());pred=json.loads((P/'GENERATED_ERROR_GRAM_PREDICTION_V1.json').read_text());replay=[]
 for cell in pred['rows']:
  for name in ('graph','separate'):
   actual=sum(c['candidates'][name+'_generated']['error_energy'][2] for c in d['summaries'] if c['domain']=='stdlib' and c['family']==cell['family'] and c['cohort']==cell['cohort'])
   replay.append(abs(actual-cell['energy'][name])/cell['energy'][name])
 failures=[];counts={mode:dict(absolute_failures=0,relative_covariance_failures=0,relative_isotropic_failures=0,total=0) for mode in ('boundary','generated')}
 for c in d['summaries']:
  for mode in counts:
   for j,g in enumerate(c['candidates']['graph_'+mode]['relative_error']):
    b=c['candidates']['separate_'+mode]['relative_error'][j];i=c['candidates']['isotropic_'+mode]['relative_error'][j];absolute=g>(.20 if c['family']=='change' else .15);relative=g>1.1*b;iso=g>1.1*i;counts[mode]['total']+=1;counts[mode]['absolute_failures']+=absolute;counts[mode]['relative_covariance_failures']+=relative;counts[mode]['relative_isotropic_failures']+=iso
    if absolute or relative or iso:failures.append(dict(**{k:c[k] for k in ('panel','domain','family','cohort')},interface=mode,component=j+1 if j<3 else 'combined',error=g,covariance_ratio=g/b,isotropic_ratio=g/i,absolute_failure=absolute,covariance_failure=relative,isotropic_failure=iso))
 assert max(replay)<1e-5
 out=dict(counts=counts,failures=failures,independent_gram_prediction_max_relative_difference=max(replay),scope='Centered scalar errors; these are not the earlier centered vocabulary-effect metrics. Both interfaces replayed with same rows. No endpoint or new fresh acceptance.')
 (P/'GENERATED_RESIDUAL_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(counts));print('Gram replay',max(replay));print(json.dumps([f for f in failures if f['interface']=='generated'][:12],indent=2))
if __name__=='__main__':main()
