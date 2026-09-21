import json
from pathlib import Path
P=Path(__file__).resolve().parent

def main():
 d=json.loads((P/'INTERCHANGE_TRANSFER_V1.json').read_text());old=json.loads((P/'GENERATED_RESIDUAL_V1.json').read_text());key=lambda c:tuple(c[k] for k in ('panel','domain','family','cohort'));prior={key(c):c for c in old['summaries']};counts={};failures=[]
 for label,candidate in [('original',None),('paired','graph'),('cartesian','cartesian')]:
  counts[label]={}
  for interface in ('boundary','generated'):
   count=dict(absolute=0,covariance_relative=0,isotropic_relative=0,distinct_relative=0,total=0)
   for c in d['summaries']:
    values=(prior[key(c)]['candidates']['graph_'+interface] if candidate is None else c['candidates'][candidate+'_'+interface])['relative_error']
    for j,error in enumerate(values):
     cov=c['candidates']['separate_'+interface]['relative_error'][j];iso=c['candidates']['isotropic_'+interface]['relative_error'][j];absolute=error>(.20 if c['family']=='change' else .15);cf=error>1.1*cov;inf=error>1.1*iso;count['total']+=1;count['absolute']+=absolute;count['covariance_relative']+=cf;count['isotropic_relative']+=inf;count['distinct_relative']+=cf or inf
     if absolute or cf or inf:failures.append(dict(candidate=label,interface=interface,**{k:c[k] for k in ('panel','domain','family','cohort')},component=j+1 if j<3 else 'combined',error=error,covariance_ratio=error/cov,isotropic_ratio=error/iso,absolute_failure=absolute))
   counts[label][interface]=count
 out=dict(counts=counts,failures=failures,scope='Same two opened panels and323baselines. Neither normalizer-clamped fit objective is itself actual native donor behavior. No fresh acceptance.')
 (P/'INTERCHANGE_TRANSFER_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(counts,indent=2));print(json.dumps([f for f in failures if f['absolute_failure']],indent=2))
if __name__=='__main__':main()
