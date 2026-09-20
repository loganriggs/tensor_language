"""Paired uncertainty for the preregistered rank-two skip, no rank selection."""
import json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent

def main():
 rng=np.random.default_rng(260931);rows=[]
 for domain,oldname in [('fineweb','NATIVE_QUARTIC_BRANCH_V1.json'),('code','CODE_SHIFT_BRANCH_V1.json')]:
  old=json.loads((P/oldname).read_text());new=json.loads((P/f'QUADRATIC_SKIP_BRANCH_{domain.upper()}_V1.json').read_text());assert old['token_hash']==new['token_hash'];docs=old['plan']['documents'];indices=rng.integers(0,len(docs),(10000,len(docs)))
  a=[next(r for r in old['records'] if r['document']==d and r['arm']=='quartic10') for d in docs];b=[next(r for r in new['records'] if r['document']==d and r['arm']=='skip2') for d in docs]
  assert max(abs(x['native_ce']-y['native_ce']) for x,y in zip(a,b))<1e-6
  for key in ['ce_added','kl','logit_mse','argmax_agreement']:
   delta=np.array([y[key]-x[key] for x,y in zip(a,b)]);rows.append(dict(domain=domain,metric=key,rank2_minus_base=float(delta.mean()),ci95=np.quantile(delta[indices].mean(1),[.025,.975]).tolist(),improved_documents=int((delta>0 if key=='argmax_agreement' else delta<0).sum()),documents=len(docs)))
 result=dict(records=rows,scope='Paired reused-panel diagnostic for fixed preregistered rank2. Related code files and repeated panel use limit inference; no rank selection.')
 (P/'QUADRATIC_SKIP_PAIRED_GAIN_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
