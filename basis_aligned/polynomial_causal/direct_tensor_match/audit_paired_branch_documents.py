"""Post-screen paired document uncertainty; never changes registered verdicts."""
from pathlib import Path
import json
import numpy as np
P=Path(__file__).resolve().parent

def main():
 source=json.loads((P/'PAIRED_ROOT_BRANCH_V1.json').read_text())
 rng=np.random.default_rng(2609212008)
 results=[]
 for domain in sorted({r['domain'] for r in source['rows']}):
  rows=[r for r in source['rows'] if r['domain']==domain]
  ids=sorted({r['document'] for r in rows}); lookup={(r['document'],r['arm']):r for r in rows}
  n=len(ids); samples=rng.integers(0,n,size=(10000,n))
  for reference in ['narrow26','parent656']:
   a=np.array([lookup[i,'paired384']['error_energy'] for i in ids])
   b=np.array([lookup[i,reference]['error_energy'] for i in ids])
   ratios=np.sqrt(a[samples].sum(1)/b[samples].sum(1))
   loo=np.sqrt((a.sum()-a)/(b.sum()-b))
   ce=np.array([lookup[i,'paired384']['ce_added']-lookup[i,reference]['ce_added'] for i in ids])
   kl=np.array([lookup[i,'paired384']['kl']-lookup[i,reference]['kl'] for i in ids])
   results.append(dict(domain=domain,reference=reference,documents=n,
    effect_error_ratio=float(np.sqrt(a.sum()/b.sum())),
    bootstrap95_effect_ratio=np.quantile(ratios,[.025,.975]).tolist(),
    leave_one_out_effect_ratio_range=[float(loo.min()),float(loo.max())],
    documents_paired_lower_error=int((a<b).sum()),
    mean_ce_difference=float(ce.mean()),bootstrap95_ce_difference=np.quantile(ce[samples].mean(1),[.025,.975]).tolist(),
    mean_kl_difference=float(kl.mean()),bootstrap95_kl_difference=np.quantile(kl[samples].mean(1),[.025,.975]).tolist()))
 out=dict(scope='Post-hoc document bootstrap on opened panels. Descriptive, no independent OOD or adjusted hypothesis-test claim.',seed=2609212008,resamples=10000,results=results)
 (P/'PAIRED_ROOT_BRANCH_DOCUMENT_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
