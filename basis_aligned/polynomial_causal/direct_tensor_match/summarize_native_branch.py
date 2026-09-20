"""Document bootstrap for registered native branch screen (no fitting/selection)."""
import json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent

def main():
 result=json.loads((P/'NATIVE_QUARTIC_BRANCH_V1.json').read_text())
 records=result['records'];docs=result['plan']['documents'];rng=np.random.default_rng(260920)
 indices=rng.integers(0,len(docs),size=(10000,len(docs)));summaries={}
 for arm in result['plan']['arms']:
  rows=[next(r for r in records if r['document']==d and r['arm']==arm) for d in docs]
  summary={}
  for key in ['ce_added','kl','logit_mse','argmax_agreement']:
   values=np.array([r[key] for r in rows]);means=values[indices].mean(1)
   summary[key]=dict(mean=float(values.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),min=float(values.min()),max=float(values.max()))
  summaries[arm]=summary
 a=np.array([next(r['logit_mse'] for r in records if r['document']==d and r['arm']=='quartic10') for d in docs])
 b=np.array([next(r['logit_mse'] for r in records if r['document']==d and r['arm']=='ablation') for d in docs])
 ratio=np.sqrt(a[indices].mean(1)/b[indices].mean(1))
 output=dict(documents=len(docs),bootstrap_seed=260920,replicates=10000,arms=summaries,logit_norm_ratio_ci95=np.quantile(ratio,[.025,.975]).tolist(),scope='Document-resampled uncertainty within one FineWeb panel, not OOD certification.')
 (P/'NATIVE_QUARTIC_BRANCH_UNCERTAINTY_V1.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))
if __name__=='__main__':main()
