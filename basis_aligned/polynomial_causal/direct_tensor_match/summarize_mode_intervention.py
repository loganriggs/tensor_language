"""Paired document bootstrap of scalar-removal effects, preserving weak modes."""
import json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent

def main():
 result=json.loads((P/'NATIVE_MODE_INTERVENTION_V1.json').read_text());docs=result['plan']['documents'];rng=np.random.default_rng(260921);ix=rng.integers(0,len(docs),(10000,len(docs)));out={}
 for mode in result['plan']['modes']:
  rows=[next(r for r in result['records'] if r['mode']==mode and r['document']==d) for d in docs]
  def sums(k):return np.array([r[k] for r in rows])[ix].sum(1)
  entry={}
  for prefix in ['effect','ce_effect']:
   n=sums('native_'+prefix+'_energy');p=sums('predicted_'+prefix+'_energy');dot=sums(prefix+'_dot');err=sums(prefix+'_error_energy')
   entry[prefix+'_cosine_ci95']=np.quantile(dot/np.sqrt(n*p),[.025,.975]).tolist();entry[prefix+'_relative_error_ci95']=np.quantile(np.sqrt(err/n),[.025,.975]).tolist()
  out[str(mode)]=entry
 output=dict(bootstrap_seed=260921,replicates=10000,modes=out,scope='Uncertainty conditional on one FineWeb panel and fixed feature definitions, not OOD.')
 (P/'NATIVE_MODE_INTERVENTION_UNCERTAINTY_V1.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))
if __name__=='__main__':main()
