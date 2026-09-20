"""Paired document uncertainty for the frozen fresh confirmation."""
import json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent

def main():
 rng=np.random.default_rng(260934);output={}
 for domain in ['FINEWEB','CODE']:
  b=json.loads((P/f'BLEND_CONFIRMATION_BRANCH_{domain}_V1.json').read_text());m=json.loads((P/f'BLEND_CONFIRMATION_MODE_{domain}_PRIMARY_V1.json').read_text());docs=b['plan']['documents'];ix=rng.integers(0,len(docs),(10000,len(docs)));rows={}
  for key in ['ce_added','kl','logit_mse']:
   a=np.array([next(r[key] for r in b['records'] if r['document']==d and r['arm']=='primary') for d in docs]);control=np.array([next(r[key] for r in b['records'] if r['document']==d and r['arm']=='gaussian') for d in docs]);rows[key]=dict(primary_mean=float(a.mean()),primary_ci95=np.quantile(a[ix].mean(1),[.025,.975]).tolist(),paired_delta_mean=float((a-control).mean()),paired_delta_ci95=np.quantile((a-control)[ix].mean(1),[.025,.975]).tolist())
  rows['modes']={}
  for mode in m['plan']['modes']:
   rr=[next(r for r in m['records'] if r['document']==d and r['mode']==mode) for d in docs]
   def sums(k):return np.array([r[k] for r in rr])[ix].sum(1)
   n=sums('native_effect_energy');p=sums('predicted_effect_energy');dot=sums('effect_dot');e=sums('effect_error_energy');rows['modes'][str(mode)]=dict(error_ci95=np.quantile(np.sqrt(e/n),[.025,.975]).tolist(),cosine_ci95=np.quantile(dot/np.sqrt(n*p),[.025,.975]).tolist())
  output[domain.lower()]=rows
 result=dict(seed=260934,replicates=10000,records=output,scope='Fresh frozen confirmation document bootstrap. Local code sources related; no claim of corpus-wide independence. Strict point-estimate failures retained.')
 (P/'BLEND_CONFIRMATION_UNCERTAINTY_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
