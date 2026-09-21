"""Paired document bootstrap of frozen source-extraction effects; no refitting."""
from pathlib import Path
import json,numpy as np
p=Path(__file__).resolve().parent;d=json.loads((p/'MIDPOINT_SOURCE_NATIVE_V1.json').read_text());rng=np.random.default_rng(20260921);result={}
for domain in d['summary']:
 result[domain]={};docs=d['plan']['documents'][domain];idx=rng.integers(len(docs),size=(10000,len(docs)))
 for cohort in ['all','continuation','spaced_word']:
  rows={a:[next(r for r in d['records'] if r['domain']==domain and r['document']==doc and r['cohort']==cohort and r['candidate']==a) for doc in docs] for a in d['summary'][domain]}
  arr=lambda a,k:np.array([r[k] for r in rows[a]])
  sites=arr('native','sites')[idx].sum(1);valid=sites>0;den=arr('native','reference_energy')[idx].sum(1);r={}
  for arm in ['linear','isotropic_16','covariance_16','covariance_64']:
   error=arr(arm,'error_energy')[idx].sum(1);difference=(arr(arm,'ce_added_sum')-arr('native','ce_added_sum'))[idx].sum(1)/np.maximum(sites,1)
   r[arm]=dict(effect_error_95=np.quantile(np.sqrt(error[valid]/den[valid]),[.025,.975]).tolist(),ce_difference_95=np.quantile(difference[valid],[.025,.975]).tolist(),valid_draws=int(valid.sum()))
  ratio=np.sqrt(arr('covariance_16','error_energy')[idx].sum(1)/np.maximum(arr('linear','error_energy')[idx].sum(1),1e-30))
  r['covariance16_vs_linear_error_ratio_95']=np.quantile(ratio[valid],[.025,.975]).tolist();result[domain][cohort]=r
out=dict(intervals=result,scope='10000 paired document draws; conditional descriptive intervals, no simultaneous coverage or refitting. FineWeb fresh, stdlib reused. Error reference exact native rank1 feature, not whole native operator.')
(p/'MIDPOINT_SOURCE_NATIVE_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n')
for domain,cohorts in result.items():
 for cohort,rows in cohorts.items():print(domain,cohort,'cov16',rows['covariance_16'],'ratio vs linear',rows['covariance16_vs_linear_error_ratio_95'])
