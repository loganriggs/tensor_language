"""Conditional recipient-document bootstrap for fixed donor interchanges."""
from pathlib import Path
import json,numpy as np
p=Path(__file__).resolve().parent;rng=np.random.default_rng(2130);outputs={}
for stem,numerator,denominator in [('MIDPOINT_SOURCE_INTERCHANGE','covariance_16','isotropic_0'),('MIDPOINT_SHARED_SOURCE_NATIVE','shared16','covariance_16'),('MIDPOINT_BALANCED_SOURCE_NATIVE','shared16','covariance_16')]:
 data=json.loads((p/(stem+'_V1.json')).read_text());result={}
 for domain in data['summary']:
  docs=data['plan']['recipient_documents'][domain];draw=rng.integers(len(docs),size=(10000,len(docs)));result[domain]={}
  for family in ['hybrid','change']:
   result[domain][family]={}
   for cohort in ['all','continuation','spaced_word']:
    def totals(arm,key):return np.array([sum(r[key] for r in data['records'] if r['domain']==domain and r['document']==doc and r['candidate']==arm and r['family']==family and r['cohort']==cohort) for doc in docs])
    err=totals(numerator,'error_energy')[draw].sum(1);base=totals(denominator,'error_energy')[draw].sum(1);ref=totals(numerator,'reference_energy')[draw].sum(1);valid=(base>0)&(ref>0)
    result[domain][family][cohort]=dict(effect_error_95=np.quantile(np.sqrt(err[valid]/ref[valid]),[.025,.975]).tolist(),error_ratio_95=np.quantile(np.sqrt(err[valid]/base[valid]),[.025,.975]).tolist(),valid_draws=int(valid.sum()))
 extra={}
 if stem=='MIDPOINT_SHARED_SOURCE_NATIVE':
  extra['full32_max_summary_replay']=max(abs(data['summary'][d]['shared32'][f][c]['effect_relative_error']-data['summary'][d]['covariance_16'][f][c]['effect_relative_error']) for d in data['summary'] for f in ['hybrid','change'] for c in ['all','continuation','spaced_word']);assert extra['full32_max_summary_replay']<1e-7
  prior=json.loads((p/'MIDPOINT_SOURCE_INTERCHANGE_V1.json').read_text());extra['reused_code_baseline_replay']=max(abs(data['summary']['stdlib']['covariance_16'][f][c]['effect_relative_error']-prior['summary']['stdlib']['covariance_16'][f][c]['effect_relative_error']) for f in ['hybrid','change'] for c in ['all','continuation','spaced_word']);assert extra['reused_code_baseline_replay']<1e-7
 out=dict(numerator=numerator,denominator=denominator,intervals=result,controls=extra,scope='10000 paired recipient-document bootstrap draws, conditional on fixed deterministic donors. Donors repeat and may serve different recipients; these intervals do not capture uncertainty from donor selection. No simultaneous coverage.')
 (p/(stem+'_AUDIT_V1.json')).write_text(json.dumps(out,indent=2)+'\n');outputs[stem]=out
for name,out in outputs.items():
 print(name,out['controls'])
 for d,families in out['intervals'].items():
  for f,cohorts in families.items():print(d,f,{c:v['error_ratio_95'] for c,v in cohorts.items()})
