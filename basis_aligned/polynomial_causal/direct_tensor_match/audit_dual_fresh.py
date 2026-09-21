"""Audit all fresh comparison cells and paired recipient uncertainty for three frozen graphs."""
from pathlib import Path
import json,numpy as np
P=Path(__file__).parent;d=json.loads((P/'DUAL_FRESH_NATIVE_V1.json').read_text());rng=np.random.default_rng(10340);rows=[]
candidates=['graph','isotropic_graph','covariance_graph'];baselines=['separate','isotropic_baseline'];names=candidates+baselines
for cell in d['cells']:
 domain=cell['domain'];docs=d['plan']['recipient_documents'][domain];index={doc:i for i,doc in enumerate(docs)};arrays={name:np.zeros((len(docs),2)) for name in names}
 for r in d['records']:
  if r['domain']==domain and all(r[k]==cell[k] for k in ['selection','family','cohort']) and r['candidate'] in arrays:arrays[r['candidate']][index[r['document']]]+=[r['error_energy'],r['reference_energy']]
 point={name:float(np.sqrt(a[:,0].sum()/a[:,1].sum())) for name,a in arrays.items()}
 for name,value in point.items():assert abs(value-d['summary'][domain][cell['selection']][name][cell['family']][cell['cohort']]['effect_relative_error'])<1e-12
 ids=rng.integers(0,len(docs),(4000,len(docs)));totals={name:a[ids].sum(1) for name,a in arrays.items()};valid=np.logical_and.reduce([(a[:,1]>0)&(a[:,0]>0) for a in totals.values()]);boot={name:np.sqrt(a[valid,0]/a[valid,1]) for name,a in totals.items()}
 for candidate in candidates:
  comparisons={name:dict(point=point[candidate]/point[name],interval95=np.quantile(boot[candidate]/boot[name],[.025,.975]).tolist()) for name in baselines+(['covariance_graph'] if candidate!='covariance_graph' else [])}
  rows.append(dict(**{k:cell[k] for k in ['domain','selection','family','cohort']},candidate=candidate,point_errors=point,error_interval95=np.quantile(boot[candidate],[.025,.975]).tolist(),ratios=comparisons,valid_bootstrap_replicates=int(valid.sum())))
counts={}
for candidate in candidates:
 cells=[c for c in d['all_candidate_comparisons'] if c['candidate']==candidate];mine=[r for r in rows if r['candidate']==candidate]
 counts[candidate]=dict(cells=len(cells),absolute_failures=sum(not c['absolute_pass'] for c in cells),covariance_baseline_failures=sum(not c['covariance_relative_pass'] for c in cells),isotropic_baseline_failures=sum(not c['isotropic_relative_pass'] for c in cells),either_baseline_failures=sum(not(c['covariance_relative_pass'] and c['isotropic_relative_pass']) for c in cells),covariance_baseline_lower_CI_above_1_10=sum(r['ratios']['separate']['interval95'][0]>1.1 for r in mine),isotropic_baseline_lower_CI_above_1_10=sum(r['ratios']['isotropic_baseline']['interval95'][0]>1.1 for r in mine))
assert (counts['graph']['absolute_failures']==0)==d['predictions']['pred_b_absolute']
assert (counts['graph']['covariance_baseline_failures']==0)==d['predictions']['pred_c_relative']
assert (counts['graph']['either_baseline_failures']==0)==d['predictions']['pred_d_both_baselines']
out=dict(all_point_summaries_replay=True,predictions=d['predictions'],counts=counts,records=rows,scope='4000paired recipient-document/file bootstrap resamples, fixed donor map and frozen fits. Pointwise95%intervals without multiplicity correction; no uncertainty over training, donor choice or pretraining overlap. Originalfit covarianceguard remains failed; purearms descriptive, not selected as replacementprimary.')
(P/'DUAL_FRESH_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(counts,indent=2))
