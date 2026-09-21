"""Recompute fresh native gates and paired recipient-level uncertainty against both baselines."""
from pathlib import Path
import json,numpy as np
P=Path(__file__).parent;d=json.loads((P/'COMPACT_GROUP_FRESH_NATIVE_V1.json').read_text());rng=np.random.default_rng(9171);records=[]
for cell in d['cells']:
 domain=cell['domain'];docs=d['plan']['recipient_documents'][domain];index={doc:i for i,doc in enumerate(docs)};arrays={name:np.zeros((len(docs),2)) for name in ['graph','separate','partial']}
 for r in d['records']:
  if r['domain']==domain and all(r[k]==cell[k] for k in ['selection','family','cohort']) and r['candidate'] in arrays:arrays[r['candidate']][index[r['document']]]+=[r['error_energy'],r['reference_energy']]
 point={name:float(np.sqrt(a[:,0].sum()/a[:,1].sum())) for name,a in arrays.items()};assert abs(point['graph']-cell['graph_error'])<1e-12 and abs(point['separate']-cell['baseline_error'])<1e-12
 expected=d['summary'][domain][cell['selection']]['partial'][cell['family']][cell['cohort']]['effect_relative_error'];assert abs(point['partial']-expected)<1e-12
 ids=rng.integers(0,len(docs),(4000,len(docs)));summed={name:a[ids].sum(1) for name,a in arrays.items()};valid=np.logical_and.reduce([(a[:,1]>0)&(a[:,0]>0) for a in summed.values()]);boot={name:np.sqrt(a[valid,0]/a[valid,1]) for name,a in summed.items()}
 record={k:cell[k] for k in ['domain','selection','family','cohort']};record.update(recipient_units=len(docs),point_errors=point,error_interval95=np.quantile(boot['graph'],[.025,.975]).tolist(),ratios={name:dict(point=point['graph']/point[name],interval95=np.quantile(boot['graph']/boot[name],[.025,.975]).tolist()) for name in ['separate','partial']},valid_bootstrap_replicates=int(valid.sum()));records.append(record)
combined=[c for c in d['combined_group_cells']];replay=all(c['absolute_pass'] and c['relative_pass'] and c['partial_relative_pass'] for c in combined);assert replay==d['predictions']['pred_d_combined_group']
out=dict(all_point_summaries_replay=True,combined_group_gate=replay,original_absolute_failures=[c for c in d['cells'] if not c['absolute_pass']],original_relative_failures=[c for c in d['cells'] if not c['relative_pass']],combined_failures=[c for c in combined if not(c['absolute_pass'] and c['relative_pass'] and c['partial_relative_pass'])],records=records,scope='4000paired recipient-document/file resamples; donor map and fitted candidate fixed. All original gates retained. Does not quantify donor selection, refitting or model-pretraining overlap. No semantic promotion.')
(P/'COMPACT_GROUP_FRESH_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='records'},indent=2))
