"""Recompute native gates and paired document-bootstrap intervals."""
from pathlib import Path
import json,numpy as np
P=Path(__file__).resolve().parent;d=json.loads((P/'PARTIAL_GRAPH_FRESH_NATIVE_V1.json').read_text());rng=np.random.default_rng(729);records=[]
for cell in d['cells']:
 domain=cell['domain'];docs=d['plan']['recipient_documents'][domain];arrays={name:np.zeros((len(docs),2)) for name in ['graph','separate']};index={doc:i for i,doc in enumerate(docs)}
 for r in d['records']:
  if r['domain']==domain and all(r[k]==cell[k] for k in ['selection','family','cohort']) and r['candidate'] in arrays:
   arrays[r['candidate']][index[r['document']]]+=[r['error_energy'],r['reference_energy']]
 g=arrays['graph'];b=arrays['separate'];point=np.sqrt(g[:,0].sum()/g[:,1].sum());base=np.sqrt(b[:,0].sum()/b[:,1].sum());assert abs(point-cell['graph_error'])<1e-12 and abs(base-cell['baseline_error'])<1e-12
 ids=rng.integers(0,len(docs),(4000,len(docs)));gg=g[ids].sum(1);bb=b[ids].sum(1);valid=(gg[:,1]>0)&(bb[:,1]>0)&(bb[:,0]>0);ge=np.sqrt(gg[valid,0]/gg[valid,1]);be=np.sqrt(bb[valid,0]/bb[valid,1]);ratio=ge/be
 records.append({**{k:cell[k] for k in ['domain','selection','family','cohort']},'recipient_units':len(docs),'point_graph_error':point,'point_ratio':point/base,'graph_error_interval95':np.quantile(ge,[.025,.975]).tolist(),'ratio_interval95':np.quantile(ratio,[.025,.975]).tolist(),'valid_bootstrap_replicates':int(valid.sum())})
mode3=[c for c in d['cells'] if c['selection']=='mode3'];identical=max(abs(c['graph_error']-c['baseline_error']) for c in mode3)
out=dict(records=records,all_point_summaries_replay=True,private_mode3_max_error_difference=identical,absolute_failures=[c for c in d['cells'] if not c['absolute_pass']],relative_failures=[c for c in d['cells'] if not c['relative_pass']],scope='4000 paired recipient-document/file bootstrap replicates, donors frozen. Includes all registered recipient units, including zero-count subgroup units. Does not quantify donor-map or training-selection uncertainty. Point thresholds are unchanged.')
(P/'PARTIAL_GRAPH_FRESH_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(private_mode3_max_error_difference=identical,absolute_failures=out['absolute_failures'],relative_failures=out['relative_failures'],failed_cell_intervals=[r for r in records if r['domain']=='fineweb' and r['selection']=='mode3' and r['family']=='natural' and r['cohort']=='continuation']),indent=2))
