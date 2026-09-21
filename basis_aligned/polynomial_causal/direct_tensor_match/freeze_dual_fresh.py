"""Freeze donors and preregister original constituent plus common-writer gates."""
from pathlib import Path
import hashlib,json,torch
P=Path(__file__).parent;torch.set_num_threads(2)
meta=json.loads((P/'DUAL_FRESH_ROWS_V1.json').read_text());previous=[]
for name in ['SHARED_MODE3_FRESH_ROWS_V1.json','SHARED_MODE3_SECOND_ROWS_V1.json','PARTIAL_GRAPH_FRESH_ROWS_V1.json','PARTIAL_GRAPH_FRESH_ROWS_V2.json','COMPACT_GROUP_FRESH_ROWS_V1.json']:previous+=json.loads((P/name).read_text())['fineweb_documents']
for key in ['document_id','text_sha256','prefix_sha256']:
 assert len({r[key] for r in meta['fineweb_documents']})==32
 assert not({r[key] for r in previous}&{r[key] for r in meta['fineweb_documents']})
tokens=torch.load(P/'DUAL_FRESH_TOKENS_V1.pt',weights_only=True);donors={};stats={}
for domain,t in tokens.items():
 assert t.shape==(32 if domain=='fineweb' else 16,257);flat=t[:,16:256].flatten();groups={}
 for i,v in enumerate(flat.tolist()):groups.setdefault(v,[]).append(i)
 mapping=torch.full_like(flat,-1)
 for group in groups.values():
  uses={j:0 for j in group}
  for i in group:
   options=[j for j in group if j//240!=i//240]
   if options:
    j=min(options,key=lambda j:(uses[j],abs(j%240-i%240),j));mapping[i]=j;uses[j]+=1
 valid=mapping>=0;assert torch.all(flat[valid]==flat[mapping[valid]]) and torch.all(torch.arange(len(flat))[valid]//240!=mapping[valid]//240)
 counts=torch.bincount(mapping[valid]);docs=torch.bincount(mapping[valid]//240);stats[domain]=dict(pairs=int(valid.sum()),max_site_fraction=float(counts.max()/valid.sum()),max_document_fraction=float(docs.max()/valid.sum()));donors[domain]=mapping
file=P/'DUAL_FRESH_DONORS_V1.pt';torch.save(donors,file)
files=['DUAL_FRESH_MIXED_V1.pt','DUAL_FRESH_ISOTROPIC_V1.pt','DUAL_FRESH_COVARIANCE_V1.pt','DUAL_FRESH_BASELINE_CALIBRATION_SHAPED_V1.pt','DUAL_FRESH_BASELINE_NATIVE_ISOTROPIC_V1.pt','PARTIAL_GRAPH_ORIGINAL_FORMS_V1.pt','DUAL_FRESH_DONORS_V1.pt','compact_source_graph.py','global_mixed_source_graph.py','source_interface.py']
freeze=json.loads((P/'DUAL_FRESH_FREEZE_V1.json').read_text())
for item in freeze['candidates'].values():assert hashlib.sha256((P/item['file']).read_bytes()).hexdigest()==item['sha256']
executor=P.parents[1]/'bilinear_quotient/ops/run_partial_graph_fresh_v1.py'
plan=dict(file_sha256={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in files},native_executor_sha256=hashlib.sha256(executor.read_bytes()).hexdigest(),tokens_sha256={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in tokens.items()},recipient_documents={'fineweb':[r['document_id'] for r in meta['fineweb_documents']],'stdlib':[r['file'] for r in meta['code_sources']]},donor_concentration=stats,source_products={'graph':592,'baseline':1152,'isotropic_graph':592,'covariance_graph':592,'isotropic_baseline':1152},stored_float_coefficients={'graph':1342028,'baseline_writer_deduplicated':1340940},scope='Fresh sixth identified-document panel; same48captures for all3frozenwidegraphs andbothmatchedpairbaselines. Mixed primary must meet every individual/combined absolute15%natural/hybrid,20%change and<=1.10bothbaselines. Purearms descriptive, no outcome-based arm selection. Priorfit covarianceguardFAIL remains; this separate behavioral metric-transfer experiment is not adoption. Nativez/h supplied.',freeze=freeze,primary='graph',baseline_names=['separate','isotropic_baseline'])
(P/'DUAL_FRESH_PLAN_V1.json').write_text(json.dumps(plan,indent=2)+'\n');print(stats)
