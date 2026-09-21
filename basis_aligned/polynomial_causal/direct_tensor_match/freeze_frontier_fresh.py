"""Freeze donors and preregister original constituent plus common-writer gates."""
from pathlib import Path
import hashlib,json,torch
P=Path(__file__).parent;torch.set_num_threads(2)
meta=json.loads((P/'FRONTIER_FRESH_ROWS_V1.json').read_text());previous=[]
for name in ['SHARED_MODE3_FRESH_ROWS_V1.json','SHARED_MODE3_SECOND_ROWS_V1.json','PARTIAL_GRAPH_FRESH_ROWS_V1.json','PARTIAL_GRAPH_FRESH_ROWS_V2.json','COMPACT_GROUP_FRESH_ROWS_V1.json','DUAL_FRESH_ROWS_V1.json']:previous+=json.loads((P/name).read_text())['fineweb_documents']
for key in ['document_id','text_sha256','prefix_sha256']:
 assert len({r[key] for r in meta['fineweb_documents']})==32
 assert not({r[key] for r in previous}&{r[key] for r in meta['fineweb_documents']})
tokens=torch.load(P/'FRONTIER_FRESH_TOKENS_V1.pt',weights_only=True);donors={};stats={}
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
file=P/'FRONTIER_FRESH_DONORS_V1.pt';torch.save(donors,file)
files=['FRONTIER_FRESH_GRAPH_V1.pt','FRONTIER_FRESH_BASELINE_CALIBRATION_SHAPED_V1.pt','FRONTIER_FRESH_BASELINE_NATIVE_ISOTROPIC_V1.pt','PARTIAL_GRAPH_ORIGINAL_FORMS_V1.pt','FRONTIER_FRESH_DONORS_V1.pt','pairwise_component_interface.py','pairwise_reader_graph.py','quadratic_pair_blocks.py','source_interface.py']
freeze=json.loads((P/'FRONTIER_FRESH_FREEZE_V1.json').read_text())
for item in freeze['candidates'].values():assert hashlib.sha256((P/item['file']).read_bytes()).hexdigest()==item['sha256']
from pack_reader_graph_artifacts import counts
program=torch.load(P/freeze['candidates']['graph']['file'],weights_only=True);before=counts(program);program['residual_writer']=program['pairs']['0']['residual_writer'];assert counts(program)==before;torch.save(program,P/'FRONTIER_FRESH_GRAPH_V1.pt')
executor=P.parents[1]/'bilinear_quotient/ops/run_partial_graph_fresh_v1.py'
plan=dict(file_sha256={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in files},native_executor_sha256=hashlib.sha256(executor.read_bytes()).hexdigest(),tokens_sha256={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in tokens.items()},recipient_documents={'fineweb':[r['document_id'] for r in meta['fineweb_documents']],'stdlib':[r['file'] for r in meta['code_sources']]},donor_concentration=stats,source_products={'graph':1120,'baseline':969,'isotropic_baseline':969},stored_float_coefficients={'graph':1131980,'baseline_writer_deduplicated':1129758},scope='Seventh distinct-document panel:32FW+16code, frozen64correction graph versus both323width resource-matched pair baselines. Every individual/combined natural/hybrid error<=15%,change<=20%,and<=1.10bothbaselines across domains/cohorts. Nativez/h supplied. Opened-state reconstruction passed; original20%costbar failed,15.7%saving retained. No refit against new rows; semantic identity and extraction remain unproven',freeze=freeze,primary='graph',baseline_names=['separate','isotropic_baseline'])
(P/'FRONTIER_FRESH_PLAN_V1.json').write_text(json.dumps(plan,indent=2)+'\n');print(stats)
