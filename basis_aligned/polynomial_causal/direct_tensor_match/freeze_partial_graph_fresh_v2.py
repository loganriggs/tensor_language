"""Fresh donors and identities; private320 fixed by the registered capacity plan."""
from pathlib import Path
import json,torch,hashlib
P=Path(__file__).resolve().parent;torch.set_num_threads(2)
meta=json.loads((P/'PARTIAL_GRAPH_FRESH_ROWS_V2.json').read_text());previous=[]
for name in ['SHARED_MODE3_FRESH_ROWS_V1.json','SHARED_MODE3_SECOND_ROWS_V1.json','PARTIAL_GRAPH_FRESH_ROWS_V1.json']:previous+=json.loads((P/name).read_text())['fineweb_documents']
assert len({r['document_id'] for r in meta['fineweb_documents']})==32
for key in ['document_id','text_sha256','prefix_sha256']:assert not({r[key] for r in previous}&{r[key] for r in meta['fineweb_documents']})
tokens=torch.load(P/'PARTIAL_GRAPH_FRESH_TOKENS_V2.pt',weights_only=True);donors={};stats={}
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
file=P/'PARTIAL_GRAPH_FRESH_DONORS_V2.pt';torch.save(donors,file)
plan=json.loads((P/'PARTIAL_GRAPH_FRESH_PLAN_V1.json').read_text());files=['PARTIAL_GRAPH_FROZEN_V2.pt','PARTIAL_GRAPH_BASELINES_V2.pt','PARTIAL_GRAPH_ORIGINAL_FORMS_V1.pt','PARTIAL_GRAPH_FRESH_DONORS_V2.pt']
plan.update(private_width=320,file_sha256={name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in files},tokens_sha256={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in tokens.items()},recipient_documents={'fineweb':[r['document_id'] for r in meta['fineweb_documents']],'stdlib':[r['file'] for r in meta['code_sources']]},donor_concentration=stats,source_products={'graph':576,'baseline':832},stored_float_coefficients={'graph':971660,'baseline_writer_deduplicated':971660},scope='Fourth newdocument-indexedpanel,32FineWeb+16newcodefiles, excludingprevious3panels. Private320 capacity selectedbeforeoutcomes; identicalprivatebranch suppliedtobaseline. Sharedfirsttwo components unchanged. All originalnativegates retained; nativeinputs stillrequired.')
(P/'PARTIAL_GRAPH_FRESH_PLAN_V2.json').write_text(json.dumps(plan,indent=2)+'\n');print(stats)
