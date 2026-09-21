"""New rows/donors only; preserve previously frozen program and weight files."""
from pathlib import Path
import json,hashlib,torch
P=Path(__file__).resolve().parent
plan=json.loads((P/'SHARED_MODE3_FRESH_PLAN_V1.json').read_text())
assert hashlib.sha256((P/'SHARED_MODE3_FRESH_PROGRAMS_V1.pt').read_bytes()).hexdigest()==plan['program_sha256']
assert hashlib.sha256((P/'MODE3_ORIGINAL_SOURCE_FORMS_V1.pt').read_bytes()).hexdigest()==plan['fold_sha256']
meta=json.loads((P/'SHARED_MODE3_SECOND_ROWS_V1.json').read_text());first=json.loads((P/'SHARED_MODE3_FRESH_ROWS_V1.json').read_text())
assert len({r['document_id'] for r in meta['fineweb_documents']})==32
assert not({r['document_id'] for r in meta['fineweb_documents']}&{r['document_id'] for r in first['fineweb_documents']})
tokens=torch.load(P/'SHARED_MODE3_SECOND_TOKENS_V1.pt',weights_only=True);donors={};stats={}
for domain,t in tokens.items():
 assert t.shape==(32 if domain=='fineweb' else 16,257)
 flat=t[:,16:256].flatten();groups={}
 for i,v in enumerate(flat.tolist()):groups.setdefault(v,[]).append(i)
 mapping=torch.full_like(flat,-1)
 for group in groups.values():
  uses={j:0 for j in group}
  for i in group:
   choices=[j for j in group if j//240!=i//240]
   if choices:
    j=min(choices,key=lambda j:(uses[j],abs(j%240-i%240),j));mapping[i]=j;uses[j]+=1
 valid=mapping>=0;assert torch.all(flat[valid]==flat[mapping[valid]])
 assert torch.all(torch.arange(len(flat))[valid]//240!=mapping[valid]//240)
 counts=torch.bincount(mapping[valid]);doccounts=torch.bincount(mapping[valid]//240)
 stats[domain]=dict(pairs=int(valid.sum()),max_site_fraction=float(counts.max()/valid.sum()),max_document_fraction=float(doccounts.max()/valid.sum()))
 donors[domain]=mapping
file=P/'SHARED_MODE3_SECOND_DONORS_V1.pt';torch.save(donors,file)
plan.update(donor_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),tokens_sha256={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in tokens.items()},recipient_documents={'fineweb':[r['document_id'] for r in meta['fineweb_documents']],'stdlib':[r['file'] for r in meta['code_sources']]},donor_concentration=stats,scope='Second32newdocument-indexedFW prefixes and16newstdlibfiles. Plain256 primary selected onfirstpanel, frozenbeforesecondoutcomes; no parameterrefit. Samebalanced donor rule and gates; native mode3 removal reference, nativeinputs retained.')
plan['predictions']['primary']='shared_plain256'
(P/'SHARED_MODE3_SECOND_PLAN_V1.json').write_text(json.dumps(plan,indent=2)+'\n')
print(stats)
