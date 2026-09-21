"""Freeze programs, native mode identity and balanced donors before evaluation."""
from pathlib import Path
import torch,json,hashlib
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
shared=torch.load(P/'SHARED_EXPANDED_SOURCE_PROGRAMS_V1.pt',weights_only=True);ind=torch.load(P/'EXPANDED_SOURCE_METRIC_PROGRAMS_V1.pt',weights_only=True)
programs={'independent128':ind['128'],'shared_plain256':shared['plain_256'],'shared_affine256':shared['affine_256']}
programfile=P/'SHARED_MODE3_FRESH_PROGRAMS_V1.pt';torch.save(programs,programfile)
mode=torch.load(P/'MIDPOINT_NATIVE_OBSERVER_MODES_V1.pt',weights_only=True)
for p in programs.values():assert torch.equal(p['h_reader'],mode['A'][:,2])
ck='/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin'
state=torch.load(ck,weights_only=True,mmap=True,map_location='cpu');L=state['transformer.h.16.mlp.Left.weight'].double();R=state['transformer.h.16.mlp.Right.weight'].double();D=state['transformer.h.16.mlp.Down.weight'].double();lam=state['transformer.h.17.lambdas'][0].double();fold={}
for k,reader in [('a',mode['A'][:,2]),('b',mode['B'][:,2])]:
 raw=L.T@((lam*(D.T@reader))[:,None]*R);fold[k]={'matrix':(raw+raw.T)/2}
foldfile=P/'MODE3_ORIGINAL_SOURCE_FORMS_V1.pt';torch.save(fold,foldfile)
tokens=torch.load(P/'SHARED_MODE3_FRESH_TOKENS_V1.pt',weights_only=True);rowmeta=json.loads((P/'SHARED_MODE3_FRESH_ROWS_V1.json').read_text());donors={};stats={}
for domain,t in tokens.items():
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
donorfile=P/'SHARED_MODE3_FRESH_DONORS_V1.pt';torch.save(donors,donorfile)
plan=dict(program_sha256=hashlib.sha256(programfile.read_bytes()).hexdigest(),donor_sha256=hashlib.sha256(donorfile.read_bytes()).hexdigest(),fold_sha256=hashlib.sha256(foldfile.read_bytes()).hexdigest(),tokens_sha256={k:hashlib.sha256(t.numpy().tobytes()).hexdigest() for k,t in tokens.items()},recipient_documents={'fineweb':[r['document_id'] for r in rowmeta['fineweb_documents']],'stdlib':list(range(len(tokens['stdlib'])))},mode_index=2,donor_concentration=stats,
 predictions={'primary':'shared_affine256','natural_max_error':.15,'hybrid_max_error':.15,'change_max_error':.20,'relative_to_independent128_max':1.10,'all_domains_and_cohorts':True},
 scope='32 document-identified FW prefixes excluding knowncachedexcerpts,16 reusedstdlib snippets. Context256 versusfit64. Source interchange balanced bydonorreuse, samecurrenttoken/differentdocument on FW. Attention17fixed; recomputeRMS/lastMLP. Reference native mode3 removal, notfullsource orsemanticconfirmation.')
(P/'SHARED_MODE3_FRESH_PLAN_V1.json').write_text(json.dumps(plan,indent=2)+'\n')
print({'mode':2,'programs':list(programs),'donors':stats})
