"""Freeze coefficient-selected graph, original factors, and document donors."""
from pathlib import Path
import json,hashlib,torch
P=Path(__file__).resolve().parent;torch.set_num_threads(2)
fit=json.loads((P/'PROFILED_PARTIAL_GRAPH_FIT_V1.json').read_text());assert all(fit['predictions'].values());audit=json.loads((P/'PROFILED_PARTIAL_GRAPH_AUDIT_V1.json').read_text());assert audit['all_checks_pass']
graph=torch.load(P/'PROFILED_PARTIAL_GRAPH_PROGRAMS_V1.pt',weights_only=True)[fit['winner']];torch.save(graph,P/'PARTIAL_GRAPH_FROZEN_V1.pt')
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True)
fold=dict(matrices=torch.stack([Q for pair in d['pairs'] for Q in pair['Qs']]),A=torch.stack([p['a'] for p in d['pairs']],1),B=torch.stack([p['b'] for p in d['pairs']],1),alpha=torch.stack([p['alpha'] for p in d['pairs']]),beta=torch.stack([p['beta'] for p in d['pairs']]))
assert torch.equal(fold['A'][:,:2],graph['shared_mixed']['h_readers']);assert torch.equal(fold['A'][:,2],graph['private_pair']['h_reader']);torch.save(fold,P/'PARTIAL_GRAPH_ORIGINAL_FORMS_V1.pt')
meta=json.loads((P/'PARTIAL_GRAPH_FRESH_ROWS_V1.json').read_text());prior=[row for version in ['FRESH','SECOND'] for row in json.loads((P/f'SHARED_MODE3_{version}_ROWS_V1.json').read_text())['fineweb_documents']]
assert len({r['document_id'] for r in meta['fineweb_documents']})==32
for key in ['document_id','text_sha256','prefix_sha256']:assert not ({r[key] for r in prior}&{r[key] for r in meta['fineweb_documents']})
tokens=torch.load(P/'PARTIAL_GRAPH_FRESH_TOKENS_V1.pt',weights_only=True);donors={};stats={}
for domain,t in tokens.items():
 assert t.shape==(32 if domain=='fineweb' else 16,257);flat=t[:,16:256].flatten();groups={}
 for i,v in enumerate(flat.tolist()):groups.setdefault(v,[]).append(i)
 mapping=torch.full_like(flat,-1)
 for group in groups.values():
  uses={j:0 for j in group}
  for i in group:
   choices=[j for j in group if j//240!=i//240]
   if choices:
    j=min(choices,key=lambda j:(uses[j],abs(j%240-i%240),j));mapping[i]=j;uses[j]+=1
 valid=mapping>=0;assert torch.all(flat[valid]==flat[mapping[valid]]) and torch.all(torch.arange(len(flat))[valid]//240!=mapping[valid]//240)
 counts=torch.bincount(mapping[valid]);docs=torch.bincount(mapping[valid]//240);stats[domain]=dict(pairs=int(valid.sum()),max_site_fraction=float(counts.max()/valid.sum()),max_document_fraction=float(docs.max()/valid.sum()));donors[domain]=mapping
file=P/'PARTIAL_GRAPH_FRESH_DONORS_V1.pt';torch.save(donors,file)
files=['PARTIAL_GRAPH_FROZEN_V1.pt','MULTIMODE_PAIR_BASELINES_V1.pt','PARTIAL_GRAPH_ORIGINAL_FORMS_V1.pt','PARTIAL_GRAPH_FRESH_DONORS_V1.pt']
plan=dict(parent_winner=fit['winner'],file_sha256={name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in files},tokens_sha256={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in tokens.items()},recipient_documents={'fineweb':[r['document_id'] for r in meta['fineweb_documents']],'stdlib':[r['file'] for r in meta['code_sources']]},donor_concentration=stats,predictions=dict(pred_a_instrument='finalstate<1e-5/source<1e-4; exact balanced same-token different-document donor constraints',pred_b_absolute='natural/hybrid<=.15 and change<=.20 everydomain/cohort/selection',pred_c_relative='every graph effecterror<=1.10 separatebaseline'),selections=['mode1','mode2','mode3','combined'],source_products={'graph':512,'baseline':768},stored_float_coefficients={'graph':897804,'baseline_writer_deduplicated':897804},scope='Third new document-indexed panel,32FineWeb+16newcodefiles, context256. Coefficient-only primary selection before outcomes. Source substitution holds attention17 background fixed; native finalRMS andsoftcap retained. Nativeinputproducers remain dependencies.')
(P/'PARTIAL_GRAPH_FRESH_PLAN_V1.json').write_text(json.dumps(plan,indent=2)+'\n');print(stats)
