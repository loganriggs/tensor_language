"""Restricted stage-two graph edit: delete one shared product, refit, accept by error.
Five width6 discoveries from registered runs; refit every possible deletion for
400stepsAdam.05, targetwidth4. Accept only if coefficient AND response error<1%.
Pred allfive retain fidelity and at least4/5 reachwidth4. No claimed unique factors.
"""
from pathlib import Path
import math,json,time,torch
from learned_quadratic_factors import dense
from profiled_learned_quadratic_factors import fit
p=Path(__file__).resolve().parent;torch.set_num_threads(2);start=time.monotonic();prior=json.loads((p/'PROFILED_FULL_QUADRATIC_WIDTH6_V1.json').read_text());records=[]
for index,name in enumerate(['independent_products','shared_input','shared_output','squares','cancelling_terms']):
 torch.manual_seed(950+index);d,k,V=8,4,5;rand=lambda *s:torch.randn(*s,dtype=torch.float64);a,b,c=rand(k,d),rand(k,d),rand(V,k)
 if name=='shared_input':a=a[:1].expand(k,-1).clone()
 if name=='shared_output':c=c[:,:1]@rand(1,k)
 if name=='squares':b=a.clone()
 if name=='cancelling_terms':
  aa,bb,cc=rand(2,d),rand(2,d),rand(V,2);a=torch.cat([a,aa,aa]);b=torch.cat([b,bb,bb]);c=torch.cat([c,cc,-cc],1)
 an,bn=a.norm(dim=1),b.norm(dim=1);a/=an[:,None];b/=bn[:,None];c=c*an*bn;c/=dense(c,a,b).norm();x,v=rand(96,d),rand(96,d);old=min([r for r in prior['records'] if r['family']==name],key=lambda r:r['coefficient_error']**2+r['response_error']**2);result,factors=fit((c,a,b),6,'adam',.05,960+old['seed'],3000,x,v);assert abs(result['coefficient_error']-old['coefficient_error'])<1e-5
 width=6;history=[]
 while width>4:
  candidates=[]
  for delete in range(width):
   ids=[i for i in range(width) if i!=delete];cc,aa,bb=factors;initial=(cc[:,ids],aa[ids]*math.sqrt(d),bb[ids]*math.sqrt(d));r,f=fit((c,a,b),width-1,'adam',.05,990+delete,400,x,v,initial);candidates.append((r['coefficient_error']**2+r['response_error']**2,r,f,delete))
  candidates.sort(key=lambda t:t[0]);_,candidate,updated,deleted=candidates[0];accepted=candidate['coefficient_error']<.01 and candidate['response_error']<.01;history.append(dict(proposed_width=width-1,deleted_product=deleted,accepted=accepted,candidates=[dict(deleted=t[3],coefficient_error=t[1]['coefficient_error'],response_error=t[1]['response_error']) for t in candidates]))
  if not accepted:break
  width-=1;result,factors=candidate,updated
 records.append(dict(family=name,initial_width=6,final_width=width,coefficient_error=result['coefficient_error'],response_error=result['response_error'],stored_coefficients=(2*d+V)*width,initial_stored_coefficients=(2*d+V)*6,history=history,factors=[t.tolist() for t in factors]));print(json.dumps({k:v for k,v in records[-1].items() if k not in ('history','factors')}),flush=True)
pred=dict(pred_a_fidelity=all(r['coefficient_error']<.01 and r['response_error']<.01 for r in records),pred_b_simplicity=sum(r['final_width']==4 for r in records)>=4)
result=dict(predictions=pred,records=records,seconds=time.monotonic()-start,scope='Restricted graph search over shared-product deletion plus continuous refitting. Starting wide decompositions selected by fitting loss among previously registered random starts; replay regenerates factors not retained in old JSON. Products shared across outputs counted once, all dense input/output coefficients counted. No new-data/native semantic or arbitrary-DAG claim.')
(p/'QUADRATIC_DELETE_REFIT_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(predictions=pred,seconds=result['seconds'])))
