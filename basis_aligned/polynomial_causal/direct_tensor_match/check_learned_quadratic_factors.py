"""Five distinct planted structures, random starts, joint-direction Adam/Muon sweep."""
from pathlib import Path
import json,time,torch
from learned_quadratic_factors import fit,controls,dense
p=Path(__file__).resolve().parent;out=p/'LEARNED_FULL_QUADRATIC_TOYS_V1.json';assert not out.exists();torch.set_num_threads(2);start=time.monotonic();check=controls();records=[]
for index,name in enumerate(['independent_products','shared_input','shared_output','squares','cancelling_terms']):
 torch.manual_seed(950+index);d,k,V=8,4,5;rand=lambda *s:torch.randn(*s,dtype=torch.float64);a,b,c=rand(k,d),rand(k,d),rand(V,k)
 if name=='shared_input':a=a[:1].expand(k,-1).clone()
 if name=='shared_output':c=c[:,:1]@rand(1,k)
 if name=='squares':b=a.clone()
 if name=='cancelling_terms':
  aa,bb,cc=rand(2,d),rand(2,d),rand(V,2);a=torch.cat([a,aa,aa]);b=torch.cat([b,bb,bb]);c=torch.cat([c,cc,-cc],1)
 an,bn=a.norm(dim=1),b.norm(dim=1);a/=an[:,None];b/=bn[:,None];c=c*an*bn;c/=dense(c,a,b).norm();x,v=rand(96,d),rand(96,d)
 for optimizer in ('adam','muon'):
  for lr in (.01,.05):
   for seed in (0,1):
    result,_=fit((c,a,b),k,optimizer,lr,960+seed,600,x,v);result.update(family=name,optimizer=optimizer,lr=lr,seed=seed);records.append(result);print(json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)
configs=[]
for optimizer in ('adam','muon'):
 for lr in (.01,.05):
  rr=[r for r in records if r['optimizer']==optimizer and r['lr']==lr];configs.append(dict(optimizer=optimizer,lr=lr,worst_coefficient_error=max(r['coefficient_error'] for r in rr),mean_coefficient_error=sum(r['coefficient_error'] for r in rr)/len(rr),individual_recoveries=sum(r['coefficient_error']<.01 and r['response_error']<.01 for r in rr)))
selected=min(configs,key=lambda r:(r['worst_coefficient_error'],r['mean_coefficient_error']));pred=dict(pred_a_instrument=max(check.values())<1e-12,pred_b_family_recovery=all(any(r['coefficient_error']<.01 and r['response_error']<.01 for r in records if r['family']==name) for name in set(r['family'] for r in records)),pred_c_common_configuration=selected['individual_recoveries']==10)
result=dict(predictions=pred,records=records,controls=check,configurations=configs,selected=selected,seconds=time.monotonic()-start,scope='Five structural families; random initial student at planted minimal width4, teacher may have redundant cancelling terms. Exact symmetric coefficient plus artificial directional-response objective. Selection minimizes worst then mean coefficient error across all10family/start cases. Functional recovery does not identify unique factors or prove native optimizer superiority.')
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
