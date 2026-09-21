"""Five planted source structures under exact joint quartic coefficient loss.
40fits:Adam/Muon x rates.01/.05 x2starts x5structures,400steps.
Recovery bar relative coefficient norm<1e-3 in>=4/5structures by somefit.
Toy only; no native effect or wholemodel objective claim.
"""
from pathlib import Path
import torch,json,time,math
from quartic_pair_metric import numerator_forms,inner,squared_error
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.manual_seed(2500);d=6;r=2;start=time.perf_counter();base=torch.randn(d,r,dtype=torch.float64)/d**.5;other=torch.randn(d,r,dtype=torch.float64)/d**.5
positive=torch.ones(r,dtype=torch.float64);signed=torch.tensor([1.,-1.],dtype=torch.float64)
cases=[('independent_psd',base,other,positive,positive),('signed',base,other,signed,signed),('shared_subspace',base,base@torch.tensor([[.8,.3],[-.2,1.1]],dtype=torch.float64),signed,positive),('opposite_forms',base,base,positive,-positive),('separated_scale',base*10**.5,other/10**.5,positive,positive)]
records=[];saved={}
for name,trueU,trueV,signA,signB in cases:
 Qa=(trueU*signA)@trueU.T;Qb=(trueV*signB)@trueV.T;A,B=numerator_forms(Qa,Qb,.7,-.4);norm=inner(A,B,A,B);casebest=None
 for optimizer in ['adam','muon']:
  for lr in [.01,.05]:
   for seed in [0,1]:
    gen=torch.Generator().manual_seed(22500+seed);U=torch.nn.Parameter(torch.randn(d,r,dtype=torch.float64,generator=gen)/d**.5);V=torch.nn.Parameter(torch.randn(d,r,dtype=torch.float64,generator=gen)/d**.5);opt=torch.optim.Adam([U,V],lr=lr) if optimizer=='adam' else torch.optim.Muon([U,V],lr=lr,weight_decay=0.,adjust_lr_fn='match_rms_adamw');best=None
    for step in range(401):
     qa=(U*signA)@U.T;qb=(V*signB)@V.T;C,D=numerator_forms(qa,qb,.7,-.4);loss=squared_error(A,B,C,D)/norm;value=float(loss.detach())
     if best is None or value<best[0]:best=(value,step,U.detach().clone(),V.detach().clone())
     if step==400:break
     opt.zero_grad();loss.backward();opt.step()
     for group in opt.param_groups:group['lr']=lr*(.05+.95*.5*(1+math.cos(math.pi*(step+1)/400)))
    value,step,bu,bv=best;ra=(bu*signA)@bu.T;rb=(bv*signB)@bv.T;row=dict(structure=name,optimizer=optimizer,learning_rate=lr,seed=seed,relative_coefficient_error=max(value,0.)**.5,best_step=step,source_matrix_errors=[float((ra-Qa).norm()/Qa.norm()),float((rb-Qb).norm()/Qb.norm())]);records.append(row)
    if casebest is None or value<casebest[0]:casebest=(value,bu,bv)
 saved[name]=dict(U=casebest[1],V=casebest[2],signA=signA,signB=signB,Qa=Qa,Qb=Qb)
 print(name,'best',min(x['relative_coefficient_error'] for x in records if x['structure']==name),flush=True)
summary={name:{opt:min(x['relative_coefficient_error'] for x in records if x['structure']==name and x['optimizer']==opt) for opt in ['adam','muon']} for name,*_ in cases};recovered=sum(min(v.values())<1e-3 for v in summary.values());out=dict(predictions=dict(pred_a_recovery=recovered>=4),summary=summary,records=records,recovered_structures=recovered,seconds=time.perf_counter()-start,scope='Known expressive rank2source families; exact symmetrized quartic numerator coefficient objective with fixed t/s/u terms.400step budget,2starts; failure does not prove impossibility or universal optimizerinferiority. No normalization denominator/data/nativebehavior fit.');(p/'QUARTIC_NUMERATOR_TOY_FITS_V1.json').write_text(json.dumps(out,indent=2)+'\n');torch.save(saved,p/'QUARTIC_NUMERATOR_TOY_FITS_V1.pt');print(json.dumps({k:v for k,v in out.items() if k!='records'},indent=2))
