"""Compare exact and unbiased sampled coefficient objectives at paired initialization."""
import itertools,json,time,math
from pathlib import Path
import torch
from core import Model,terms
from sweep import target_cases
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);out=P/'STOCHASTIC_QUARTIC_TOYS_V1.json';assert not out.exists();start=time.perf_counter();d=6;canonical=terms(d,4);lookup={t:i for i,t in enumerate(canonical)};ordered=list(itertools.product(range(d),repeat=4));mapping=torch.tensor([lookup[tuple(sorted(t))] for t in ordered]);multiplicity=torch.bincount(mapping,minlength=len(canonical)).double();patterns=[''.join(map(str,sorted([t.count(x) for x in set(t)],reverse=True))) for t in ordered];keys=['4','31','22','211','1111'];groups=[torch.tensor([i for i,k in enumerate(patterns) if k==key]) for key in keys]
 cases=[x for x in target_cases() if x['degree']==4];spike=torch.zeros(3,len(canonical));spike[0,0]=1.;cases.append(dict(name='coordinate_fourth_power',kind='dag',width=1,target=spike));records=[]
 for case,mode,optname,seed in itertools.product(cases,['exact','uniform','stratified'],['adam','muon'],[0,1]):
  torch.manual_seed(seed);model=Model(d,3,case['kind'],case['width'],4);target=case['target'];den=(target.square()/multiplicity).sum();opt=torch.optim.Adam(model.parameters(),lr=.05) if optname=='adam' else torch.optim.Muon(model.parameters(),lr=.05,weight_decay=0.,adjust_lr_fn='match_rms_adamw');empty=0;history=[]
  for step in range(1200):
   opt.zero_grad();coeff=model();delta=coeff-target
   if mode=='exact':loss=(delta.square()/multiplicity).sum()/den
   else:
    if mode=='uniform':ids=torch.randint(len(ordered),(128,));weights=torch.full((128,),len(ordered)/128)
    else:
     selected=[];ww=[]
     for i,g in enumerate(groups):
      n=26 if i<3 else 25;selected.append(g[torch.randint(len(g),(n,))]);ww.append(torch.full((n,),len(g)/n))
     ids=torch.cat(selected);weights=torch.cat(ww)
    slots=mapping[ids];entry=delta[:,slots]/multiplicity[slots];loss=(entry.square().sum(0)*weights).sum()/den;empty+=int(bool((target[:,slots]==0).all()))
   if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite loss')
   if step%200==0:history.append(dict(step=step,exact_relative_error=float(((delta.detach().square()/multiplicity).sum()/den).sqrt())))
   loss.backward();opt.step()
   for group in opt.param_groups:group['lr']=.05*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/1200)))
  with torch.no_grad():error=float((((model()-target).square()/multiplicity).sum()/den).sqrt())
  row=dict(case=case['name'],mode=mode,optimizer=optname,seed=seed,final_frobenius_error=error,empty_teacher_batches=empty,steps=1200,history=history);records.append(row);print(case['name'],mode,optname,seed,error,empty,flush=True)
  out.write_text(json.dumps(dict(records=records,seconds=time.perf_counter()-start,scope='36 paired-initialization toy fits, exact teacher denominator; final checkpoints evaluated exactly. Small canonical materialization only.'),indent=2)+'\n')
if __name__=='__main__':main()
