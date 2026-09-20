import itertools,json,math,time
from pathlib import Path
import torch
from quadratic_square import square_inner,symmetric_bilinear
P=Path(__file__).resolve().parent

def spectral_factors(Q,k):
 ev,V=torch.linalg.eigh(Q);pos=torch.where(ev>1e-10)[0].flip(0)[:k];neg=torch.where(ev< -1e-10)[0][:k];u=[];v=[]
 for j in range(max(len(pos),len(neg))):
  p=V[:,pos[j]]*ev[pos[j]].sqrt() if j<len(pos) else torch.zeros(len(Q));n=V[:,neg[j]]*(-ev[neg[j]]).sqrt() if j<len(neg) else torch.zeros(len(Q));u.append(p+n);v.append(p-n)
 return torch.stack(u,1),torch.stack(v,1)

def main():
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(1724);rotation=torch.linalg.qr(torch.randn(8,8))[0];spectra={'rank4_psd':[1,.8,.6,.4,0,0,0,0],'rank4_balanced':[1,.8,-.6,-.4,0,0,0,0],'rank4_unbalanced':[1,.8,.6,-.4,0,0,0,0],'full_balanced':[1,.8,.6,.4,-1,-.8,-.6,-.4],'full_psd':[1,.8,.6,.4,.2,.1,.05,.01]};out=P/'SIGNED_QUADRATIC_WIDTH_V1.json';assert not out.exists();rows=[];baselines=[];start=time.perf_counter()
 # Independent dense validation for indefinite noncommuting Q,S.
 raw=torch.randn(4,4,requires_grad=True);Q=(raw+raw.T)/2;raw2=torch.randn(4,4);S=(raw2+raw2.T)/2
 def dense(a):return (torch.einsum('ij,kl->ijkl',a,a)+torch.einsum('ik,jl->ijkl',a,a)+torch.einsum('il,jk->ijkl',a,a))/3
 got=square_inner(Q,S);ref=(dense(Q)*dense(S)).sum();g=torch.autograd.grad(got,raw,retain_graph=True)[0];h=torch.autograd.grad(ref,raw)[0];check=max(float((got-ref).abs().detach()),float((g-h).abs().max()));assert check<1e-11
 for name,spectrum in spectra.items():
  ev=torch.tensor(spectrum);teacher=(rotation*ev)@rotation.T;teacher/=teacher.norm();den=square_inner(teacher,teacher);minimum=max(int((ev>0).sum()),int((ev<0).sum()));u,v=spectral_factors(teacher,minimum);construction=float((symmetric_bilinear(u,v)-teacher).norm());assert construction<1e-11
  for k in [2,4,8]:
   u,v=spectral_factors(teacher,k);q=symmetric_bilinear(u,v);c=square_inner(q,teacher)/square_inner(q,q);err=float(1-c.square()*square_inner(q,q)/den);baselines.append(dict(family=name,width=k,minimum_exact_width=minimum,construction_error=construction,error=math.sqrt(max(0,err))))
   for optname,lr,seed in itertools.product(['adam','muon'],[.01,.05],[0,1]):
    torch.manual_seed(seed);U=torch.nn.Parameter(torch.randn(8,k));V=torch.nn.Parameter(torch.randn(8,k));opt=torch.optim.Adam([U,V],lr=lr) if optname=='adam' else torch.optim.Muon([U,V],lr=lr,weight_decay=0.,adjust_lr_fn='match_rms_adamw');best=float('inf');bestq=None
    for step in range(601):
     opt.zero_grad();q=symmetric_bilinear(U,V);q=q/q.norm();selfnorm=square_inner(q,q);cross=square_inner(q,teacher);c=cross/selfnorm;loss=-(cross.square()/selfnorm)/den;e=float((1+loss).detach())
     if e<best:best=e;bestq=q.detach().clone();bestc=float(c.detach());beststep=step
     if step==600:break
     loss.backward();opt.step()
     for group in opt.param_groups:group['lr']=lr*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/600)))
    cosine=float((bestq*teacher).sum().abs());row=dict(family=name,width=k,minimum_exact_width=minimum,optimizer=optname,lr=lr,seed=seed,error=math.sqrt(max(0,best)),signed_squared_error=best,selected_step=beststep,quadratic_abs_cosine=cosine,writer=bestc);rows.append(row)
   print(name,k,'best',min(r['error'] for r in rows if r['family']==name and r['width']==k),flush=True);out.write_text(json.dumps(dict(records=rows,baselines=baselines,validation_max_absolute_error=check,seconds=time.perf_counter()-start,scope='Known shared quadratic-square family with learned dense signed bilinear factors. Inertia bound applies exact quadratic representation; spectral truncation is only an achievable quartic baseline. Best-training checkpoint selection.'),indent=2)+'\n')
if __name__=='__main__':main()
