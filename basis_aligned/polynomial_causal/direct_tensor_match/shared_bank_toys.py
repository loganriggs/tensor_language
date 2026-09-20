import itertools,json,math,time
from pathlib import Path
import torch
from shared_quadratic_bank import normalize_bank,bank_gram,native_bank_cross,roots
P=Path(__file__).resolve().parent

def teacher_for(family):
 torch.manual_seed(1751);d=6;b=3;k=2;U=torch.randn(b,k,d);V=torch.randn(b,k,d)
 if family in ['coordinate','rotated_signed']:
  U.zero_();V.zero_()
  for i in range(b):
   U[i,0,2*i]=1;V[i,0,2*i]=1;U[i,1,2*i+1]=.7;V[i,1,2*i+1]=.7 if family=='coordinate' else -.7
  if family=='rotated_signed':
   rotation=torch.linalg.qr(torch.randn(d,d))[0];U=U@rotation;V=V@rotation
 U,V=normalize_bank(U,V);i,j=roots(b,U.device);C=torch.zeros(3,len(i));pairs=list(zip(i.tolist(),j.tolist()))
 if family=='dense':C=torch.randn_like(C)
 else:
  selected=[(0,1),(0,2),(1,2)] if family in ['coordinate','rotated_signed'] else ([(0,0),(0,1),(0,2)] if family=='common_factor' else [(0,0),(1,1),(2,2)])
  for o,pair in enumerate(selected):C[o,pairs.index(pair)]=[1.,.8,.6][o]
 C/=((C.T@C)*bank_gram(U,V)).sum().sqrt();D=torch.zeros(b,b*k)
 for a in range(b):D[a,a*k:(a+1)*k]=1
 teacher=[C,torch.eye(b)[i],torch.eye(b)[j],D,U.reshape(b*k,d),V.reshape(b*k,d)]
 return teacher,U,V

def main():
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);out=P/'SHARED_BANK_TOYS_V1.json';assert not out.exists();rows=[];start=time.perf_counter()
 for family in ['coordinate','rotated_signed','dense','common_factor','squares']:
  teacher,tU,tV=teacher_for(family);tQ=(tU.transpose(-1,-2)@tV+tV.transpose(-1,-2)@tU)/2;tspan=torch.linalg.qr(tQ.flatten(1).T)[0]
  for optimizer,lr,seed in itertools.product(['adam','muon'],[.005,.05],[0,1]):
   torch.manual_seed(seed);us=[torch.nn.Parameter(torch.randn(2,6)/6**.5) for _ in range(3)];vs=[torch.nn.Parameter(torch.randn(2,6)/6**.5) for _ in range(3)];params=us+vs;opt=torch.optim.Adam(params,lr=lr) if optimizer=='adam' else torch.optim.Muon(params,lr=lr,weight_decay=0.,adjust_lr_fn='match_rms_adamw');best=float('inf');base=None;history=[]
   for step in range(601):
    opt.zero_grad();U,V=normalize_bank(torch.stack(us),torch.stack(vs));G=bank_gram(U,V);cross=native_bank_cross(teacher,U,V);C=torch.linalg.solve(G+1e-8*G.diag().mean()*torch.eye(len(G)),cross.T).T;loss=((C@G)*C).sum()-2*(cross*C).sum();e=float((1+loss).detach())
    if e<best:best=e;beststep=step;bu=U.detach().clone();bv=V.detach().clone();bc=C.detach().clone()
    if base is None:base=max(float(-loss.detach()),1e-20)
    if step%100==0:history.append(dict(step=step,signed_squared_error=e,condition=float(torch.linalg.cond(G).detach())))
    if step==600:break
    (loss/base).backward();opt.step()
    for group in opt.param_groups:group['lr']=lr*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/600)))
   q=(bu.transpose(-1,-2)@bv+bv.transpose(-1,-2)@bu)/2;span=torch.linalg.qr(q.flatten(1).T)[0];cos=torch.linalg.svdvals(tspan.T@span);row=dict(family=family,optimizer=optimizer,lr=lr,seed=seed,error=math.sqrt(max(0,best)),signed_squared_error=best,selected_step=beststep,span_principal_cosines=cos.tolist(),history=history,U=bu.tolist(),V=bv.tolist(),C=bc.tolist());rows.append(row);print(family,optimizer,lr,seed,row['error'],float(cos.min()),flush=True);out.write_text(json.dumps(dict(records=rows,seconds=time.perf_counter()-start,scope='Five planted low-rank sharedquadratic dictionaries,40fits. Exact coefficient objective; dense fitted outputcore, no sparsity/individualidentity/nativeclaim.'),indent=2)+'\n')
if __name__=='__main__':main()
