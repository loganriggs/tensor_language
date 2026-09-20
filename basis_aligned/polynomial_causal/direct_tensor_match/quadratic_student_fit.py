"""Exact quadratic coefficient contractions for small student against wide teacher."""
import torch

def gram(a,b):
 return .5*((a@a.T)*(b@b.T)+(a@b.T)*(b@a.T))

def cross(C,A,B,a,b):
 return .5*C@((A@a.T)*(B@b.T)+(A@b.T)*(B@a.T))

def normalize(a,b):
 norm2=.5*(a.square().sum(1)*b.square().sum(1)+(a*b).sum(1).square());scale=norm2.clamp_min(1e-24).pow(.25)
 return a/scale[:,None],b/scale[:,None]

def check():
 import json
 from pathlib import Path
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(1906);C=torch.randn(3,7);A=torch.randn(7,5);B=torch.randn(7,5);a=torch.randn(2,5,requires_grad=True);b=torch.randn(2,5,requires_grad=True);M=torch.randn(5,5);L=torch.linalg.cholesky(M@M.T+.2*torch.eye(5));Aw,Bw,aw,bw=A@L,B@L,a@L,b@L
 T=.5*(torch.einsum('vk,ki,kj->vij',C,Aw,Bw)+torch.einsum('vk,kj,ki->vij',C,Aw,Bw));features=.5*(aw[:,:,None]*bw[:,None,:]+bw[:,:,None]*aw[:,None,:]);reference=T.flatten(1)@features.flatten(1).T;got=cross(C,Aw,Bw,aw,bw);G=gram(aw,bw);value=float(((got-reference).norm()/reference.norm()).detach());selferror=float(((G-features.flatten(1)@features.flatten(1).T).norm()/G.norm()).detach());g1=torch.autograd.grad(got.square().sum()+G.square().sum(),[a,b],retain_graph=True);g2=torch.autograd.grad(reference.square().sum()+(features.flatten(1)@features.flatten(1).T).square().sum(),[a,b]);gradient=max(float((x-y).norm()/y.norm()) for x,y in zip(g1,g2));assert max(value,selferror,gradient)<1e-12
 out=dict(cross_error=value,self_error=selferror,gradient_error=gradient,scope='Exact wide-teacher/small-student M-weighted quadratic contractions, independentlydensechecked. Nativecompactfitnotyetexecuted.');(Path(__file__).resolve().parent/'QUADRATIC_STUDENT_FIT_CHECK_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':check()

def fit(C,A,B,width,transform,optimizer='adam',lr=.005,seed=0,steps=400):
 """Original-coordinate readers; exact weighted coefficient self/cross objective.

 Analytic output solve uses FP64 and a 1e-6 relative diagonal ridge. Selection uses
 the unregularized reconstructed gain, not the ridge objective. Teacher norm is
 unnecessary for optimization or comparisons against other widths.
 """
 import math
 torch.manual_seed(seed);d=A.shape[1];device=A.device
 a=torch.nn.Parameter(torch.randn(width,d,device=device,dtype=A.dtype)/d**.5);b=torch.nn.Parameter(torch.randn_like(a)/d**.5)
 opt=torch.optim.Adam([a,b],lr=lr) if optimizer=='adam' else torch.optim.Muon([a,b],lr=lr,weight_decay=0.,adjust_lr_fn='match_rms_adamw')
 Aw,Bw=A@transform,B@transform;best=-float('inf');saved=None;base=None
 for step in range(steps+1):
  opt.zero_grad();u,v=normalize(a,b);uw,vw=u@transform,v@transform;G=gram(uw,vw).double();X=cross(C,Aw,Bw,uw,vw).double();writer=torch.linalg.solve(G+1e-6*G.diag().mean()*torch.eye(width,device=device,dtype=torch.float64),X.T).T
  loss=((writer@G)*writer).sum()-2*(writer*X).sum();gain=float(-loss.detach())
  if not math.isfinite(gain):raise FloatingPointError('nonfinite quadratic gain')
  if gain>best:best=gain;saved=(u.detach().clone(),v.detach().clone(),writer.detach().clone());selected=step
  if base is None:base=max(gain,1e-30)
  if step==steps:break
  (loss/base).backward();opt.step()
  for group in opt.param_groups:group['lr']=lr*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/steps)))
 return dict(gain=best,selected_step=selected,initial_gain=base),saved
