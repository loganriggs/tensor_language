"""CPU precision screen; no native intervention or behavioral claim."""
from pathlib import Path
import json,torch
from head17_source_interface_v1 import CHECKPOINT
from two_writer_direct_bank_v1 import prepare
from two_writer_rational_basis_v1 import scalar_functions
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);torch.manual_seed(7131107)
 sd=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu')
 l,r,d=[sd['transformer.h.9.mlp.'+k+'.weight'].double() for k in ['Left','Right','Down']]
 ll,rr,dd=[sd['transformer.h.10.mlp.'+k+'.weight'].double() for k in ['Left','Right','Down']]
 def K(x,y,L=l,R=r,D=d):return ((x@L.T)*(y@R.T)+(y@L.T)*(x@R.T))@D.T
 w0=torch.randn(1152,dtype=torch.float64);w0/=w0.square().mean().sqrt()
 orth=torch.randn_like(w0);orth-=w0*(orth@w0)/(w0@w0);orth/=orth.square().mean().sqrt()
 z=torch.randn_like(w0);rho0=z.square().mean()+torch.finfo(torch.float32).eps;m0=K(z,z)/2/rho0;rows=[]
 for cosine in [.9,.999999,1.]:
  w=torch.stack([w0,cosine*w0+(1-cosine*cosine)**.5*orth]);beta=w@z/1152;gamma=w@w.T/1152
  p=torch.stack([K(wi,z)-2*bi*m0 for wi,bi in zip(w,beta)])
  q=torch.stack([K(w[i],w[j])-2*gamma[i,j]*m0 for i,j in [(0,0),(0,1),(1,1)]])
  v=torch.cat([w,p,q]);bank=prepare(v.float(),ll.float(),rr.float(),dd.float(),rho0.reshape(1).float(),beta.float(),gamma.float())
  scale=float(K(v[0],v[0],ll,rr,dd).norm())
  for av,bv in [(1.,1.),(1.,-1.),(.3,-.7),(2.,-2.)]:
   a=torch.tensor(av,dtype=torch.float64);b=torch.tensor(bv,dtype=torch.float64);den=rho0-2*(beta[0]*a+beta[1]*b)+gamma[0,0]*a*a+2*gamma[0,1]*a*b+gamma[1,1]*b*b
   u=torch.stack([-a,-b,-a/den,-b/den,a*a/(2*den),a*b/den,b*b/(2*den)])
   resp=u@v;ref=K(resp,resp,ll,rr,dd)
   predicted=scalar_functions(a.float(),b.float(),den.float())@bank
   rf=u.float()@v.float();direct=K(rf,rf,ll.float(),rr.float(),dd.float())
   norm=float(ref.norm());err=float((predicted.double()-ref).norm());derr=float((direct.double()-ref).norm())
   rows.append(dict(cosine=cosine,a=av,b=bv,reference_norm=norm,constituent_scale=scale,material=norm>=1e-6*scale,bank_relative_error=err/norm if norm else None,direct_relative_error=derr/norm if norm else None,bank_scaled_absolute_error=err/scale,direct_scaled_absolute_error=derr/scale))
 result={'pred_a':all(x['bank_relative_error']<=1e-3 for x in rows if x['material']),'rows':rows,'scope':'Actual MLP9/10 weights, synthetic common-input writer directions/context. FP32 bank and response contractions against FP64; not native BF16 or behavioral validation.'}
 (P/'TWO_WRITER_PRECISION_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
