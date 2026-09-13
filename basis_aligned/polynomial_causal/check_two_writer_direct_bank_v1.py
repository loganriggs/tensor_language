"""Actual-weight composition and exact polynomial-rank checks, CPU only."""
from pathlib import Path
import json,torch,sympy as sp
from head17_source_interface_v1 import CHECKPOINT
from two_writer_rational_basis_v1 import coefficient_system,scalar_functions
from two_writer_direct_bank_v1 import prepare as direct_bank
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);torch.manual_seed(7131101)
 generic,products,_=coefficient_system({(0,0):3,(1,0):-2,(0,1):4,(2,0):2,(1,1):2,(0,2):5})
 exactrank=sp.Matrix(products.tolist()).applyfunc(sp.Rational).rank();basisrank=sp.Matrix(generic.tolist()).applyfunc(sp.Rational).rank();assert exactrank==basisrank==19
 sd=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu');w=torch.load(P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1/program.pt',weights_only=True)['writers'].double();w=w/w.square().mean(-1,keepdim=True).sqrt()
 l,r,d=[sd['transformer.h.9.mlp.'+k+'.weight'].double() for k in ['Left','Right','Down']];ll,rr,dd=[sd['transformer.h.10.mlp.'+k+'.weight'].double() for k in ['Left','Right','Down']];lam=float(sd['transformer.h.10.lambdas'][0]);eps=torch.finfo(torch.float32).eps
 def B(x):return ((x@l.T)*(x@r.T))@d.T
 def K(x,y,L=l,R=r,D=d):return ((x@L.T)*(y@R.T)+(y@L.T)*(x@R.T))@D.T
 gamma=w@w.T/1152;rows=[]
 for context in range(4):
  z=torch.randn(1152,dtype=torch.float64);rho0=z.square().mean()+eps;beta=w@z/1152;m0=B(z)/rho0
  p=torch.stack([K(wi,z)-2*bi*m0 for wi,bi in zip(w,beta)])
  q=torch.stack([K(w[i],w[j])-2*gamma[i,j]*m0 for i,j in [(0,0),(0,1),(1,1)]])
  vectors=lam*torch.cat([w,p,q]);rho={(0,0):float(rho0),(1,0):float(-2*beta[0]),(0,1):float(-2*beta[1]),(2,0):float(gamma[0,0]),(1,1):float(2*gamma[0,1]),(0,2):float(gamma[1,1])}
  G,C,pairs=coefficient_system(rho);transform=torch.linalg.lstsq(G,C).solution;polyerror=float((G@transform-C).norm()/C.norm())
  bank=torch.stack([K(vectors[i],vectors[j],ll,rr,dd) for i,j in pairs]);small=transform@bank
  # Fold coefficient combinations before the final Down map as a second schedule.
  left=vectors@ll.T;right=vectors@rr.T
  hidden=torch.stack([left[i]*right[j]+left[j]*right[i] for i,j in pairs]);direct_small=(transform@hidden)@dd.T
  bankerror=float((small-direct_small).norm()/small.norm());analytic=direct_bank(vectors,ll,rr,dd,rho0.reshape(1),beta,gamma);analytic_error=float((analytic-small).norm()/small.norm());assert analytic_error<1e-10;small=analytic;maximum=0.;response_error=0.
  for av,bv in [(a,b) for a in [-.7,0.,.3,1.2] for b in [-.4,0.,.2,1.]]:
   a=torch.tensor(av,dtype=torch.float64);b=torch.tensor(bv,dtype=torch.float64);new=z-a*w[0]-b*w[1];den=new.square().mean()+eps
   u=torch.stack([-a,-b,-a/den,-b/den,a*a/(2*den),a*b/den,b*b/(2*den)])
   response=u@vectors;actual=lam*(new+B(new)/den-z-m0)
   if av or bv:
    response_error=max(response_error,float((response-actual).norm()/actual.norm()))
    reference=K(response,response,ll,rr,dd);pred=scalar_functions(a,b,den)@small;maximum=max(maximum,float((pred-reference).norm()/reference.norm()))
  rows.append(dict(context=context,coefficient_identity_error=polyerror,direct_bank_error=analytic_error,response_identity_error=response_error,product_identity_error=maximum,fold_before_down_error=bankerror))
 result={'pred_a':exactrank==basisrank==19,'pred_b':max(r['response_identity_error'] for r in rows)<1e-10,'pred_c':max(r['product_identity_error'] for r in rows)<1e-10,'rows':rows,'exact_polynomial_rank':exactrank,'basis_rank':basisrank,'writer_cosine':float((w[0]@w[1])/(w[0].norm()*w[1].norm())),'old_products':28,'new_products':19,'scope':'Actual head8.2/head9.8 writer directions normalized and transplanted to a common MLP9 input for algebra only, four synthetic contexts, actual MLP9/10 weights, signed two-writer strengths. Exact self-product within one combined branch, not arbitrary independent cross-response product. No native behavioral test or semantic interpretation.'}
 (P/'TWO_WRITER_DIRECT_BANK_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
