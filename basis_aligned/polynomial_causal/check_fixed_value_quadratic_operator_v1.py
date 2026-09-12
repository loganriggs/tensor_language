"""Independent dense cubic oracle for fixed-value weighted quadratic SVD.
A metric/adjoint<=1e-12; B truncated SVD reconstruction error equals singular
value tail<=1e-12; C polynomial execution<=1e-12. No native result inferred.
"""
import torch,json
from pathlib import Path
from fixed_value_quadratic_operator_v1 import metric_power,apply,adjoint
P=Path(__file__).resolve().parent

def basis(n):
 rows=[]
 for i in range(n):
  for j in range(i,n):
   z=torch.zeros(n,n);z[i,j]=1 if i==j else 2**-.5
   if i!=j:z[j,i]=2**-.5
   rows.append(z)
 return torch.stack(rows)

def cubic(x,v):return (torch.einsum('i,jk->ijk',v,x)+torch.einsum('j,ik->ijk',v,x)+torch.einsum('k,ij->ijk',v,x))/3

def main():
 torch.set_default_dtype(torch.float64);torch.manual_seed(174);n=5;q=3;a=torch.randn(q,n);b=torch.randn(q,n);v=torch.randn(n);sb=basis(n);qb=basis(q);x=torch.randn(n,n);x=(x+x.T)/2;y=torch.randn(q,q);y=(y+y.T)/2
 lifted=torch.stack([cubic(z,v).flatten() for z in sb]);gram=lifted@lifted.T;expected=torch.einsum('aij,bij->ab',sb,metric_power(sb,v,1));me=float((gram-expected).norm()/gram.norm());ae=float(abs((apply(x,a,b,v)*y).sum()-(x*adjoint(y,a,b,v)).sum()))
 op=torch.stack([torch.einsum('qij,ij->q',qb,apply(z,a,b,v)) for z in sb],1);u,s,vh=torch.linalg.svd(op,full_matrices=False);rank=2;raw=metric_power(torch.einsum('rs,sij->rij',vh[:rank],sb),v,-.5);query=torch.einsum('qr,qij->rij',u[:,:rank]*s[:rank],qb)
 # Independently construct query/source cubic coefficients from the native equation.
 target=torch.stack([cubic((a.T@z@b+b.T@z@a)/2,v).flatten() for z in qb]);approx=torch.einsum('qr,ri->qi',u[:,:rank]*s[:rank],torch.stack([cubic(z,v).flatten() for z in raw]));err=(target-approx).square().sum();tail=s[rank:].square().sum();te=float(abs(err-tail)/target.square().sum())
 xx=torch.randn(20,n);qq=torch.randn(20,q);direct=torch.einsum('ni,nj,nk,qijk,nq->n',xx,xx,xx,approx.reshape(len(qb),n,n,n),torch.einsum('ni,qij,nj->nq',qq,qb,qq));factored=(xx@v)*(torch.einsum('ni,rij,nj->nr',xx,raw,xx)*torch.einsum('ni,rij,nj->nr',qq,query,qq)).sum(-1);pe=float((direct-factored).norm()/direct.norm())
 result=dict(pred_a=max(me,ae)<=1e-12,pred_b=te<=1e-12,pred_c=pe<=1e-12,metric_relative_error=me,adjoint_absolute_error=ae,svd_tail_relative_error=te,polynomial_execution_relative_error=pe,scope='Independent small dense symmetric cubic oracle. Fixed-value family, rank2 separated query/source quadratics; global matrix approximation within this family only. No normalization/input closure or native effect evidence.')
 (P/'FIXED_VALUE_QUADRATIC_OPERATOR_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
