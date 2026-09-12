from pathlib import Path
import torch,json
from quadratic_product_core_v1 import gram,target_cross,pairs,features,quadratic_values
from coupled_quartic_writer_v1 import solve
P=Path(__file__).resolve().parent
torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.manual_seed(91710)
b=torch.randn(3,5,2);nu=torch.randn(3,2);q=torch.einsum('kdi,ki,kei->kde',b,nu,b);ij=pairs(3);ts=[]
for i,j in ij.T:
 a=q[i];c=q[j]
 t=(torch.einsum('ab,cd->abcd',a,c)+torch.einsum('ac,bd->abcd',a,c)+torch.einsum('ad,bc->abcd',a,c)+torch.einsum('ab,cd->abcd',c,a)+torch.einsum('ac,bd->abcd',c,a)+torch.einsum('ad,bc->abcd',c,a))/6
 ts.append(t)
t=torch.stack(ts);kd=t.flatten(1)@t.flatten(1).T;k=gram(b,nu)
# Target is precisely the off-diagonal q0*q1 interaction.
target=t[1:2]
def oracle(x):return torch.einsum('oabcd,na,nb,nc,nd->no',target,x[:,0],x[:,1],x[:,2],x[:,3])
c=target_cross(b,nu,oracle);cd=t.flatten(1)@target.flatten(1).T;w,diag=solve(k,c)
error=float((t.flatten(1).T@w-target.flatten(1).T).norm()/target.norm())
diagonal=ij[0]==ij[1];ws,_=solve(k[diagonal][:,diagonal],c[diagonal]);square_error=float((t[diagonal].flatten(1).T@ws-target.flatten(1).T).norm()/target.norm())
x=torch.randn(9,5);z=quadratic_values(b,nu,x);base=features(b,nu,x)@w
z0=z.clone();z0[:,0]=0;z1=z.clone();z1[:,1]=0;z01=z.clone();z01[:,:2]=0
execute=lambda z:(z[:,ij[0]]*z[:,ij[1]])@w
cross=execute(z01)-execute(z0)-execute(z1)+base
expected=z[:,0:1]*z[:,1:2]*w[1]
errs=[float((k-kd).norm()/kd.norm()),float((c-cd).norm()/cd.norm()),float((cross-expected).norm()/expected.norm())]
r=dict(pred_a=max(errs[:2])<=1e-10,pred_b=error<=1e-10 and square_error>=.1,pred_c=errs[2]<=1e-10,identity_errors=errs,mixed_relative_error=error,square_only_relative_error=square_error,normal_solve=diag)
p=P/'QUADRATIC_PRODUCT_CORE_V1_CONTROL.json';assert not p.exists();p.write_text(json.dumps(r,indent=2)+'\n');print(r);assert r['pred_a'] and r['pred_b'] and r['pred_c']
