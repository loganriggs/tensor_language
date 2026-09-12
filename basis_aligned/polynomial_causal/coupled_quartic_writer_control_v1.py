from pathlib import Path
import json,torch
from coupled_quartic_writer_v1 import gram,target_cross,solve
P=Path(__file__).resolve().parent
torch.set_default_dtype(torch.float64);torch.manual_seed(91601)
b=torch.randn(4,5,3);nu=torch.randn(4,3)
q=torch.einsum('kdi,ki,kei->kde',b,nu,b)
t=(torch.einsum('kab,kcd->kabcd',q,q)+torch.einsum('kac,kbd->kabcd',q,q)+torch.einsum('kad,kbc->kabcd',q,q))/3
kd=t.flatten(1)@t.flatten(1).T;k=gram(b,nu)
true=torch.randn(4,2);target=torch.einsum('ko,kabcd->oabcd',true,t)
def oracle(x):return torch.einsum('oabcd,na,nb,nc,nd->no',target,x[:,0],x[:,1],x[:,2],x[:,3])
c=target_cross(b,nu,oracle);cd=t.flatten(1)@target.flatten(1).T
w,diagnostics=solve(k,c)
errors=[float((k-kd).norm()/kd.norm()),float((c-cd).norm()/cd.norm()),float((w-true).norm()/true.norm())]
result=dict(pred_a=max(errors)<=1e-10,relative_errors=errors,diagnostics=diagnostics)
path=P/'COUPLED_QUARTIC_WRITER_V1_CONTROL.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n');print(result);assert result['pred_a']
