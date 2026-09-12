"""Dense oracle, identity trace and selfadjointness checks for weighted trace."""
from pathlib import Path
import itertools,json
import torch
from quartic_weighted_trace_v1 import producer_core,native_action,fitted_action
from composed_quartic_contraction_v1 import contract
from quartic_repeated_input_v1 import native_traces
torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(91881)
d=5;l,r=torch.randn(2,4,d);down=torch.randn(3,4);l1,r1=torch.randn(2,6,3);out=torch.randn(1,6);scale=.7
indices=torch.tensor(list(itertools.product(range(d),repeat=4)));slots=torch.eye(d)[indices]
t=contract(slots,l,r,down,l1,r1,out,scale).reshape(d,d,d,d)
h=producer_core(down,l1,r1,out[0],scale)
q,z=torch.randn(2,d,d);q=(q+q.T)/2;z=(z+z.T)/2
actual=native_action(l,r,h,q);expected=torch.einsum('ijab,ab->ij',t,q)
native_error=float((actual-expected).norm()/expected.norm())
identity=native_traces(l,r,down,l1,r1,out,scale)[0]
identity_error=float((native_action(l,r,h,torch.eye(d))-identity).norm()/identity.norm())
adjoint=abs(float((actual*z).sum()-(q*native_action(l,r,h,z)).sum()))/max(1.,float(actual.norm()*z.norm()))
b=torch.linalg.qr(torch.randn(3,d,2),mode='reduced')[0];n=torch.randn(3,2);mix=torch.randn(3)
matrices=torch.einsum('kdi,ki,kei->kde',b,n,b)
f=(torch.einsum('kij,kab->kijab',matrices,matrices)+torch.einsum('kia,kjb->kijab',matrices,matrices)+torch.einsum('kib,kja->kijab',matrices,matrices))/3
ft=torch.einsum('k,kijab->ijab',mix,f)
fexpected=torch.einsum('ijab,ab->ij',ft,q);factual=fitted_action(b,n,mix,q)
fitted_error=float((factual-fexpected).norm()/fexpected.norm())
result=dict(pred_a=max(native_error,identity_error,adjoint,fitted_error)<=1e-10,native_error=native_error,identity_error=identity_error,selfadjoint_error=adjoint,fitted_error=fitted_error)
path=Path(__file__).resolve().parent/'QUARTIC_WEIGHTED_TRACE_V1_CONTROL.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n');print(result);assert result['pred_a']
