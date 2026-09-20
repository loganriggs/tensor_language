from pathlib import Path
import json,torch
from attention_reader_fold import fold,rms

def main():
 torch.set_num_threads(2);torch.manual_seed(1701);dtype=torch.float64
 b,t,d,h,o=2,5,12,3,4
 x=torch.randn(b,t,d,dtype=dtype,requires_grad=True);first=torch.randn_like(x)
 weights=[torch.randn(d,d,dtype=dtype)/d**.5 for _ in range(5)];out=torch.randn(d,d,dtype=dtype)/d**.5
 phase=torch.randn(t,1,d//h//2,dtype=dtype);cos,sin=phase.cos(),phase.sin();reader=torch.randn(b,o,t,d,dtype=dtype);eps=torch.finfo(torch.float32).eps
 parts,y=fold(x,first,weights,out,.3,cos,sin,h,reader,eps)
 oracle=torch.stack([torch.autograd.grad((y*reader[:,i]).sum(),x,retain_graph=True)[0] for i in range(o)],1)
 err=float((parts.sum(0)-oracle).abs().max().detach())
 # Every factor is live; dropping any branch must change this planted reader.
 omitted=[float(v.norm().detach()) for v in parts]
 # Independent directional finite difference tests the forward and backward pairing.
 dx=torch.randn_like(x);step=1e-5
 yp=fold(x.detach()+step*dx,first,weights,out,.3,cos,sin,h,reader,eps)[1]
 ym=fold(x.detach()-step*dx,first,weights,out,.3,cos,sin,h,reader,eps)[1]
 finite=((yp-ym)[:,None]*reader).sum((2,3))/(2*step)
 predicted=(parts.sum(0)*dx[:,None]).sum((2,3))
 fd=float((finite-predicted).abs().max().detach())
 assert err<1e-10 and fd<1e-7 and min(omitted)>1e-4
 result=dict(autograd_max_error=err,finite_difference_max_error=fd,branch_norms=omitted,scope='Synthetic FP64 five-factor adjoint; native fidelity and circuit sparsity untested')
 target=Path(__file__).with_name('ATTENTION_READER_FOLD_CPU_V1.json')
 if target.exists():raise FileExistsError(target)
 target.write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
