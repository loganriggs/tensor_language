"""Actual MLP10 FP64 control of mixed multiplication/norm decomposition."""
from pathlib import Path
import json
import torch
from mlp_two_edit_mixed_v1 import decompose,EPS
from head17_source_interface_v1 import CHECKPOINT
P=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2);torch.manual_seed(332)
    sd=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu')
    L=sd['transformer.h.10.mlp.Left.weight'].double();R=sd['transformer.h.10.mlp.Right.weight'].double();D=sd['transformer.h.10.mlp.Down.weight'].double()
    z=torch.randn(24,1152,dtype=torch.float64);c=.1*torch.randn_like(z);r=.2*torch.randn_like(z)
    def g(x):return x+((x@L.T)*(x@R.T))@D.T/(x.square().mean(-1,keepdim=True)+EPS)
    reference=g(z+c+r)-g(z+c)-g(z+r)+g(z)
    parts=decompose(z,c,r,L,R,D);swap=decompose(z,r,c,L,R,D)
    result=dict(mixed_relative_error=float((parts['total']-reference).norm()/reference.norm()),edit_exchange_error=float((swap['total']-parts['total']).norm()/parts['total'].norm()),scope='Actual-weight synthetic FP64 algebra only. Bias and residual cancel. Normalization remainder includes all changes in denominators; group labels are convention-dependent, not semantic units.')
    assert max(result[k] for k in ('mixed_relative_error','edit_exchange_error'))<1e-10
    (P/'MLP_TWO_EDIT_MIXED_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
