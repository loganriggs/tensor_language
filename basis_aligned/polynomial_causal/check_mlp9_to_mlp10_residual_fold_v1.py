"""Actual-weight composed response/product identity on signed synthetic edits."""
from pathlib import Path
import json
import torch
from head17_source_interface_v1 import CHECKPOINT
from mlp9_to_mlp10_residual_fold_v1 import execute
P=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2);torch.manual_seed(346)
    sd=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu')
    program=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)
    w=program['direction'].double();scale=float(sd['transformer.h.10.lambdas'][0]);eps=torch.finfo(torch.float32).eps
    L,R,D=[sd['transformer.h.9.mlp.'+k+'.weight'].double() for k in ('Left','Right','Down')]
    z=torch.randn(24,1152,dtype=torch.float64);a=torch.linspace(-.5,.5,24)[:,None].double();b=.3*torch.cos(torch.arange(24).double())[:,None]
    def mlp(x):return ((x@L.T)*(x@R.T))@D.T/(x.square().mean(-1,keepdim=True)+eps)
    base=mlp(z);fc,fr=execute(z,base,a,b,program,scale)
    rc=scale*(-a*w+mlp(z-a*w)-base);rr=scale*(-b*w+mlp(z-b*w)-base)
    L10,R10,D10=[sd['transformer.h.10.mlp.'+k+'.weight'].double() for k in ('Left','Right','Down')]
    def cross(x,y):return ((x@L10.T)*(y@R10.T)+(y@L10.T)*(x@R10.T))@D10.T
    expected=cross(rc,rr);actual=cross(fc,fr)
    result=dict(child_change_error=float((fc-rc).norm()/rc.norm()),remainder_change_error=float((fr-rr).norm()/rr.norm()),composed_product_error=float((actual-expected).norm()/expected.norm()),scope='Actualweight FP64 synthetic signed edits. One nativeMLP9 context andfixedwriterprogram; attentionpartners andMLP10jointnorm are external. Productcontrol comparesnumerators atsame denominator, not native causal outcomes.')
    assert max(result[k] for k in ('child_change_error','remainder_change_error','composed_product_error'))<1e-10
    (P/'MLP9_TO_MLP10_RESIDUAL_FOLD_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
