"""Actual-weight native FP32 branch control for the integrated input generator."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parents[1]))
from jacclust.tt_model import Bilinear,CausalBilinearSelfAttention
from head17_source_interface_v1 import CHECKPOINT
from composed_mlp10_inputs_v1 import execute


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.manual_seed(350)
    sd=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu')
    cfg=SimpleNamespace(n_head=9,n_embd=1152,squared_attn=True,bilinear_attn=True,
                        expansion_factor=4,gated=False)
    mlp=Bilinear(cfg).eval();att=CausalBilinearSelfAttention(cfg).eval()
    for module,prefix in [(mlp,'transformer.h.9.mlp.'),(att,'transformer.h.10.attn.')]:
        module.load_state_dict({k[len(prefix):]:v for k,v in sd.items() if k.startswith(prefix)})
    program=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)
    w=program['direction'].float();lam=sd['transformer.h.10.lambdas']
    z9=torch.randn(2,17,1152);x0=torch.randn_like(z9);first=torch.randn_like(z9)
    m0=mlp(F.rms_norm(z9,(1152,)));h0=z9+m0
    raw0=lam[0]*h0+lam[1]*x0;att0=att(F.rms_norm(raw0,(1152,)),first)[0];z0=raw0+att0
    matrices=[getattr(att,k).weight.double() for k in ('c_q','c_k','c_q2','c_k2','c_v')]
    L,R,D=[sd['transformer.h.10.mlp.'+k+'.weight'].double() for k in ('Left','Right','Down')]
    def cross(c,r,rho):return ((c@L.T)*(r@R.T)+(r@L.T)*(c@R.T))@D.T/rho
    rel=lambda x,y:float((x-y).norm()/y.norm().clamp_min(1e-30))
    cells=[]
    for strength in (.003,.03,.3):
        a=strength*torch.linspace(-1,1,34).reshape(2,17,1)
        b=strength*torch.cos(torch.arange(34)).reshape(2,17,1)
        prediction=execute(z9,m0.double()-mlp.Down_bias.double(),raw0,z0,first.double(),a,b,
                           program,float(lam[0]),matrices,float(att.lamb),att.c_proj.weight.double())
        direct=[]
        for amplitude in (a,b):
            za=z9-amplitude*w;ha=za+mlp(F.rms_norm(za,(1152,)))
            raw=lam[0]*ha+lam[1]*x0
            state=raw+att(F.rms_norm(raw,(1152,)),first)[0]
            direct.append(state.double()-z0.double())
        c,r=direct
        rho=(z0.double()+c+r).square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps
        native_product=cross(c,r,rho)
        predicted_product=cross(prediction['child'],prediction['remainder'],prediction['joint_rho'])
        cells.append(dict(strength=strength,child_error=rel(prediction['child'],c),
                          remainder_error=rel(prediction['remainder'],r),
                          joint_denominator_error=rel(prediction['joint_rho'],rho),
                          product_error=rel(predicted_product,native_product)))
    result=dict(cells=cells,scope='Actual-weight synthetic FP32 MLP9/attention10 branch execution; '
                'generated inputs and joint denominator, then FP64 MLP10 product comparison. '
                'Two 17-token contexts and three signed-amplitude scales. Native pristine '
                'states/first values retained; no corpus, downstream effect or extraction claim.')
    (P/'COMPOSED_MLP10_INPUTS_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    assert max(c[k] for c in cells for k in ('child_error','remainder_error','product_error'))<.01


if __name__=='__main__':main()
