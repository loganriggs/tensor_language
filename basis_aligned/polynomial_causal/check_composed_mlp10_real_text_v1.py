"""Two fixed real prefixes: CPU input-generator check while the GPU lane is occupied."""
from pathlib import Path
import json,sys,time,signal
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'bilinear_quotient/ops'))
from fastload import load_model_fast
from composed_mlp10_inputs_v1 import execute


@torch.no_grad()
def main():
    signal.alarm(120);torch.set_num_threads(2);tic=time.perf_counter()
    model=load_model_fast().eval()
    assert next(model.parameters()).device.type=='cpu'
    rows=json.loads((P/'FIRST_TOKEN_PATH_FRESH_V1_ROWS.json').read_text())['rows']
    rows+=json.loads((P/'FIRST_TOKEN_REMOVAL_NEWLINE_FRESH_V1_ROWS.json').read_text())['rows']
    child=torch.load(P/'CROSSFIRST_STATE_EXECUTOR_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['fields']
    parent=torch.load(P/'CROSSFIRST_HIERARCHY_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['parent_fields']
    program=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True,map_location='cpu')
    w=program['direction'];b9=model.transformer.h[9];b10=model.transformer.h[10]
    matrices=[getattr(b10.attn,k).weight.double() for k in ('c_q','c_k','c_q2','c_k2','c_v')]
    L,R,D=[getattr(b10.mlp,k).weight.double() for k in ('Left','Right','Down')]
    rel=lambda x,y:float((x-y).norm()/y.norm().clamp_min(1e-30))
    cells=[]
    for index in (0,96):
        ids=torch.tensor([rows[index]['ids']]);x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
        for block in model.transformer.h[:9]:x,v1=block(x,v1,x0)
        raw9=b9.lambdas[0]*x+b9.lambdas[1]*x0
        att9,v1=b9.attn(F.rms_norm(raw9,(1152,)),v1)
        z9=raw9+att9;m9=b9.mlp(F.rms_norm(z9,(1152,)));h9=z9+m9
        raw0=b10.lambdas[0]*h9+b10.lambdas[1]*x0
        att0=b10.attn(F.rms_norm(raw0,(1152,)),v1)[0];z0=raw0+att0
        a=child[index][...,None];b=(parent[index]-child[index])[...,None]
        generated=execute(z9,m9.double()-b9.mlp.Down_bias.double(),raw0,z0,v1.double(),a,b,
                          program,float(b10.lambdas[0]),matrices,float(b10.attn.lamb),b10.attn.c_proj.weight.double())
        changes=[];states=[]
        for amplitude in (a,b):
            za=raw9+(att9-(amplitude*w).to(att9.dtype))
            ha=za+b9.mlp(F.rms_norm(za,(1152,)))
            raw=b10.lambdas[0]*ha+b10.lambdas[1]*x0
            state=raw+b10.attn(F.rms_norm(raw,(1152,)),v1)[0]
            changes.append(state.double()-z0.double())
            states.append((state+b10.mlp(F.rms_norm(state,(1152,)))).double())
        c,r=changes;rho=(z0.double()+c+r).square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps
        def product(c,r,rho):return ((c@L.T)*(r@R.T)+(r@L.T)*(c@R.T))@D.T/rho
        expected=product(c,r,rho)
        predicted=product(generated['child'],generated['remainder'],generated['joint_rho'])
        h0=(z0+b10.mlp(F.rms_norm(z0,(1152,)))).double();bar=states[0]+states[1]-h0
        outcomes=[]
        for delta in (torch.zeros_like(expected),expected,predicted):
            state=(bar+delta).float()
            for block in model.transformer.h[11:]:state,_=block(state,v1,x0)
            logits=(30*torch.tanh(model.lm_head(F.rms_norm(state[:,-1],(1152,)))/30))[0]
            if index<96:
                target=logits[rows[index]['uk_id']]-logits[rows[index]['us_id']]
                control=logits[rows[index]['control_ids'][0]]-logits[rows[index]['control_ids'][1]]
            else:target=-logits.log_softmax(-1)[198];control=logits[198]-logits[11]
            outcomes.append([float(target),float(control)])
        effects=torch.tensor(outcomes,dtype=torch.float64)[1:]-torch.tensor(outcomes[0],dtype=torch.float64)
        cells.append(dict(native_effects=effects[0].tolist(),predicted_effects=effects[1].tolist(),absolute_effect_errors=(effects[1]-effects[0]).abs().tolist(),row=index,tokens=len(rows[index]['ids']),child_error=rel(generated['child'],c),
                          remainder_error=rel(generated['remainder'],r),
                          joint_denominator_error=rel(generated['joint_rho'],rho),
                          product_error=rel(predicted,expected),product_reference_norm=float(expected.norm())))
        print(json.dumps(cells[-1]),flush=True)
    result=dict(cells=cells,seconds=time.perf_counter()-tic,
                scope='Fixed rows0 and96, actual native CPU pristine prefix and head9 edits through attention10. '
                'Previously frozen GPU-produced scalar fields supplied identically to direct and predicted branches. '
                'Three native suffix readouts compare generated versus direct-product local effects at an additive background. No data fitting, fresh OOD or CPU/GPU equivalence claim. '
                'Small real-state check; does not replace pending160-prefix GPU experiment.')
    (P/'COMPOSED_MLP10_REAL_TEXT_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
