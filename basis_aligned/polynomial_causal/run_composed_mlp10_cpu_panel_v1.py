"""Registered full existing-panel CPU check of integrated interaction prediction."""
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
    signal.alarm(300);torch.set_num_threads(2);tic=time.perf_counter()
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
    for index in range(160):
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
        if (index+1)%24==0:print('completed',index+1,flush=True)
    groups=[]
    for lo,hi,label in [(24*k,24*(k+1),'regional'+str(k)) for k in range(4)]+[(96+16*k,112+16*k,'newline'+str(k)) for k in range(4)]:
        direct=torch.tensor([c['native_effects'] for c in cells[lo:hi]],dtype=torch.float64)
        predicted=torch.tensor([c['predicted_effects'] for c in cells[lo:hi]],dtype=torch.float64)
        groups.append(dict(group=label,relative_errors=[rel(predicted[:,e],direct[:,e]) for e in range(2)],
                           maxabs_errors=(predicted-direct).abs().max(0).values.tolist(),
                           same_sign=(predicted*direct>0).sum(0).tolist(),
                           opposite_sign=(predicted*direct<0).sum(0).tolist(),
                           predicted_zero_reference_nonzero=((predicted==0)&(direct!=0)).sum(0).tolist(),
                           reference_zero_predicted_nonzero=((direct==0)&(predicted!=0)).sum(0).tolist()))
    result=dict(pred_a=all(g['relative_errors'][0]<=.02 for g in groups[:4]),
                pred_b=all(max(c['child_error'],c['remainder_error'])<=.01 and c['joint_denominator_error']<=.001 for c in cells),
                groups=groups,cells=cells,seconds=time.perf_counter()-tic,
                scope='All160 existing prefixes; native CPU pristine prefix/head9 edits and three suffix readouts. '
                'Frozen GPU-derived scalar fields supplied identically to both branches. Pristine context and additive '
                'background/fullsuffix retained. No fitting,freshOOD,CPU/GPU equivalence or global extraction claim.')
    (P/'COMPOSED_MLP10_CPU_PANEL_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cells'},indent=2))


if __name__=='__main__':main()
