"""Registered eight-prefix original-interaction screen.

A all post10 branch state relative errors <=1e-5.
B regional original interaction target error <=2%, control <=5% (group vector).
C every branch final absolute output discrepancy <=1e-4.
FineWeb relative effects descriptive, not waived or pooled into regional bar.
Price: eight pristine prefixes,24 reference changed branches,24 generated
branches,64 suffix evaluations; CPU only with120second cap. No fitting.
"""
from pathlib import Path
import json,sys,time,signal
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'bilinear_quotient/ops'))
from fastload import load_model_fast
from composed_mlp10_context_v1 import prepare
from composed_joint_response_v1 import branch
from regional_cue_row_check_v1 import validate


@torch.no_grad()
def main():
    signal.alarm(120);torch.set_num_threads(2);tic=time.perf_counter()
    model=load_model_fast().eval()
    assert next(model.parameters()).device.type=='cpu'
    rows=json.loads((P/'FIRST_TOKEN_PATH_FRESH_V1_ROWS.json').read_text())['rows']
    row_check=validate(rows)
    rows+=json.loads((P/'FIRST_TOKEN_REMOVAL_NEWLINE_FRESH_V1_ROWS.json').read_text())['rows']
    child=torch.load(P/'CROSSFIRST_STATE_EXECUTOR_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['fields']
    parent=torch.load(P/'CROSSFIRST_HIERARCHY_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['parent_fields']
    program=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True,map_location='cpu')
    w=program['direction'];b9=model.transformer.h[9];b10=model.transformer.h[10]
    matrices=[getattr(b10.attn,k).weight.double() for k in ('c_q','c_k','c_q2','c_k2','c_v')]
    rel=lambda x,y:float((x-y).norm()/y.norm().clamp_min(1e-30))
    cells=[]
    for index in (0,24,48,72,96,112,128,144):
        ids=torch.tensor([rows[index]['ids']]);x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
        for block in model.transformer.h[:9]:x,v1=block(x,v1,x0)
        raw9=b9.lambdas[0]*x+b9.lambdas[1]*x0
        att9,v1=b9.attn(F.rms_norm(raw9,(1152,)),v1)
        z9=raw9+att9;m9=b9.mlp(F.rms_norm(z9,(1152,)));h9=z9+m9
        raw0=b10.lambdas[0]*h9+b10.lambdas[1]*x0
        z0=raw0+b10.attn(F.rms_norm(raw0,(1152,)),v1)[0]
        a=child[index][...,None];b=(parent[index]-child[index])[...,None]
        context=prepare(z9,m9.double()-b9.mlp.Down_bias.double(),raw0,z0,v1.double(),
                        program,float(b10.lambdas[0]),matrices,float(b10.attn.lamb),b10.attn.c_proj.weight.double())
        def post(z):
            z=z.float()
            return z+b10.mlp(F.rms_norm(z,(1152,)))
        native=[post(z0)];generated=[native[0]]
        for amplitude in (a,b,a+b):
            za=raw9+(att9-(amplitude*w).to(att9.dtype))
            ha=za+b9.mlp(F.rms_norm(za,(1152,)))
            raw=b10.lambdas[0]*ha+b10.lambdas[1]*x0
            state=raw+b10.attn(F.rms_norm(raw,(1152,)),v1)[0]
            native.append(post(state));generated.append(post(branch(amplitude,context)))
        outcomes=[]
        for initial in native+generated:
            state=initial
            for block in model.transformer.h[11:]:state,_=block(state,v1,x0)
            logits=(30*torch.tanh(model.lm_head(F.rms_norm(state[:,-1],(1152,)))/30))[0]
            if index<96:
                target=logits[rows[index]['uk_id']]-logits[rows[index]['us_id']]
                control=logits[rows[index]['control_ids'][0]]-logits[rows[index]['control_ids'][1]]
            else:target=-logits.log_softmax(-1)[198];control=logits[198]-logits[11]
            outcomes.append([float(target),float(control)])
        scores=torch.tensor(outcomes,dtype=torch.float64).reshape(2,4,2)
        interactions=scores[:,3]-scores[:,1]-scores[:,2]+scores[:,0]
        cells.append(dict(row=index,state_errors=[rel(g.double(),n.double()) for g,n in zip(generated,native)],
                          native_scores=scores[0].tolist(),generated_scores=scores[1].tolist(),
                          native_interaction=interactions[0].tolist(),generated_interaction=interactions[1].tolist(),
                          absolute_output_error=float((scores[1]-scores[0]).abs().max())))
    groups=[]
    for name,subset in [('regional',cells[:4]),('FineWeb',cells[4:])]:
        n=torch.tensor([c['native_interaction'] for c in subset],dtype=torch.float64)
        g=torch.tensor([c['generated_interaction'] for c in subset],dtype=torch.float64)
        groups.append(dict(name=name,relative_errors=[rel(g[:,j],n[:,j]) for j in range(2)],
                           max_absolute_effect_errors=(g-n).abs().amax(0).tolist(),
                           reference_norms=n.norm(dim=0).tolist()))
    result=dict(pred_a=max(max(c['state_errors']) for c in cells)<=1e-5,
                pred_b=groups[0]['relative_errors'][0]<=.02 and groups[0]['relative_errors'][1]<=.05,
                pred_c=max(c['absolute_output_error'] for c in cells)<=1e-4,
                groups=groups,cells=cells,row_check=row_check,seconds=time.perf_counter()-tic,
                scope='Eight historical prefixes, original two-edit interaction with generated individual and joint states. Pristine context, frozen scalar fields and native weights/suffix retained. FP32 MLP10 applied to rounded generated z10. FineWeb is training-domain text, no fresh OOD claim.')
    (P/'COMPOSED_JOINT_NATIVE_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cells'},indent=2))


if __name__=='__main__':main()
