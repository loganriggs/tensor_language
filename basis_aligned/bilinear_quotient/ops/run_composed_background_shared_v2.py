#!/usr/bin/env python3
# BQGATE:160prefixes;800suffixreadouts;180seconds.
"""pred_a generated backgroundstate<=1e-5 and fullproduct<=.001relative.
pred_b regional target<=.01/control<=.05relative;totaloutput<=1e-4;no material flips.
pred_c FineWeb target<=.1relative,totaloutput<=1e-4 eachgroup.
Price160pristineprefixes,320native branch checks,800suffixreadouts;180seconds.
Null: backgrounddrift or numerical amplification prevents full-panel preservation.
"""
from pathlib import Path
import json,sys,time,signal,os
from hashlib import sha256
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parents[2]/'polynomial_causal';sys.path.insert(0,str(P))
sys.path.insert(0,str(P.parent/'bilinear_quotient/ops'))
from fastload import load_model_fast
from composed_mlp10_inputs_v2 import execute


@torch.no_grad()
def main():
    files=json.loads((P/'COMPOSED_BACKGROUND_SHARED_V2_BINDING.json').read_text())['files'];assert all(sha256(Path(k).read_bytes()).hexdigest()==v for k,v in files.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('160prefixes;5suffixreadouts each;180seconds');return
    assert not (P/'COMPOSED_BACKGROUND_SHARED_V2_RESULT.json').exists()
    signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;tic=time.perf_counter()
    model=load_model_fast().cuda().eval()
    assert next(model.parameters()).device.type=='cuda'
    rows=json.loads((P/'FIRST_TOKEN_PATH_FRESH_V1_ROWS.json').read_text())['rows']
    rows+=json.loads((P/'FIRST_TOKEN_REMOVAL_NEWLINE_FRESH_V1_ROWS.json').read_text())['rows']
    child=torch.load(P/'CROSSFIRST_STATE_EXECUTOR_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['fields']
    parent=torch.load(P/'CROSSFIRST_HIERARCHY_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['parent_fields']
    program=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True,map_location='cpu')
    child={k:v.cuda() for k,v in child.items()};parent={k:v.cuda() for k,v in parent.items()};program={k:v.cuda() for k,v in program.items()};w=program['direction'];b9=model.transformer.h[9];b10=model.transformer.h[10]
    matrices=[getattr(b10.attn,k).weight.double() for k in ('c_q','c_k','c_q2','c_k2','c_v')]
    L,R,D=[getattr(b10.mlp,k).weight.double() for k in ('Left','Right','Down')]
    rel=lambda x,y:float((x-y).norm()/y.norm().clamp_min(1e-30))
    cells=[]
    for index in range(160):
        ids=torch.tensor([rows[index]['ids']],device='cuda');x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
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
        generated_states=[]
        for name in ('child','remainder'):
            zg=(z0.double()+generated[name]).float()
            generated_states.append((zg+b10.mlp(F.rms_norm(zg,(1152,)))).double())
        generated_bar=generated_states[0]+generated_states[1]-h0
        outcomes=[]
        for initial in (bar,bar+expected,bar+predicted,generated_bar,generated_bar+predicted):
            state=initial.float()
            for block in model.transformer.h[11:]:state,_=block(state,v1,x0)
            logits=(30*torch.tanh(model.lm_head(F.rms_norm(state[:,-1],(1152,)))/30))[0]
            if index<96:
                target=logits[rows[index]['uk_id']]-logits[rows[index]['us_id']]
                control=logits[rows[index]['control_ids'][0]]-logits[rows[index]['control_ids'][1]]
            else:target=-logits.log_softmax(-1)[198];control=logits[198]-logits[11]
            outcomes.append([float(target),float(control)])
        scores=torch.tensor(outcomes,dtype=torch.float64)
        effects=scores[1:3]-scores[0]
        transported=scores[4]-scores[3]
        background_drift=scores[3]-scores[0]
        total_error=scores[4]-scores[1]
        cells.append(dict(background_state_error=rel(generated_bar,bar),background_drift=background_drift.tolist(),transported_effects=transported.tolist(),transported_effect_errors=(transported-effects[0]).tolist(),total_output_errors=total_error.tolist(),native_effects=effects[0].tolist(),predicted_effects=effects[1].tolist(),absolute_effect_errors=(effects[1]-effects[0]).abs().tolist(),row=index,tokens=len(rows[index]['ids']),child_error=rel(generated['child'],c),
                          remainder_error=rel(generated['remainder'],r),
                          joint_denominator_error=rel(generated['joint_rho'],rho),
                          product_error=rel(predicted,expected),product_reference_norm=float(expected.norm())))
        if index%40==39:print('completed',index+1,flush=True)
    groups=[]
    for lo,hi,label in [(24*k,24*(k+1),'regional'+str(k)) for k in range(4)]+[(96+16*k,112+16*k,'fineweb'+str(k)) for k in range(4)]:
        selected=cells[lo:hi];reference=torch.tensor([c['native_effects'] for c in selected],dtype=torch.float64);pred=torch.tensor([c['transported_effects'] for c in selected],dtype=torch.float64)
        groups.append(dict(group=label,target_relative_error=rel(pred[:,0],reference[:,0]),control_relative_error=rel(pred[:,1],reference[:,1]),maximum_background_drift=max(abs(v) for c in selected for v in c['background_drift']),maximum_total_output_error=max(abs(v) for c in selected for v in c['total_output_errors']),material_target_sign_reversals=int(((pred[:,0]*reference[:,0]<0)&(reference[:,0].abs()>=1e-5)).sum())))
    result={'pred_a':max(c['background_state_error'] for c in cells)<=1e-5 and max(c['product_error'] for c in cells)<=.001,
            'pred_b':all(g['target_relative_error']<=.01 and g['control_relative_error']<=.05 and g['maximum_total_output_error']<=1e-4 and g['material_target_sign_reversals']==0 for g in groups[:4]),
            'pred_c':all(g['target_relative_error']<=.1 and g['maximum_total_output_error']<=1e-4 for g in groups[4:]),
            'cells':cells,'groups':groups,'seconds':time.perf_counter()-tic,
            'scope':'Current shared input generator with generated additive MLP10 background;160historicalprefixes. Pristine native context, frozen scalar fields and native suffix supplied. No changed states/denominators/background used by candidate. No freshOOD or whole causal interaction sufficiency claim.'}
    (P/'COMPOSED_BACKGROUND_SHARED_V2_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
