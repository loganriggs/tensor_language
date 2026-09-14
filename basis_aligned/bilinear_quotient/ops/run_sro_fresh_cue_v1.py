#!/usr/bin/env python3
# BQGATE:576bodyforwards;72prefixes;180seconds;no fitting.
"""pred_a shared/separate writes<=1e-10, Mobius replay<=1e-12.
pred_b native capability >=10/12positive pairs AND fullhead effect>=10% eachfamily.
pred_c withheld R-corner cue prediction error<=.10 EACHfamily.
Null: cue reduction fails on fixed new templates or native capability is absent.
Price576bodyforwards,72prefixes,180seconds; full native weights and four observed
corners retained. No quantization, fitting, or global parameter saving claim.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from run_even_value_factorial_native_v1 import setup,cpu_control
from even_value_factorial_v1 import coefficients,reconstruct
from sparse_path_stability_atlas_v1 import digest
from regional_cue_row_check_v1 import validate
STEM='SRO_FRESH_CUE_V1'


@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
    assert all(digest(k)==v for k,v in binding.items())
    rows=json.loads((P/(STEM+'_ROWS.json')).read_text())['rows']
    assert len(rows)==72
    validate(rows[:24]);validate(rows[24:48],expected_token_differences=2);validate(rows[48:])
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print('576bodyforwards;72prefixes; eight masks; CPUcontrol',cpu_control());return
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_ARTIFACT.pt')
    assert not out.exists() and not artifact.exists()
    start=time.perf_counter();signal.alarm(180);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();graph,separate,scalar,writer=setup('cuda')
    context={};checks=[]
    def hook(module,args,output):
        mask=context['mask']
        if not mask:return output
        current=args[0];first=args[1].reshape(*current.shape[:2],9,128)[:,:,8]
        states=graph.state(current,first_values=first)
        gains=[int(bool(mask&bit)) for bit in [1,2,4]]
        delta=graph.write(states,gains)
        even,odd=separate(current,first)
        selected=scalar.scalar(current,context['tokens'],1)[...,None]*writer
        reference=gains[0]*selected+gains[1]*(even-selected)+gains[2]*odd
        checks.append(float((delta-reference).norm()/reference.norm().clamp_min(1e-30)))
        return output[0]-delta.to(output[0].dtype),output[1]
    handle=model.transformer.h[9].attn.register_forward_hook(hook)
    cube=torch.zeros(8,72,2,dtype=torch.float64);count=0
    try:
        for i,row in enumerate(rows):
            ids=torch.tensor([row['ids']],device='cuda')
            for mask in range(8):
                context.update(tokens=ids,mask=mask)
                x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
                for block in model.transformer.h:x,v1=block(x,v1,x0)
                scores=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
                cube[mask,i,0]=(scores[row['uk_id']]-scores[row['us_id']]).cpu()
                cube[mask,i,1]=(scores[row['control_ids'][0]]-scores[row['control_ids'][1]]).cpu()
                count+=1
    finally:handle.remove()
    assert count==576
    coeff=coefficients(cube);replay=float((reconstruct(coeff)-cube).abs().max())
    cells=[]
    for family in range(3):
        ix=[i for i,r in enumerate(rows) if r['family']==family]
        paired=cube[:,ix[::2],0]-cube[:,ix[1::2],0]
        effects=paired-paired[0];scale=effects[7].norm().clamp_min(1e-8)
        errors=[float((effects[mask&5]-effects[mask]).norm()/scale) for mask in [2,3,6,7]]
        controls=cube[:,ix,1]-cube[0,ix,1]
        control_errors=[float((controls[mask&5]-controls[mask]).norm()/controls[7].norm().clamp_min(1e-8)) for mask in [2,3,6,7]]
        positive=int((paired[0]>0).sum());materiality=float(effects[7].norm()/paired[0].norm().clamp_min(1e-8))
        cells.append(dict(family=family,native_positive_pairs=positive,head_effect_to_native=materiality,
                          cue_prediction_errors=errors,control_prediction_errors=control_errors,
                          capability_pass=positive>=10 and materiality>=.1,reduction_pass=max(errors)<=.1,
                          native_paired_contrasts=paired[0].tolist(),full_head_paired_effects=effects[7].tolist()))
    torch.save(dict(cube=cube,coefficients=coeff),artifact)
    result={'pred_a':max(checks)<=1e-10 and replay<=1e-12,
            'pred_b':all(c['capability_pass'] for c in cells),
            'pred_c':all(c['reduction_pass'] for c in cells),
            'max_live_write_error':max(checks),'mobius_replay_maxabs':replay,'families':cells,
            'body_forwards':count,'seconds':time.perf_counter()-start,'artifact_sha256':digest(artifact),
            'source_shas':binding,'scope':'New constructions; fixed reduction calibrated from four local corners, evaluated on four held-out R-containing corners. Original model/context and observed-corner costs retained; no autonomous prediction or semantic identification.'}
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True)


if __name__=='__main__':main()
