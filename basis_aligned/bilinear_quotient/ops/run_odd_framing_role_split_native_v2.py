#!/usr/bin/env python3
# BQGATE:240bodyforwards;48prefixes;180seconds;no fitting.
"""Correction-only V2: arm4 removes O rather than full S+R+O.
pred_a exact role partition, frozen anchors, finite, 240 forwards.
pred_b native capability and live framing effect each template.
pred_c instruction error<=.35 and description error>=.50 each template.
pred_d description error<=.35 and instruction error>=.50 each template.
"""
import json,os,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from run_even_value_factorial_native_v1 import setup,cpu_control
from odd_source_positions_v1 import source_channels,select_sources
from odd_framing_role_split_v1 import role_masks
from sparse_path_stability_atlas_v1 import digest
from regional_cue_row_check_v1 import validate
STEM='ODD_FRAMING_ROLE_SPLIT_NATIVE_V2';ARM_ORDER=['native','remove_odd_description','remove_odd_instruction','remove_odd_framing','remove_all_odd']
READOUTS=[('target',None),('work_jobs',(670,3946)),('cat_dog',(3797,3290)),('red_blue',(2266,4171)),('monday_tuesday',(3321,3431)),('apple_orange',(17180,10912))]


@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    rows=json.loads((P/'ODD_FRAMING_FRESH_V1_ROWS.json').read_text())['rows'];assert len(rows)==48;validate(rows);masks=role_masks(rows)
    prior=torch.load(P/'ODD_FRAMING_FRESH_V1_ARTIFACT.pt',map_location='cpu',weights_only=True)['values']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        v1=json.loads((P/'ODD_FRAMING_ROLE_SPLIT_NATIVE_V1_RESULT.json').read_text());assert not v1['pred_a'] and v1['max_frozen_anchor_error']>1
        control=json.loads((P/'ODD_FRAMING_ROLE_SPLIT_V1_CPU_CONTROL.json').read_text());assert control['pred_a'] and prior.shape==(5,48,2)
        print('240bodyforwards;correction-only arm4;CPUcontrol',cpu_control());return
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not artifact.exists()
    start=time.perf_counter();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();graph,_,_,_=setup('cuda');context={};checks=[]
    rel=lambda x,y:float((x-y).norm()/y.norm().clamp_min(1e-8))
    def hook(module,args,output):
        arm=context['arm']
        if arm==0:return output
        current,first=args;channels=source_channels(graph,current,first);pieces={k:select_sources(channels,v) for k,v in context['masks'].items()}
        checks.append(rel(pieces['description']+pieces['instruction'],pieces['framing']))
        states=graph.state(current,first_values=first.reshape(*current.shape[:2],9,128)[:,:,8])
        selected={1:pieces['description'],2:pieces['instruction'],3:pieces['framing'],4:states[2]}[arm]
        delta=selected@graph.p['output'].double().T
        return output[0]-delta.to(output[0].dtype),output[1]
    handle=model.transformer.h[9].attn.register_forward_hook(hook);values=torch.zeros(5,48,6,dtype=torch.float64);count=0
    try:
        for i,row in enumerate(rows):
            ids=torch.tensor([row['ids']],device='cuda')
            for arm in range(5):
                context.update(arm=arm,masks={k:v[None].to('cuda') for k,v in masks[i].items()})
                x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
                for block in model.transformer.h:x,v1=block(x,v1,x0)
                scores=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
                pairs=[(row['uk_id'],row['us_id'])]+[pair for _,pair in READOUTS[1:]]
                for j,(a,b) in enumerate(pairs):values[arm,i,j]=(scores[a]-scores[b]).cpu()
                count+=1
    finally:handle.remove()
    assert count==240 and bool(torch.isfinite(values).all());effect=values-values[0]
    anchor=max(float((values[0,:,:2]-prior[0]).abs().max()),float((values[3,:,:2]-prior[1]).abs().max()),float((values[4,:,:2]-prior[3]).abs().max()));cells=[]
    for family in range(2):
        ix=[i for i,r in enumerate(rows) if r['family']==family];native=values[0,ix[::2],0]-values[0,ix[1::2],0]
        cue=effect[:,ix[::2],0]-effect[:,ix[1::2],0];desc_err=rel(cue[1],cue[3]);inst_err=rel(cue[2],cue[3]);target_rms=[float(cue[a].square().mean().sqrt()) for a in (1,2,3)]
        controls={}
        for arm,name in [(1,'description'),(2,'instruction')]:controls[name]={READOUTS[j][0]:float(effect[arm,ix,j].square().mean().sqrt()/max(target_rms[arm-1],1e-8)) for j in range(2,6)}
        cells.append({'family':family,'native_positive_pairs':int((native>0).sum()),'native_cue_mean':float(native.mean()),'description_to_framing_cue_error':desc_err,'instruction_to_framing_cue_error':inst_err,'target_rms':{'description':target_rms[0],'instruction':target_rms[1],'framing':target_rms[2]},'new_control_ratios':controls,'separate_effect_composition_error':rel(cue[1]+cue[2],cue[3]),'capability_pass':int((native>0).sum())>=10 and target_rms[2]>=1e-5,'instruction_hypothesis_pass':inst_err<=.35 and desc_err>=.5,'description_hypothesis_pass':desc_err<=.35 and inst_err>=.5})
    torch.save({'values':values,'effects':effect},artifact)
    result={'pred_a':anchor<=1e-5 and max(checks)<=1e-10 and count==240 and bool(torch.isfinite(values).all()),'pred_b':all(c['capability_pass'] for c in cells),'pred_c':all(c['instruction_hypothesis_pass'] for c in cells),'pred_d':all(c['description_hypothesis_pass'] for c in cells),'max_frozen_anchor_error':anchor,'max_source_partition_error':max(checks),'families':cells,'body_forwards':count,'seconds':time.perf_counter()-start,'artifact_sha256':digest(artifact),'source_shas':binding,'arm_order':ARM_ORDER,'readout_order':[x[0] for x in READOUTS],'supersedes_control_only':'V1 arm4 removed full S+R+O while labeled all-O; native/framing/role data remain valid, but V1 pred_a failed and its receipt is preserved.','scope':'Correction-only native source-role split within the fresh O framing effect. Native upstream context and suffix remain; no unique semantic unit, corpus OOD, static compression, or quantization claim.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True);signal.alarm(0)


if __name__=='__main__':main()
