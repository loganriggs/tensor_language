#!/usr/bin/env python3
# BQGATE:192bodyforwards;48prefixes;180seconds;no fitting.
"""pred_a exact value split and frozen anchors;pred_b capability/live.
pred_c current-only wins;pred_d inherited-only wins;pred_e separate effects compose.
"""
import json,os,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from run_even_value_factorial_native_v1 import setup,cpu_control
from odd_source_positions_v1 import select_sources
from odd_semantic_positions_v1 import semantic_masks
from odd_source_swap_interaction_v1 import source_factors
from odd_value_source_split_v1 import value_parts
from sparse_path_stability_atlas_v1 import digest
from regional_cue_row_check_v1 import validate
STEM='ODD_VALUE_SOURCE_SPLIT_NATIVE_V1';ARMS=['native','full_value_swap','current_value_swap','inherited_value_swap'];READOUTS=[('target',None),('work_jobs',(670,3946)),('cat_dog',(3797,3290)),('red_blue',(2266,4171)),('monday_tuesday',(3321,3431)),('apple_orange',(17180,10912))]

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items());rows=json.loads((P/'ODD_FRAMING_FRESH_V1_ROWS.json').read_text())['rows'];assert len(rows)==48;validate(rows);masks=semantic_masks(rows);prior=torch.load(P/'ODD_SOURCE_SWAP_INTERACTION_NATIVE_V2_ARTIFACT.pt',map_location='cpu',weights_only=True)['values']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        assert json.loads((P/'ODD_VALUE_SOURCE_SPLIT_V1_CPU_CONTROL.json').read_text())['pred_a'];print('192bodyforwards;48 captures+3x48 value swaps;CPUcontrol',cpu_control());return
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not artifact.exists();start=time.perf_counter();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();graph,_,_,_=setup('cuda');context={'arm':0};cache=[];factor_errors=[];delta_errors=[];source_norms=[];rel=lambda x,y:float((x-y).norm()/y.norm().clamp_min(1e-8))
    def hook(module,args,output):
        current,first=args;arm=context['arm']
        if arm==0:cache.append((current.detach().cpu(),first.detach().cpu()));return output
        i=context['i'];mask=masks[i]['framing'][None].to(current.device);dc,df=cache[i^1];dc,df=dc.to(current.device),df.to(first.device);value=current.clone();first_d=first.clone();value[:,mask[0]]=dc[:,mask[0]];first_d[:,mask[0]]=df[:,mask[0]]
        routing,v0=source_factors(graph,current,current,current,first);co,io=value_parts(graph,current,first);cd,id_=value_parts(graph,value,first_d);full=cd+id_;current_only=cd+io;inherited_only=co+id_
        factor_errors.append(rel(co+io,v0));delta_errors.append(rel(full-v0,(current_only-v0)+(inherited_only-v0)));source_norms.extend([float(select_sources(routing*(current_only-v0),mask).norm()),float(select_sources(routing*(inherited_only-v0),mask).norm())])
        replacement={1:full,2:current_only,3:inherited_only}[arm];delta=select_sources(routing*(replacement-v0),mask)@graph.p['output'].double().T
        return output[0]+delta.to(output[0].dtype),output[1]
    handle=model.transformer.h[9].attn.register_forward_hook(hook);values=torch.zeros(4,48,6,dtype=torch.float64);count=0
    def forward(i,arm):
        nonlocal count
        row=rows[i];ids=torch.tensor([row['ids']],device='cuda');x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
        for block in model.transformer.h:x,v1=block(x,v1,x0)
        scores=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];pairs=[(row['uk_id'],row['us_id'])]+[pair for _,pair in READOUTS[1:]]
        for j,(a,b) in enumerate(pairs):values[arm,i,j]=(scores[a]-scores[b]).cpu()
        count+=1
    try:
        for i in range(48):forward(i,0)
        assert len(cache)==48
        for arm in range(1,4):
            context['arm']=arm
            for i in range(48):context['i']=i;forward(i,arm)
    finally:handle.remove()
    assert count==192 and bool(torch.isfinite(values).all());effect=values-values[0];anchor=max(float((values[0,:,:2]-prior[0,:,:2]).abs().max()),float((values[1,:,:2]-prior[3,:,:2]).abs().max()));cells=[]
    for family in range(2):
        ix=[i for i,r in enumerate(rows) if r['family']==family];native=values[0,ix[::2],0]-values[0,ix[1::2],0];cue=effect[:,ix[::2],0]-effect[:,ix[1::2],0];full=cue[1];cur_err=rel(cue[2],full);first_err=rel(cue[3],full);composition=rel(cue[2]+cue[3],full);target=float(full.square().mean().sqrt());controls={ARMS[a]:{READOUTS[j][0]:float(effect[a,ix,j].square().mean().sqrt()/max(target,1e-8)) for j in range(2,6)} for a in range(1,4)}
        cells.append({'family':family,'native_positive_pairs':int((native>0).sum()),'native_cue_mean':float(native.mean()),'full_value_swap_target_rms':target,'current_to_full_cue_error':cur_err,'inherited_to_full_cue_error':first_err,'separate_effect_composition_error':composition,'control_ratios_to_full_target':controls,'capability_pass':int((native>0).sum())>=10 and target>=1e-5,'current_hypothesis_pass':cur_err<=.2 and first_err>=.5,'inherited_hypothesis_pass':first_err<=.2 and cur_err>=.5,'behavioral_composition_pass':composition<=.1})
    torch.save({'values':values,'effects':effect},artifact);result={'pred_a':anchor<=1e-5 and max(factor_errors)<=1e-10 and max(delta_errors)<=1e-10 and min(source_norms)>1e-8 and count==192 and bool(torch.isfinite(values).all()),'pred_b':all(c['capability_pass'] for c in cells),'pred_c':all(c['current_hypothesis_pass'] for c in cells),'pred_d':all(c['inherited_hypothesis_pass'] for c in cells),'pred_e':all(c['behavioral_composition_pass'] for c in cells),'max_frozen_anchor_error':anchor,'max_value_factor_error':max(factor_errors),'max_source_delta_recomposition_error':max(delta_errors),'min_source_delta_norm':min(source_norms),'families':cells,'body_forwards':count,'seconds':time.perf_counter()-start,'artifact_sha256':digest(artifact),'source_shas':binding,'arm_order':ARMS,'readout_order':[x[0] for x in READOUTS],'scope':'Exact current/inherited value-source split of all-query-fixed O framing interchange with native suffix. No fitting, upstream necessity, corpus OOD, compression adoption, or quantization.'};out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True);signal.alarm(0)

if __name__=='__main__':main()
