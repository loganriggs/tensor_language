#!/usr/bin/env python3
# BQGATE:240bodyforwards;48prefixes;180seconds;no fitting.
"""Correction-only V2: pred_a exact expansion and frozen native anchor;pred_b capability/live.
pred_c additive-no-mixed error<=.10;pred_d value-only wins;pred_e routing-only wins.
"""
import json,os,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from run_even_value_factorial_native_v1 import setup,cpu_control
from odd_source_positions_v1 import source_channels,select_sources
from odd_semantic_positions_v1 import semantic_masks
from odd_source_swap_interaction_v1 import source_factors
from sparse_path_stability_atlas_v1 import digest
from regional_cue_row_check_v1 import validate
STEM='ODD_SOURCE_SWAP_INTERACTION_NATIVE_V2';ARMS=['native','full_swap','routing_swap','value_swap','additive_no_mixed'];READOUTS=[('target',None),('work_jobs',(670,3946)),('cat_dog',(3797,3290)),('red_blue',(2266,4171)),('monday_tuesday',(3321,3431)),('apple_orange',(17180,10912))]

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items());rows=json.loads((P/'ODD_FRAMING_FRESH_V1_ROWS.json').read_text())['rows'];assert len(rows)==48;validate(rows);masks=semantic_masks(rows);prior=torch.load(P/'ODD_FRAMING_SOURCE_SWAP_V1_ARTIFACT.pt',map_location='cpu',weights_only=True)['values']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        assert json.loads((P/'ODD_SOURCE_SWAP_INTERACTION_V1_CPU_CONTROL.json').read_text())['pred_a'];v1=json.loads((P/'ODD_SOURCE_SWAP_INTERACTION_NATIVE_V1_RESULT.json').read_text());assert not v1['pred_a'] and v1['pred_b'] and v1['pred_c'] and v1['pred_d'] and not v1['pred_e'];print('240bodyforwards;all recipient queries fixed;CPUcontrol',cpu_control());return
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not artifact.exists();start=time.perf_counter();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();graph,_,_,_=setup('cuda');context={'arm':0};cache=[];replays=[];algebra=[];mixed_norms=[];rel=lambda x,y:float((x-y).norm()/y.norm().clamp_min(1e-8))
    def hook(module,args,output):
        current,first=args;arm=context['arm']
        if arm==0:cache.append((current.detach().cpu(),first.detach().cpu()));return output
        i=context['i'];mask=masks[i]['framing'][None].to(current.device);dc,df=cache[i^1];dc,df=dc.to(current.device),df.to(first.device);key=current.clone();value=current.clone();first_d=first.clone();key[:,mask[0]]=dc[:,mask[0]];value[:,mask[0]]=dc[:,mask[0]];first_d[:,mask[0]]=df[:,mask[0]]
        r0,v0=source_factors(graph,current,current,current,first);rd,_=source_factors(graph,current,key,current,first);_,vd=source_factors(graph,current,current,value,first_d);orig=r0*v0;full=rd*vd;route=rd*v0;val=r0*vd;mixed=(rd-r0)*(vd-v0)
        replays.append(rel(orig,source_channels(graph,current,first)));algebra.append(rel(full-orig,(route-orig)+(val-orig)+mixed));mixed_norms.append(float(select_sources(mixed,mask).norm()))
        replacement={1:full,2:route,3:val,4:route+val-orig}[arm];delta=(select_sources(replacement,mask)-select_sources(orig,mask))@graph.p['output'].double().T
        return output[0]+delta.to(output[0].dtype),output[1]
    handle=model.transformer.h[9].attn.register_forward_hook(hook);values=torch.zeros(5,48,6,dtype=torch.float64);count=0
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
        for arm in range(1,5):
            context['arm']=arm
            for i in range(48):context['i']=i;forward(i,arm)
    finally:handle.remove()
    assert count==240 and bool(torch.isfinite(values).all());effect=values-values[0];native_anchor=float((values[0,:,:2]-prior[0,:,:2]).abs().max());legacy_full_gap=float((values[1,:,:2]-prior[1,:,:2]).abs().max());cells=[]
    for family in range(2):
        ix=[i for i,r in enumerate(rows) if r['family']==family];native=values[0,ix[::2],0]-values[0,ix[1::2],0];cue=effect[:,ix[::2],0]-effect[:,ix[1::2],0];full=cue[1];errs={ARMS[a]:rel(cue[a],full) for a in (2,3,4)};target=float(full.square().mean().sqrt());controls={ARMS[a]:{READOUTS[j][0]:float(effect[a,ix,j].square().mean().sqrt()/max(target,1e-8)) for j in range(2,6)} for a in range(1,5)}
        cells.append({'family':family,'native_positive_pairs':int((native>0).sum()),'native_cue_mean':float(native.mean()),'full_swap_target_rms':target,'cue_errors_to_full_swap':errs,'control_ratios_to_full_target':controls,'capability_pass':int((native>0).sum())>=10 and target>=1e-5,'additive_no_mixed_pass':errs['additive_no_mixed']<=.10,'value_hypothesis_pass':errs['value_swap']<=.35 and errs['routing_swap']>=.5,'routing_hypothesis_pass':errs['routing_swap']<=.35 and errs['value_swap']>=.5})
    torch.save({'values':values,'effects':effect},artifact);result={'pred_a':native_anchor<=1e-5 and max(replays)<=1e-10 and max(algebra)<=1e-10 and min(mixed_norms)>1e-8 and count==240 and bool(torch.isfinite(values).all()),'pred_b':all(c['capability_pass'] for c in cells),'pred_c':all(c['additive_no_mixed_pass'] for c in cells),'pred_d':all(c['value_hypothesis_pass'] for c in cells),'pred_e':all(c['routing_hypothesis_pass'] for c in cells),'max_native_anchor_error':native_anchor,'legacy_query_mutating_full_swap_gap':legacy_full_gap,'max_source_helper_replay_error':max(replays),'max_swap_expansion_error':max(algebra),'min_mixed_source_norm':min(mixed_norms),'families':cells,'body_forwards':count,'seconds':time.perf_counter()-start,'artifact_sha256':digest(artifact),'source_shas':binding,'arm_order':ARMS,'readout_order':[x[0] for x in READOUTS],'supersedes_control_only':'V1 incorrectly required all-query-fixed full swap to replay a legacy swap that mutated nonfinal query rows; B-E and all intervention values are unchanged.','scope':'Exact all-query-fixed routing/value/mixed decomposition of paired O framing-source interchange with native suffix. No fitting, upstream necessity, corpus OOD, compression adoption, or quantization.'};out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True);signal.alarm(0)

if __name__=='__main__':main()
