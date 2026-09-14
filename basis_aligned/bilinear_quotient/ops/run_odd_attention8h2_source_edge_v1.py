#!/usr/bin/env python3
# BQGATE:192bodyforwards;48prefixes;180seconds;no fitting.
"""pred_a exact source/head replay and anchors;pred_b capability/live.
pred_c city source wins;pred_d other sources win;pred_e effects compose.
"""
import json,os,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from run_even_value_factorial_native_v1 import setup,cpu_control
from odd_source_positions_v1 import select_sources
from odd_semantic_positions_v1 import semantic_masks
from odd_contextual_positions_v1 import contextual_masks
from odd_source_swap_interaction_v1 import source_factors
from odd_value_source_split_v1 import value_parts
from sparse_path_stability_atlas_v1 import digest
from regional_cue_row_check_v1 import validate
STEM='ODD_ATTENTION8H2_SOURCE_EDGE_V1';ARMS=['native','full_head8_2','city_source','other_sources'];READOUTS=[('target',None),('work_jobs',(670,3946)),('cat_dog',(3797,3290)),('red_blue',(2266,4171)),('monday_tuesday',(3321,3431)),('apple_orange',(17180,10912))]

def head_channels(attn,current,first,head=2):
    from jacclust.tt_model import apply_rotary_emb
    B,T,C=current.shape;H=attn.n_head;D=attn.head_dim;q=attn.c_q(current).view(B,T,H,D);k=attn.c_k(current).view(B,T,H,D);q2=attn.c_q2(current).view(B,T,H,D);k2=attn.c_k2(current).view(B,T,H,D);v=attn.c_v(current).view(B,T,H,D);v=(1-attn.lamb)*v+attn.lamb*first.view_as(v);cos,sin=attn.rotary(q);q,k=apply_rotary_emb(F.rms_norm(q,(D,)),cos,sin),apply_rotary_emb(F.rms_norm(k,(D,)),cos,sin);q2,k2=apply_rotary_emb(F.rms_norm(q2,(D,)),cos,sin),apply_rotary_emb(F.rms_norm(k2,(D,)),cos,sin);a=torch.einsum('bqd,bkd->bqk',q[:,:,head],k[:,:,head])/D;b=torch.einsum('bqd,bkd->bqk',q2[:,:,head],k2[:,:,head])/D;pattern=(a*b).masked_fill(~torch.ones(T,T,dtype=torch.bool,device=current.device).tril(),0);return pattern[:,:,:,None]*v[:,None,:,head,:]

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items());rows=json.loads((P/'ODD_FRAMING_FRESH_V1_ROWS.json').read_text())['rows'];assert len(rows)==48;validate(rows);sem=semantic_masks(rows);ctx=contextual_masks(rows);prior=torch.load(P/'ODD_ATTENTION8_HEAD_WRITERS_V1_ARTIFACT.pt',map_location='cpu',weights_only=True)['values']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        assert json.loads((P/'ODD_ATTENTION8H2_SOURCE_EDGE_V1_CPU_CONTROL.json').read_text())['pred_a'];print('192bodyforwards;native/full head8.2/city/other;CPUcontrol',cpu_control());return
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not artifact.exists();start=time.perf_counter();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();graph,_,_,_=setup('cuda');context={'arm':0};a8inputs=[];preproj=[];r9=[];source_replay=[];partitions=[];outside=[];attn8=model.transformer.h[8].attn;weight=attn8.c_proj.weight[:,2*128:3*128]
    def a8pre(module,args):
        if context['arm']==0:a8inputs.append((args[0].detach().cpu(),args[1].detach().cpu()))
    def cpre(module,args):
        if context['arm']==0:preproj.append(args[0].detach().cpu())
    def b9pre(module,args):
        x,v1,x0=args;arm=context['arm']
        if arm==0:r9.append(x.detach().cpu());return None
        i=context['i'];d=i^1;framing=sem[i]['framing'][None].to(x.device);city=ctx[i]['city'][None].to(x.device);other=~city;xr,fr=a8inputs[i];xd,fd=a8inputs[d];xr,fr,xd,fd=xr.to(x.device),fr.to(x.device),xd.to(x.device),fd.to(x.device);cr=head_channels(attn8,xr,fr);cd=head_channels(attn8,xd,fd);full_r=cr.sum(-2);full_d=cd.sum(-2);city_r=select_sources(cr,city);city_d=select_sources(cd,city);other_r=select_sources(cr,other);other_d=select_sources(cd,other);source_replay.extend([float((full_r-preproj[i].to(x.device)[...,2*128:3*128]).norm()/full_r.norm().clamp_min(1e-8)),float((full_d-preproj[d].to(x.device)[...,2*128:3*128]).norm()/full_d.norm().clamp_min(1e-8))]);partitions.extend([float((city_r+other_r-full_r).norm()/full_r.norm().clamp_min(1e-8)),float((city_d+other_d-full_d).norm()/full_d.norm().clamp_min(1e-8))]);wr={1:F.linear(full_r,weight),2:F.linear(city_r,weight),3:F.linear(other_r,weight)}[arm];wd={1:F.linear(full_d,weight),2:F.linear(city_d,weight),3:F.linear(other_d,weight)}[arm];hybrid=x.clone();hybrid[:,framing[0]]+=(wd-wr)[:,framing[0]];outside.append(float((hybrid[:,~framing[0]]-x[:,~framing[0]]).abs().max()));mixed=module.lambdas[0]*hybrid+module.lambdas[1]*x0;context['hybrid_current']=F.rms_norm(mixed,(mixed.size(-1),));return None
    hs=[attn8.register_forward_pre_hook(a8pre),attn8.c_proj.register_forward_pre_hook(cpre),model.transformer.h[9].register_forward_pre_hook(b9pre)]
    def a9out(module,args,output):
        if context['arm']==0:return output
        current,first=args;mask=sem[context['i']]['framing'][None].to(current.device);routing,_=source_factors(graph,current,current,current,first);co,_=value_parts(graph,current,first);ch,_=value_parts(graph,context['hybrid_current'],first);delta=select_sources(routing*(ch-co),mask)@graph.p['output'].double().T;return output[0]+delta.to(output[0].dtype),output[1]
    hs.append(model.transformer.h[9].attn.register_forward_hook(a9out));values=torch.zeros(4,48,6,dtype=torch.float64);count=0
    def forward(i,arm):
        nonlocal count
        row=rows[i];ids=torch.tensor([row['ids']],device='cuda');x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
        for block in model.transformer.h:x,v1=block(x,v1,x0)
        scores=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];pairs=[(row['uk_id'],row['us_id'])]+[pair for _,pair in READOUTS[1:]]
        for j,(a,b) in enumerate(pairs):values[arm,i,j]=(scores[a]-scores[b]).cpu()
        count+=1
    try:
        for i in range(48):forward(i,0)
        assert len(a8inputs)==len(preproj)==len(r9)==48
        for arm in range(1,4):
            context['arm']=arm
            for i in range(48):context['i']=i;forward(i,arm)
    finally:
        for h in hs:h.remove()
    assert count==192 and bool(torch.isfinite(values).all());effect=values-values[0];anchor=max(float((values[0,:,:2]-prior[0,:,:2]).abs().max()),float((values[1,:,:2]-prior[4,:,:2]).abs().max()));cells=[]
    for family in range(2):
        ix=[i for i,r in enumerate(rows) if r['family']==family];native=values[0,ix[::2],0]-values[0,ix[1::2],0];cue=effect[:,ix[::2],0]-effect[:,ix[1::2],0];full=cue[1];city_err=float((cue[2]-full).norm()/full.norm().clamp_min(1e-8));other_err=float((cue[3]-full).norm()/full.norm().clamp_min(1e-8));composition=float((cue[2]+cue[3]-full).norm()/full.norm().clamp_min(1e-8));target=float(full.square().mean().sqrt());controls={ARMS[a]:{READOUTS[j][0]:float(effect[a,ix,j].square().mean().sqrt()/max(target,1e-8)) for j in range(2,6)} for a in range(1,4)};cells.append({'family':family,'native_positive_pairs':int((native>0).sum()),'native_cue_mean':float(native.mean()),'full_head_target_rms':target,'city_to_full_cue_error':city_err,'other_to_full_cue_error':other_err,'separate_effect_composition_error':composition,'control_ratios_to_full_target':controls,'capability_pass':int((native>0).sum())>=10 and target>=1e-5,'city_hypothesis_pass':city_err<=.35 and other_err>=.5,'other_hypothesis_pass':other_err<=.35 and city_err>=.5,'composition_pass':composition<=.1})
    torch.save({'values':values,'effects':effect},artifact);result={'pred_a':anchor<=1e-5 and max(source_replay)<=1e-5 and max(partitions)<=1e-10 and max(outside)==0 and count==192 and bool(torch.isfinite(values).all()),'pred_b':all(c['capability_pass'] for c in cells),'pred_c':all(c['city_hypothesis_pass'] for c in cells),'pred_d':all(c['other_hypothesis_pass'] for c in cells),'pred_e':all(c['composition_pass'] for c in cells),'max_frozen_anchor_error':anchor,'max_head_source_replay_error':max(source_replay),'max_city_other_partition_error':max(partitions),'max_outside_hybrid_delta':max(outside),'families':cells,'body_forwards':count,'seconds':time.perf_counter()-start,'artifact_sha256':digest(artifact),'source_shas':binding,'arm_order':ARMS,'readout_order':[x[0] for x in READOUTS],'scope':'Exact head8.2 city/other source-write donor hybrids evaluated through isolated head9.8-O current framing values and native suffix. No whole-head sufficiency, fitting, corpus OOD, compression adoption, or quantization.'};out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True);signal.alarm(0)

if __name__=='__main__':main()
