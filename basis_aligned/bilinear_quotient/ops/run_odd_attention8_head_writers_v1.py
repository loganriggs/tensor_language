#!/usr/bin/env python3
# BQGATE:528bodyforwards;48prefixes;180seconds;no fitting.
"""pred_a exact head-write sum and anchors;pred_b capability/live.
pred_c one common sufficient head;pred_d separate head effects compose.
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
STEM='ODD_ATTENTION8_HEAD_WRITERS_V1';ARMS=['native','full_attention8']+[f'attention8_head_{h}' for h in range(9)];READOUTS=[('target',None),('work_jobs',(670,3946)),('cat_dog',(3797,3290)),('red_blue',(2266,4171)),('monday_tuesday',(3321,3431)),('apple_orange',(17180,10912))]

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items());rows=json.loads((P/'ODD_FRAMING_FRESH_V1_ROWS.json').read_text())['rows'];assert len(rows)==48;validate(rows);masks=semantic_masks(rows);prior=torch.load(P/'ODD_BLOCK8_WRITER_SPLIT_V1_ARTIFACT.pt',map_location='cpu',weights_only=True)['values']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print('528bodyforwards;native+full attention8+9 head writes;CPUcontrol',cpu_control());return
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not artifact.exists();start=time.perf_counter();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();graph,_,_,_=setup('cuda');context={'arm':0};preproj=[];attn8=[];r9=[];headsum=[];outside=[];weight=model.transformer.h[8].attn.c_proj.weight
    def proj_heads(y):return [F.linear(y[...,h*128:(h+1)*128],weight[:,h*128:(h+1)*128]) for h in range(9)]
    def cpre(module,args):
        if context['arm']==0:preproj.append(args[0].detach().cpu())
    def a8out(module,args,output):
        if context['arm']==0:attn8.append(output[0].detach().cpu())
    def b9pre(module,args):
        x,v1,x0=args;arm=context['arm']
        if arm==0:r9.append(x.detach().cpu());return None
        i=context['i'];d=i^1;mask=masks[i]['framing'].to(x.device);yr,yd=preproj[i].to(x.device),preproj[d].to(x.device);heads_r,heads_d=proj_heads(yr),proj_heads(yd);full_r=sum(heads_r);full_d=sum(heads_d);headsum.append(float((full_r-attn8[i].to(x.device)).norm()/attn8[i].to(x.device).norm().clamp_min(1e-8)));delta=(full_d-full_r) if arm==1 else (heads_d[arm-2]-heads_r[arm-2]);hybrid=x.clone();hybrid[:,mask]+=delta[:,mask];outside.append(float((hybrid[:,~mask]-x[:,~mask]).abs().max()));mixed=module.lambdas[0]*hybrid+module.lambdas[1]*x0;context['hybrid_current']=F.rms_norm(mixed,(mixed.size(-1),));return None
    hs=[model.transformer.h[8].attn.c_proj.register_forward_pre_hook(cpre),model.transformer.h[8].attn.register_forward_hook(a8out),model.transformer.h[9].register_forward_pre_hook(b9pre)]
    def a9out(module,args,output):
        if context['arm']==0:return output
        current,first=args;mask=masks[context['i']]['framing'][None].to(current.device);routing,_=source_factors(graph,current,current,current,first);co,_=value_parts(graph,current,first);ch,_=value_parts(graph,context['hybrid_current'],first);delta=select_sources(routing*(ch-co),mask)@graph.p['output'].double().T;return output[0]+delta.to(output[0].dtype),output[1]
    hs.append(model.transformer.h[9].attn.register_forward_hook(a9out));values=torch.zeros(11,48,6,dtype=torch.float64);count=0
    def forward(i,arm):
        nonlocal count
        row=rows[i];ids=torch.tensor([row['ids']],device='cuda');x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
        for block in model.transformer.h:x,v1=block(x,v1,x0)
        scores=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];pairs=[(row['uk_id'],row['us_id'])]+[pair for _,pair in READOUTS[1:]]
        for j,(a,b) in enumerate(pairs):values[arm,i,j]=(scores[a]-scores[b]).cpu()
        count+=1
    try:
        for i in range(48):forward(i,0)
        assert len(preproj)==len(attn8)==len(r9)==48
        for arm in range(1,11):
            context['arm']=arm
            for i in range(48):context['i']=i;forward(i,arm)
    finally:
        for h in hs:h.remove()
    assert count==528 and bool(torch.isfinite(values).all());effect=values-values[0];anchor=max(float((values[0,:,:2]-prior[0,:,:2]).abs().max()),float((values[1,:,:2]-prior[3,:,:2]).abs().max()));cells=[]
    for family in range(2):
        ix=[i for i,r in enumerate(rows) if r['family']==family];native=values[0,ix[::2],0]-values[0,ix[1::2],0];cue=effect[:,ix[::2],0]-effect[:,ix[1::2],0];full=cue[1];target=float(full.square().mean().sqrt());heads=[]
        for h in range(9):
            arm=h+2;err=float((cue[arm]-full).norm()/full.norm().clamp_min(1e-8));controls={READOUTS[j][0]:float(effect[arm,ix,j].square().mean().sqrt()/max(target,1e-8)) for j in range(2,6)};heads.append({'head':h,'cue_error_to_full_attention':err,'control_ratios_to_full_target':controls})
        composition=float((cue[2:].sum(0)-full).norm()/full.norm().clamp_min(1e-8));cells.append({'family':family,'native_positive_pairs':int((native>0).sum()),'native_cue_mean':float(native.mean()),'full_attention_target_rms':target,'heads':heads,'head_effect_composition_error':composition,'capability_pass':int((native>0).sum())>=10 and target>=1e-5,'composition_pass':composition<=.1})
    common=[h for h in range(9) if all(c['heads'][h]['cue_error_to_full_attention']<=.35 for c in cells)];torch.save({'values':values,'effects':effect},artifact);result={'pred_a':anchor<=1e-5 and max(headsum)<=1e-5 and max(outside)==0 and count==528 and bool(torch.isfinite(values).all()),'pred_b':all(c['capability_pass'] for c in cells),'pred_c':bool(common),'pred_d':all(c['composition_pass'] for c in cells),'common_sufficient_heads':common,'max_frozen_anchor_error':anchor,'max_headsum_attention_error':max(headsum),'max_outside_hybrid_delta':max(outside),'families':cells,'body_forwards':count,'seconds':time.perf_counter()-start,'artifact_sha256':digest(artifact),'source_shas':binding,'arm_order':ARMS,'readout_order':[x[0] for x in READOUTS],'scope':'Exact attention8 c_proj head-write donor hybrids evaluated only through isolated O current framing values and native suffix. No whole-attention sufficiency, fitting, corpus OOD, compression adoption, or quantization.'};out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True);signal.alarm(0)

if __name__=='__main__':main()
