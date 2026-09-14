#!/usr/bin/env python3
# BQGATE:528bodyforwards;48prefixes;180seconds;no fitting.
"""pred_a layer0 zero, frozen native/layer9 anchors, exact restore,528 forwards.
pred_b capability/live;pred_c early onset;pred_d opposing late onset.
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
STEM='ODD_CURRENT_VALUE_LAYER_ONSET_V1';LAYERS=list(range(10));READOUTS=[('target',None),('work_jobs',(670,3946)),('cat_dog',(3797,3290)),('red_blue',(2266,4171)),('monday_tuesday',(3321,3431)),('apple_orange',(17180,10912))]

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items());rows=json.loads((P/'ODD_FRAMING_FRESH_V1_ROWS.json').read_text())['rows'];assert len(rows)==48;validate(rows);masks=semantic_masks(rows);prior=torch.load(P/'ODD_VALUE_SOURCE_SPLIT_NATIVE_V1_ARTIFACT.pt',map_location='cpu',weights_only=True)['values']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        assert all(torch.equal(torch.tensor(rows[i]['ids'])[masks[i]['framing']],torch.tensor(rows[i^1]['ids'])[masks[i^1]['framing']]) for i in range(48));print('528bodyforwards;48 native+10x48 layer onset;CPUcontrol',cpu_control());return
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not artifact.exists();start=time.perf_counter();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();graph,_,_,_=setup('cuda');context={'arm':0};residual=[[] for _ in range(10)];attn_current=[];restores=[];outside=[]
    def block_hook(layer):
        def hook(module,args):
            x,v1,x0=args;arm=context['arm']
            if arm==0:residual[layer].append(x.detach().cpu());return None
            if layer==context['layer']:
                mask=masks[context['i']]['framing'].to(x.device);donor=residual[layer][context['i']^1].to(x.device);patched=x.clone();patched[:,mask]=donor[:,mask];outside.append(float((patched[:,~mask]-x[:,~mask]).abs().max()));x=patched
            if layer==9:
                mixed=module.lambdas[0]*x+module.lambdas[1]*x0;context['propagated_current']=F.rms_norm(mixed,(mixed.size(-1),));x=residual[9][context['i']].to(x.device)
            return x,v1,x0
        return hook
    block_handles=[model.transformer.h[l].register_forward_pre_hook(block_hook(l)) for l in LAYERS]
    def attn_hook(module,args,output):
        current,first=args;arm=context['arm']
        if arm==0:attn_current.append(current.detach().cpu());return output
        baseline=attn_current[context['i']].to(current.device);restores.append(float((current-baseline).abs().max()));mask=masks[context['i']]['framing'][None].to(current.device);routing,_=source_factors(graph,current,current,current,first);co,_=value_parts(graph,current,first);cp,_=value_parts(graph,context['propagated_current'],first);delta=select_sources(routing*(cp-co),mask)@graph.p['output'].double().T
        return output[0]+delta.to(output[0].dtype),output[1]
    attn_handle=model.transformer.h[9].attn.register_forward_hook(attn_hook);values=torch.zeros(11,48,6,dtype=torch.float64);count=0
    def forward(i,arm):
        nonlocal count
        row=rows[i];ids=torch.tensor([row['ids']],device='cuda');x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
        for block in model.transformer.h:x,v1=block(x,v1,x0)
        scores=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];pairs=[(row['uk_id'],row['us_id'])]+[pair for _,pair in READOUTS[1:]]
        for j,(a,b) in enumerate(pairs):values[arm,i,j]=(scores[a]-scores[b]).cpu()
        count+=1
    try:
        for i in range(48):forward(i,0)
        assert all(len(x)==48 for x in residual) and len(attn_current)==48
        for layer in LAYERS:
            context.update(arm=layer+1,layer=layer)
            for i in range(48):context['i']=i;forward(i,layer+1)
    finally:
        attn_handle.remove()
        for h in block_handles:h.remove()
    assert count==528 and bool(torch.isfinite(values).all());effect=values-values[0];anchor=max(float((values[0,:,:2]-prior[0,:,:2]).abs().max()),float((values[10,:,:2]-prior[2,:,:2]).abs().max()));cells=[]
    for family in range(2):
        ix=[i for i,r in enumerate(rows) if r['family']==family];native=values[0,ix[::2],0]-values[0,ix[1::2],0];cue=effect[:,ix[::2],0]-effect[:,ix[1::2],0];reference=cue[10];den=float(reference.norm());profiles=[]
        for layer in LAYERS:
            v=cue[layer+1];ratio=float(v.norm()/max(den,1e-8));cos=float((v@reference)/(v.norm()*reference.norm()).clamp_min(1e-8));controls={READOUTS[j][0]:float(effect[layer+1,ix,j].square().mean().sqrt()/(reference.square().mean().sqrt().clamp_min(1e-8))) for j in range(2,6)};profiles.append({'layer':layer,'target_norm_ratio_to_layer9':ratio,'target_cosine_to_layer9':cos,'control_ratios_to_layer9_target':controls})
        early=any(x['target_norm_ratio_to_layer9']>=.5 and x['target_cosine_to_layer9']>=.8 for x in profiles[:5]);late=all(x['target_norm_ratio_to_layer9']<.2 for x in profiles[:7]) and profiles[8]['target_norm_ratio_to_layer9']>=.8 and profiles[8]['target_cosine_to_layer9']>=.8
        cells.append({'family':family,'native_positive_pairs':int((native>0).sum()),'native_cue_mean':float(native.mean()),'layer9_target_rms':float(reference.square().mean().sqrt()),'profiles':profiles,'capability_pass':int((native>0).sum())>=10 and float(reference.square().mean().sqrt())>=1e-5,'early_onset_pass':early,'late_onset_pass':late})
    torch.save({'values':values,'effects':effect},artifact);result={'pred_a':anchor<=1e-5 and float(effect[1,:,0].abs().max())<=1e-8 and max(restores)==0 and max(outside)==0 and count==528 and bool(torch.isfinite(values).all()),'pred_b':all(c['capability_pass'] for c in cells),'pred_c':all(c['early_onset_pass'] for c in cells),'pred_d':all(c['late_onset_pass'] for c in cells),'max_frozen_anchor_error':anchor,'max_layer0_target_effect':float(effect[1,:,0].abs().max()),'max_restored_attn_input_error':max(restores),'max_outside_patch_error':max(outside),'families':cells,'body_forwards':count,'seconds':time.perf_counter()-start,'artifact_sha256':digest(artifact),'source_shas':binding,'arm_order':['native']+[f'layer_{x}_source_onset' for x in LAYERS],'readout_order':[x[0] for x in READOUTS],'scope':'Layer-onset trace for paired framing information entering O current values; recipient block9 residual/query/key and native suffix restored. No generic residual-patch, semantic identification, corpus OOD, compression adoption, or quantization claim.'};out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True);signal.alarm(0)

if __name__=='__main__':main()
