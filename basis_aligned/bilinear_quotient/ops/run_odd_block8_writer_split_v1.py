#!/usr/bin/env python3
# BQGATE:240bodyforwards;48prefixes;180seconds;no fitting.
"""pred_a exact block8 identity and anchors;pred_b capability/live.
pred_c attention wins;pred_d MLP wins;pred_e carry wins;pred_f effects compose.
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
STEM='ODD_BLOCK8_WRITER_SPLIT_V1';ARMS=['native','full_block8_delta','input_carry_delta','attention8_delta','mlp8_delta'];READOUTS=[('target',None),('work_jobs',(670,3946)),('cat_dog',(3797,3290)),('red_blue',(2266,4171)),('monday_tuesday',(3321,3431)),('apple_orange',(17180,10912))]

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items());rows=json.loads((P/'ODD_FRAMING_FRESH_V1_ROWS.json').read_text())['rows'];assert len(rows)==48;validate(rows);masks=semantic_masks(rows);prior=torch.load(P/'ODD_VALUE_SOURCE_SPLIT_NATIVE_V1_ARTIFACT.pt',map_location='cpu',weights_only=True)['values']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        assert all(torch.equal(torch.tensor(rows[i]['ids'])[masks[i]['framing']],torch.tensor(rows[i^1]['ids'])[masks[i^1]['framing']]) for i in range(48));print('240bodyforwards;block8 carry/attention/MLP split;CPUcontrol',cpu_control());return
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not artifact.exists();start=time.perf_counter();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();graph,_,_,_=setup('cuda');context={'arm':0};cache={k:[] for k in ['r8','x0','attn8','mlp8','r9']};identities=[];x0zeros=[];outside=[]
    def b8pre(module,args):
        if context['arm']==0:cache['r8'].append(args[0].detach().cpu());cache['x0'].append(args[2].detach().cpu())
    def a8out(module,args,output):
        if context['arm']==0:cache['attn8'].append(output[0].detach().cpu())
    def m8out(module,args,output):
        if context['arm']==0:cache['mlp8'].append(output.detach().cpu())
    def b9pre(module,args):
        x,v1,x0=args;arm=context['arm']
        if arm==0:cache['r9'].append(x.detach().cpu());return None
        i=context['i'];d=i^1;mask=masks[i]['framing'].to(x.device);r8r,r8d=cache['r8'][i].to(x.device),cache['r8'][d].to(x.device);x0r,x0d=cache['x0'][i].to(x.device),cache['x0'][d].to(x.device);a8r,a8d=cache['attn8'][i].to(x.device),cache['attn8'][d].to(x.device);m8r,m8d=cache['mlp8'][i].to(x.device),cache['mlp8'][d].to(x.device);r9r,r9d=cache['r9'][i].to(x.device),cache['r9'][d].to(x.device)
        carry=model.transformer.h[8].lambdas[0]*(r8d-r8r)+model.transformer.h[8].lambdas[1]*(x0d-x0r);attn=a8d-a8r;mlp=m8d-m8r;full=carry+attn+mlp;identities.append(float((full-(r9d-r9r)).norm()/(r9d-r9r).norm().clamp_min(1e-8)));x0zeros.append(float((x0d[:,mask]-x0r[:,mask]).abs().max()));delta={1:full,2:carry,3:attn,4:mlp}[arm];hybrid=x.clone();hybrid[:,mask]+=delta[:,mask];outside.append(float((hybrid[:,~mask]-x[:,~mask]).abs().max()));mixed=module.lambdas[0]*hybrid+module.lambdas[1]*x0;context['hybrid_current']=F.rms_norm(mixed,(mixed.size(-1),));return None
    hs=[model.transformer.h[8].register_forward_pre_hook(b8pre),model.transformer.h[8].attn.register_forward_hook(a8out),model.transformer.h[8].mlp.register_forward_hook(m8out),model.transformer.h[9].register_forward_pre_hook(b9pre)]
    def a9out(module,args,output):
        if context['arm']==0:return output
        current,first=args;mask=masks[context['i']]['framing'][None].to(current.device);routing,_=source_factors(graph,current,current,current,first);co,_=value_parts(graph,current,first);ch,_=value_parts(graph,context['hybrid_current'],first);delta=select_sources(routing*(ch-co),mask)@graph.p['output'].double().T;return output[0]+delta.to(output[0].dtype),output[1]
    hs.append(model.transformer.h[9].attn.register_forward_hook(a9out));values=torch.zeros(5,48,6,dtype=torch.float64);count=0
    def forward(i,arm):
        nonlocal count
        row=rows[i];ids=torch.tensor([row['ids']],device='cuda');x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
        for block in model.transformer.h:x,v1=block(x,v1,x0)
        scores=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];pairs=[(row['uk_id'],row['us_id'])]+[pair for _,pair in READOUTS[1:]]
        for j,(a,b) in enumerate(pairs):values[arm,i,j]=(scores[a]-scores[b]).cpu()
        count+=1
    try:
        for i in range(48):forward(i,0)
        assert all(len(v)==48 for v in cache.values())
        for arm in range(1,5):
            context['arm']=arm
            for i in range(48):context['i']=i;forward(i,arm)
    finally:
        for h in hs:h.remove()
    assert count==240 and bool(torch.isfinite(values).all());effect=values-values[0];anchor=max(float((values[0,:,:2]-prior[0,:,:2]).abs().max()),float((values[1,:,:2]-prior[2,:,:2]).abs().max()));cells=[]
    for family in range(2):
        ix=[i for i,r in enumerate(rows) if r['family']==family];native=values[0,ix[::2],0]-values[0,ix[1::2],0];cue=effect[:,ix[::2],0]-effect[:,ix[1::2],0];full=cue[1];errs={ARMS[a]:float((cue[a]-full).norm()/full.norm().clamp_min(1e-8)) for a in (2,3,4)};composition=float((cue[2]+cue[3]+cue[4]-full).norm()/full.norm().clamp_min(1e-8));target=float(full.square().mean().sqrt());controls={ARMS[a]:{READOUTS[j][0]:float(effect[a,ix,j].square().mean().sqrt()/max(target,1e-8)) for j in range(2,6)} for a in range(1,5)}
        cells.append({'family':family,'native_positive_pairs':int((native>0).sum()),'native_cue_mean':float(native.mean()),'full_target_rms':target,'cue_errors_to_full':errs,'separate_effect_composition_error':composition,'control_ratios_to_full_target':controls,'capability_pass':int((native>0).sum())>=10 and target>=1e-5,'attention_hypothesis_pass':errs['attention8_delta']<=.35 and errs['input_carry_delta']>=.5 and errs['mlp8_delta']>=.5,'mlp_hypothesis_pass':errs['mlp8_delta']<=.35 and errs['input_carry_delta']>=.5 and errs['attention8_delta']>=.5,'carry_hypothesis_pass':errs['input_carry_delta']<=.35 and errs['attention8_delta']>=.5 and errs['mlp8_delta']>=.5,'behavioral_composition_pass':composition<=.1})
    torch.save({'values':values,'effects':effect},artifact);result={'pred_a':anchor<=1e-5 and max(identities)<=1e-10 and max(x0zeros)==0 and max(outside)==0 and count==240 and bool(torch.isfinite(values).all()),'pred_b':all(c['capability_pass'] for c in cells),'pred_c':all(c['attention_hypothesis_pass'] for c in cells),'pred_d':all(c['mlp_hypothesis_pass'] for c in cells),'pred_e':all(c['carry_hypothesis_pass'] for c in cells),'pred_f':all(c['behavioral_composition_pass'] for c in cells),'max_frozen_anchor_error':anchor,'max_block8_residual_identity_error':max(identities),'max_framing_x0_delta':max(x0zeros),'max_outside_hybrid_delta':max(outside),'families':cells,'body_forwards':count,'seconds':time.perf_counter()-start,'artifact_sha256':digest(artifact),'source_shas':binding,'arm_order':ARMS,'readout_order':[x[0] for x in READOUTS],'scope':'Exact block8 carry/attention/MLP donor-component hybrids evaluated only through isolated O current framing values and native suffix. No broad residual patch, fitting, corpus OOD, compression adoption, or quantization.'};out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True);signal.alarm(0)

if __name__=='__main__':main()
