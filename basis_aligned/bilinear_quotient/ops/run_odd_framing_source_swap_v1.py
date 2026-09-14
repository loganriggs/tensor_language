#!/usr/bin/env python3
# BQGATE:96bodyforwards;48prefixes;180seconds;no fitting.
"""pred_a native anchors, exact O source instrument, framing-only patch,96 forwards.
pred_b native capability and live paired swap effect each template.
pred_c swap cue ~=2*frozen framing-removal cue within.35 and cosine>=.8.
pred_d four control swap RMS/target swap RMS<=.5 each template.
"""
import json,os,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from run_even_value_factorial_native_v1 import setup,cpu_control
from odd_source_positions_v1 import source_channels,select_sources
from odd_semantic_positions_v1 import semantic_masks
from sparse_path_stability_atlas_v1 import digest
from regional_cue_row_check_v1 import validate
STEM='ODD_FRAMING_SOURCE_SWAP_V1';READOUTS=[('target',None),('work_jobs',(670,3946)),('cat_dog',(3797,3290)),('red_blue',(2266,4171)),('monday_tuesday',(3321,3431)),('apple_orange',(17180,10912))]

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    rows=json.loads((P/'ODD_FRAMING_FRESH_V1_ROWS.json').read_text())['rows'];assert len(rows)==48;validate(rows);masks=semantic_masks(rows);prior=torch.load(P/'ODD_FRAMING_FRESH_V1_ARTIFACT.pt',map_location='cpu',weights_only=True)['values']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        assert prior.shape==(5,48,2);print('96bodyforwards;48 captures+48 swaps;CPUcontrol',cpu_control());return
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not artifact.exists();start=time.perf_counter();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();graph,_,_,_=setup('cuda');context={'mode':'capture'};cache=[];checks=[];outside=[]
    rel=lambda x,y:float((x-y).norm()/y.norm().clamp_min(1e-8))
    def hook(module,args,output):
        current,first=args
        if context['mode']=='capture':cache.append((current.detach().cpu(),first.detach().cpu()));return output
        i=context['i'];mask=masks[i]['framing'][None].to(current.device);donor_current,donor_first=cache[i^1];patched_current=current.clone();patched_first=first.clone();patched_current[:,mask[0]]=donor_current.to(current.device)[:,mask[0]];patched_first[:,mask[0]]=donor_first.to(first.device)[:,mask[0]]
        outside.append(max(float((patched_current[:,~mask[0]]-current[:,~mask[0]]).abs().max()),float((patched_first[:,~mask[0]]-first[:,~mask[0]]).abs().max())))
        original=source_channels(graph,current,first);patched=source_channels(graph,patched_current,patched_first);state=graph.state(current,first_values=first.reshape(*current.shape[:2],9,128)[:,:,8])[2];checks.append(rel(original.sum(-2),state))
        delta=(select_sources(patched,mask)-select_sources(original,mask))@graph.p['output'].double().T
        return output[0]+delta.to(output[0].dtype),output[1]
    handle=model.transformer.h[9].attn.register_forward_hook(hook);values=torch.zeros(2,48,6,dtype=torch.float64);count=0
    def forward(i,arm):
        nonlocal count
        row=rows[i];ids=torch.tensor([row['ids']],device='cuda');x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
        for block in model.transformer.h:x,v1=block(x,v1,x0)
        scores=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];pairs=[(row['uk_id'],row['us_id'])]+[pair for _,pair in READOUTS[1:]]
        for j,(a,b) in enumerate(pairs):values[arm,i,j]=(scores[a]-scores[b]).cpu()
        count+=1
    try:
        for i in range(48):forward(i,0)
        assert len(cache)==48;context['mode']='swap'
        for i in range(48):context['i']=i;forward(i,1)
    finally:handle.remove()
    assert count==96 and bool(torch.isfinite(values).all());effect=values[1]-values[0];anchor=float((values[0,:,:2]-prior[0]).abs().max());cells=[]
    for family in range(2):
        ix=[i for i,r in enumerate(rows) if r['family']==family];native=values[0,ix[::2],0]-values[0,ix[1::2],0];swap=effect[ix[::2],0]-effect[ix[1::2],0];removal=(prior[1,ix[::2],0]-prior[0,ix[::2],0])-(prior[1,ix[1::2],0]-prior[0,ix[1::2],0]);reference=2*removal
        error=rel(swap,reference);cosine=float((swap@reference)/(swap.norm()*reference.norm()).clamp_min(1e-8));target_rms=float(swap.square().mean().sqrt());ratios={READOUTS[j][0]:float(effect[ix,j].square().mean().sqrt()/max(target_rms,1e-8)) for j in range(2,6)}
        cells.append({'family':family,'native_positive_pairs':int((native>0).sum()),'native_cue_mean':float(native.mean()),'swap_target_rms':target_rms,'swap_to_twice_removal_error':error,'swap_to_twice_removal_cosine':cosine,'control_ratios':ratios,'capability_pass':int((native>0).sum())>=10 and target_rms>=1e-5,'transport_pass':error<=.35 and cosine>=.8,'selectivity_pass':all(x<=.5 for x in ratios.values())})
    torch.save({'values':values,'swap_effects':effect},artifact);result={'pred_a':anchor<=1e-5 and max(checks)<=1e-10 and max(outside)==0 and count==96 and bool(torch.isfinite(values).all()),'pred_b':all(c['capability_pass'] for c in cells),'pred_c':all(c['transport_pass'] for c in cells),'pred_d':all(c['selectivity_pass'] for c in cells),'max_native_anchor_error':anchor,'max_source_recomposition_error':max(checks),'max_outside_framing_patch':max(outside),'families':cells,'body_forwards':count,'seconds':time.perf_counter()-start,'artifact_sha256':digest(artifact),'source_shas':binding,'scope':'Paired framing-source state interchange only inside O at head9.8. Recipient query, other branches and native suffix retained; no upstream necessity, corpus OOD, static compression, or quantization claim.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True);signal.alarm(0)

if __name__=='__main__':main()
