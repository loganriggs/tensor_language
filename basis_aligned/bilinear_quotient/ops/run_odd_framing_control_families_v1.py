#!/usr/bin/env python3
# BQGATE:96bodyforwards;48prefixes;180seconds;no fitting.
"""pred_a frozen-arm replay and exact source instrument.
pred_b native capability and live framing target effect.
pred_c median new-control ratio<=.5 and >=3/4 ratios<=.5 each template.
pred_d maximum new-control ratio<=1 each template.
Price96bodyforwards,48prefixes,180seconds; all native weights retained.
"""
import json,os,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from run_even_value_factorial_native_v1 import setup,cpu_control
from odd_source_positions_v1 import source_channels,select_sources
from odd_semantic_positions_v1 import semantic_masks
from sparse_path_stability_atlas_v1 import digest
from regional_cue_row_check_v1 import validate
STEM='ODD_FRAMING_CONTROL_FAMILIES_V1';ARM_ORDER=['native','remove_odd_framing']
READOUTS=[('target',None),('work_jobs',(670,3946)),('cat_dog',(3797,3290)),('red_blue',(2266,4171)),('monday_tuesday',(3321,3431)),('apple_orange',(17180,10912))]


@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    rows=json.loads((P/'ODD_FRAMING_FRESH_V1_ROWS.json').read_text())['rows'];assert len(rows)==48;validate(rows);masks=semantic_masks(rows)
    prior=torch.load(P/'ODD_FRAMING_FRESH_V1_ARTIFACT.pt',map_location='cpu',weights_only=True)['values']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        old=json.loads((P/'ODD_FRAMING_FRESH_V1_RESULT.json').read_text());assert old['pred_a'] and old['pred_b'] and old['pred_c'] and not old['pred_d']
        assert prior.shape==(5,48,2);print('96bodyforwards;48 frozen prefixes; CPUcontrol',cpu_control());return
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not artifact.exists()
    start=time.perf_counter();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();graph,_,_,_=setup('cuda');context={};checks=[]
    rel=lambda x,y:float((x-y).norm()/y.norm().clamp_min(1e-8))
    def hook(module,args,output):
        if context['arm']==0:return output
        current,first=args;channels=source_channels(graph,current,first);pieces={k:select_sources(channels,v) for k,v in context['masks'].items()}
        states=graph.state(current,first_values=first.reshape(*current.shape[:2],9,128)[:,:,8]);checks.append(rel(sum(pieces.values()),states[2]))
        delta=pieces['framing']@graph.p['output'].double().T
        return output[0]-delta.to(output[0].dtype),output[1]
    handle=model.transformer.h[9].attn.register_forward_hook(hook);values=torch.zeros(2,48,6,dtype=torch.float64);count=0
    try:
        for i,row in enumerate(rows):
            ids=torch.tensor([row['ids']],device='cuda')
            for arm in range(2):
                context.update(arm=arm,masks={k:v[None].to('cuda') for k,v in masks[i].items()})
                x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
                for block in model.transformer.h:x,v1=block(x,v1,x0)
                scores=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
                pairs=[(row['uk_id'],row['us_id'])]+[pair for _,pair in READOUTS[1:]]
                for j,(a,b) in enumerate(pairs):values[arm,i,j]=(scores[a]-scores[b]).cpu()
                count+=1
    finally:handle.remove()
    assert count==96 and bool(torch.isfinite(values).all());effect=values-values[0]
    anchor=float((values[:,:,:2]-prior[:2]).abs().max());cells=[]
    for family in range(2):
        ix=[i for i,r in enumerate(rows) if r['family']==family];native=values[0,ix[::2],0]-values[0,ix[1::2],0]
        target=effect[1,ix[::2],0]-effect[1,ix[1::2],0];target_rms=float(target.square().mean().sqrt())
        ratios=[]
        for j in range(2,6):ratios.append(float(effect[1,ix,j].square().mean().sqrt()/max(target_rms,1e-8)))
        ordered=sorted(ratios);median=(ordered[1]+ordered[2])/2
        cells.append({'family':family,'native_positive_pairs':int((native>0).sum()),'native_cue_mean':float(native.mean()),'framing_target_rms':target_rms,'new_control_ratios':{READOUTS[j][0]:ratios[j-2] for j in range(2,6)},'median_new_control_ratio':median,'new_controls_at_most_half':sum(x<=.5 for x in ratios),'capability_and_live_pass':int((native>0).sum())>=10 and target_rms>=1e-5,'multi_control_selectivity_pass':median<=.5 and sum(x<=.5 for x in ratios)>=3,'max_control_robustness_pass':max(ratios)<=1})
    torch.save({'values':values,'effects':effect},artifact)
    result={'pred_a':anchor<=1e-5 and max(checks)<=1e-10 and count==96 and bool(torch.isfinite(values).all()),'pred_b':all(c['capability_and_live_pass'] for c in cells),'pred_c':all(c['multi_control_selectivity_pass'] for c in cells),'pred_d':all(c['max_control_robustness_pass'] for c in cells),'max_frozen_anchor_error':anchor,'max_source_recomposition_error':max(checks),'families':cells,'body_forwards':count,'seconds':time.perf_counter()-start,'artifact_sha256':digest(artifact),'source_shas':binding,'arm_order':ARM_ORDER,'readout_order':[x[0] for x in READOUTS],'scope':'Frozen multi-control diagnostic on the fresh O framing screen. The prior work/jobs failure is retained. Native context and suffix remain; no unique semantic unit, corpus OOD, static compression, or quantization claim.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True);signal.alarm(0)


if __name__=='__main__':main()
