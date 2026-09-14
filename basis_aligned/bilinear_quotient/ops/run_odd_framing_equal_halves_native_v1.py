#!/usr/bin/env python3
# BQGATE:192bodyforwards;48prefixes;180seconds;no fitting.
"""pred_a exact halves, anchors, finite,192 forwards.
pred_b native capability and live framing effect each template.
pred_c early error<=.35 and late error>=.50 each template.
pred_d late error<=.35 and early error>=.50 each template.
"""
import json,os,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from run_even_value_factorial_native_v1 import setup,cpu_control
from odd_source_positions_v1 import source_channels,select_sources
from odd_framing_equal_halves_v1 import equal_half_masks
from sparse_path_stability_atlas_v1 import digest
from regional_cue_row_check_v1 import validate
STEM='ODD_FRAMING_EQUAL_HALVES_NATIVE_V1';ARM_ORDER=['native','remove_odd_early_half','remove_odd_late_half','remove_odd_framing']
READOUTS=[('target',None),('work_jobs',(670,3946)),('cat_dog',(3797,3290)),('red_blue',(2266,4171)),('monday_tuesday',(3321,3431)),('apple_orange',(17180,10912))]

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    rows=json.loads((P/'ODD_FRAMING_FRESH_V1_ROWS.json').read_text())['rows'];assert len(rows)==48;validate(rows);masks=equal_half_masks(rows);prior=torch.load(P/'ODD_FRAMING_FRESH_V1_ARTIFACT.pt',map_location='cpu',weights_only=True)['values']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        assert json.loads((P/'ODD_FRAMING_EQUAL_HALVES_V1_CPU_CONTROL.json').read_text())['pred_a'];print('192bodyforwards;48 frozen prefixes;CPUcontrol',cpu_control());return
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not artifact.exists();start=time.perf_counter();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();graph,_,_,_=setup('cuda');context={};checks=[];rel=lambda x,y:float((x-y).norm()/y.norm().clamp_min(1e-8))
    def hook(module,args,output):
        arm=context['arm']
        if arm==0:return output
        current,first=args;channels=source_channels(graph,current,first);pieces={k:select_sources(channels,v) for k,v in context['masks'].items()};checks.append(rel(pieces['early']+pieces['late'],pieces['framing']))
        delta={1:pieces['early'],2:pieces['late'],3:pieces['framing']}[arm]@graph.p['output'].double().T
        return output[0]-delta.to(output[0].dtype),output[1]
    handle=model.transformer.h[9].attn.register_forward_hook(hook);values=torch.zeros(4,48,6,dtype=torch.float64);count=0
    try:
        for i,row in enumerate(rows):
            ids=torch.tensor([row['ids']],device='cuda')
            for arm in range(4):
                context.update(arm=arm,masks={k:v[None].to('cuda') for k,v in masks[i].items()});x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
                for block in model.transformer.h:x,v1=block(x,v1,x0)
                scores=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];pairs=[(row['uk_id'],row['us_id'])]+[pair for _,pair in READOUTS[1:]]
                for j,(a,b) in enumerate(pairs):values[arm,i,j]=(scores[a]-scores[b]).cpu()
                count+=1
    finally:handle.remove()
    assert count==192 and bool(torch.isfinite(values).all());effect=values-values[0];anchor=max(float((values[0,:,:2]-prior[0]).abs().max()),float((values[3,:,:2]-prior[1]).abs().max()));cells=[]
    for family in range(2):
        ix=[i for i,r in enumerate(rows) if r['family']==family];native=values[0,ix[::2],0]-values[0,ix[1::2],0];cue=effect[:,ix[::2],0]-effect[:,ix[1::2],0];early_err=rel(cue[1],cue[3]);late_err=rel(cue[2],cue[3]);rms=[float(cue[a].square().mean().sqrt()) for a in (1,2,3)]
        controls={name:{READOUTS[j][0]:float(effect[arm,ix,j].square().mean().sqrt()/max(rms[arm-1],1e-8)) for j in range(2,6)} for arm,name in [(1,'early'),(2,'late')]}
        cells.append({'family':family,'native_positive_pairs':int((native>0).sum()),'native_cue_mean':float(native.mean()),'early_to_framing_cue_error':early_err,'late_to_framing_cue_error':late_err,'target_rms':{'early':rms[0],'late':rms[1],'framing':rms[2]},'new_control_ratios':controls,'separate_effect_composition_error':rel(cue[1]+cue[2],cue[3]),'capability_pass':int((native>0).sum())>=10 and rms[2]>=1e-5,'early_hypothesis_pass':early_err<=.35 and late_err>=.5,'late_hypothesis_pass':late_err<=.35 and early_err>=.5})
    torch.save({'values':values,'effects':effect},artifact);result={'pred_a':anchor<=1e-5 and max(checks)<=1e-10 and count==192 and bool(torch.isfinite(values).all()),'pred_b':all(c['capability_pass'] for c in cells),'pred_c':all(c['early_hypothesis_pass'] for c in cells),'pred_d':all(c['late_hypothesis_pass'] for c in cells),'max_frozen_anchor_error':anchor,'max_source_partition_error':max(checks),'families':cells,'body_forwards':count,'seconds':time.perf_counter()-start,'artifact_sha256':digest(artifact),'source_shas':binding,'arm_order':ARM_ORDER,'readout_order':[x[0] for x in READOUTS],'scope':'Frozen equal-count position split within fresh O framing sources; native context and suffix retained; no semantic identification, corpus OOD, static compression, or quantization claim.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True);signal.alarm(0)

if __name__=='__main__':main()
