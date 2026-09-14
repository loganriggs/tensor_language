#!/usr/bin/env python3
# BQGATE:192bodyforwards;48prefixes;180seconds;no fitting.
"""pred_a native/all-O anchors<=1e-5 and source recomposition<=1e-10.
pred_b city-only cue error versus all-O<=.20 EACHfamily.
pred_c all-O/fullhead cue>=.05 and city control/target RMS<=.50 EACHfamily.
Null: O cue influence is contextual/distributed or not selectively material.
Price192bodyforwards,48prefixes,180seconds; all native weights retained.
"""
import json
import os
import signal
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from run_even_value_factorial_native_v1 import setup,cpu_control
from odd_source_positions_v1 import source_channels,select_sources
from inherited_source_positions_v1 import paired_masks
from sparse_path_stability_atlas_v1 import digest
from regional_cue_row_check_v1 import validate

STEM='ODD_SOURCE_POSITIONS_NATIVE_V1'


@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
    assert all(digest(k)==v for k,v in binding.items())
    rows=json.loads((P/'SRO_ARTICLE_CORRECTION_V1_ROWS.json').read_text())['rows']
    assert len(rows)==48;validate(rows);masks=paired_masks(rows)
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        assert json.loads((P/'ODD_SOURCE_POSITIONS_V1_CPU_CONTROL.json').read_text())['pred_a']
        print('192bodyforwards;48prefixes; odd-source CPUcontrol',cpu_control());return
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_ARTIFACT.pt')
    assert not out.exists() and not artifact.exists()
    start=time.perf_counter();signal.alarm(180);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();graph,_,_,_=setup('cuda')
    context={};checks=[]
    rel=lambda x,y:float((x-y).norm()/y.norm().clamp_min(1e-8))
    def hook(module,args,output):
        arm=context['arm']
        if not arm:return output
        current=args[0];first=args[1]
        channels=source_channels(graph,current,first)
        city=select_sources(channels,context['source_mask']);other=select_sources(channels,~context['source_mask'])
        reference=graph.state(current,first_values=first.reshape(*current.shape[:2],9,128)[:,:,8])[2]
        checks.append(rel(city+other,reference))
        selected=city if arm==1 else other if arm==2 else city+other
        delta=selected@graph.p['output'].double().T
        return output[0]-delta.to(output[0].dtype),output[1]
    handle=model.transformer.h[9].attn.register_forward_hook(hook)
    values=torch.zeros(4,48,2,dtype=torch.float64);count=0
    try:
        for i,row in enumerate(rows):
            ids=torch.tensor([row['ids']],device='cuda')
            for arm in range(4):
                context.update(arm=arm,source_mask=masks[i][None].to('cuda'))
                x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
                for block in model.transformer.h:x,v1=block(x,v1,x0)
                scores=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
                values[arm,i,0]=(scores[row['uk_id']]-scores[row['us_id']]).cpu()
                values[arm,i,1]=(scores[row['control_ids'][0]]-scores[row['control_ids'][1]]).cpu()
                count+=1
    finally:handle.remove()
    assert count==192
    old=torch.load(P/'SRO_ARTICLE_CORRECTION_V1_ARTIFACT.pt',weights_only=True)['cube']
    anchors={'native':rel(values[0],old[0]),'all_odd':rel(values[3],old[4])}
    effect=values-values[0];cells=[]
    for family in range(2):
        ix=[i for i,r in enumerate(rows) if r['family']==family]
        cue=effect[:,ix[::2],0]-effect[:,ix[1::2],0];control=effect[:,ix,1]
        city_error=rel(cue[1],cue[3]);head=old[7]-old[0]
        headcue=head[ix[::2],0]-head[ix[1::2],0]
        materiality=float(cue[3].norm()/headcue.norm().clamp_min(1e-8))
        target_rms=float(cue[1].square().mean().sqrt());control_rms=float(control[1].square().mean().sqrt())
        selectivity=control_rms/max(target_rms,1e-8)
        cells.append({'family':family,'city_only_cue_error':city_error,'all_odd_to_fullhead_cue':materiality,'city_target_cue_rms':target_rms,'city_control_rms':control_rms,'city_control_to_target_rms':selectivity,'cue_norms':[float(x.norm()) for x in cue[1:]],'control_norms':[float(x.norm()) for x in control[1:]],'city_pass':city_error<=.2,'selectivity_pass':materiality>=.05 and selectivity<=.5})
    torch.save({'values':values,'effects':effect},artifact)
    result={'pred_a':max(anchors.values())<=1e-5 and max(checks)<=1e-10,'pred_b':all(c['city_pass'] for c in cells),'pred_c':all(c['selectivity_pass'] for c in cells),'anchors':anchors,'max_source_recomposition_error':max(checks),'families':cells,'body_forwards':count,'seconds':time.perf_counter()-start,'artifact_sha256':digest(artifact),'source_shas':binding,'arm_order':['native','remove_odd_city_sources','remove_odd_other_sources','remove_all_odd'],'scope':'Source-position localization of reflection-odd head9.8 branch on corrected panel. Query and all native context remain; city mask does not remove upstream descendants. No semantic identification, static compression, OOD or quantization claim.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True);signal.alarm(0)


if __name__=='__main__':main()
