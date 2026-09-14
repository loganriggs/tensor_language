#!/usr/bin/env python3
# BQGATE:240bodyforwards;48prefixes;180seconds;no fitting.
"""pred_a exact O source reconstruction<=1e-10; finite;240forwards.
pred_b native cue>=10/12 and O/fullhead cue>=.05 EACHtemplate.
pred_c framing error<=.35 and clause error>=.50 EACHtemplate.
pred_d framing control/target RMS<=.50 EACHtemplate.
Null: framing relay fails capability, materiality, transfer or selectivity.
Price240bodyforwards,48prefixes,180seconds; all native weights retained.
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
STEM='ODD_FRAMING_FRESH_V1';ARM_ORDER=['native','remove_odd_framing','remove_odd_clause','remove_all_odd','remove_full_head']


@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    rows=json.loads((P/'ODD_FRAMING_FRESH_V1_ROWS.json').read_text())['rows'];assert len(rows)==48;validate(rows);masks=semantic_masks(rows)
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        assert json.loads((P/'ODD_SEMANTIC_POSITIONS_V1_CPU_CONTROL_V2.json').read_text())['pred_a']
        print('240bodyforwards;48 fresh prefixes; framing CPUcontrol',cpu_control());return
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not artifact.exists()
    start=time.perf_counter();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();graph,_,_,_=setup('cuda');context={};checks=[]
    rel=lambda x,y:float((x-y).norm()/y.norm().clamp_min(1e-8))
    def hook(module,args,output):
        arm=context['arm']
        if not arm:return output
        current=args[0];first=args[1];channels=source_channels(graph,current,first)
        pieces={k:select_sources(channels,v) for k,v in context['masks'].items()};states=graph.state(current,first_values=first.reshape(*current.shape[:2],9,128)[:,:,8])
        checks.append(rel(sum(pieces.values()),states[2]))
        selected={1:pieces['framing'],2:pieces['clause'],3:sum(pieces.values())}.get(arm)
        delta=graph.write(states) if arm==4 else selected@graph.p['output'].double().T
        return output[0]-delta.to(output[0].dtype),output[1]
    handle=model.transformer.h[9].attn.register_forward_hook(hook);values=torch.zeros(5,48,2,dtype=torch.float64);count=0
    try:
        for i,row in enumerate(rows):
            ids=torch.tensor([row['ids']],device='cuda')
            for arm in range(5):
                context.update(arm=arm,masks={k:v[None].to('cuda') for k,v in masks[i].items()})
                x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
                for block in model.transformer.h:x,v1=block(x,v1,x0)
                scores=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
                values[arm,i,0]=(scores[row['uk_id']]-scores[row['us_id']]).cpu();values[arm,i,1]=(scores[row['control_ids'][0]]-scores[row['control_ids'][1]]).cpu();count+=1
    finally:handle.remove()
    assert count==240 and bool(torch.isfinite(values).all());effect=values-values[0];cells=[]
    for family in range(2):
        ix=[i for i,r in enumerate(rows) if r['family']==family];native=values[0,ix[::2],0]-values[0,ix[1::2],0]
        cue=effect[:,ix[::2],0]-effect[:,ix[1::2],0];control=effect[:,ix,1]
        framing_error=rel(cue[1],cue[3]);clause_error=rel(cue[2],cue[3]);materiality=float(cue[3].norm()/cue[4].norm().clamp_min(1e-8))
        target_rms=float(cue[1].square().mean().sqrt());control_rms=float(control[1].square().mean().sqrt());ratio=control_rms/max(target_rms,1e-8)
        cells.append({'family':family,'native_positive_pairs':int((native>0).sum()),'native_cue_mean':float(native.mean()),'all_odd_to_fullhead_cue':materiality,'framing_cue_error':framing_error,'clause_cue_error':clause_error,'framing_target_rms':target_rms,'framing_control_rms':control_rms,'framing_control_to_target_rms':ratio,'capability_pass':int((native>0).sum())>=10 and materiality>=.05,'transfer_pass':framing_error<=.35 and clause_error>=.5,'selectivity_pass':ratio<=.5})
    torch.save({'values':values,'effects':effect},artifact)
    result={'pred_a':max(checks)<=1e-10 and count==240 and bool(torch.isfinite(values).all()),'pred_b':all(c['capability_pass'] for c in cells),'pred_c':all(c['transfer_pass'] for c in cells),'pred_d':all(c['selectivity_pass'] for c in cells),'max_source_recomposition_error':max(checks),'families':cells,'body_forwards':count,'seconds':time.perf_counter()-start,'artifact_sha256':digest(artifact),'source_shas':binding,'arm_order':ARM_ORDER,'scope':'Fresh authored template/city screen of O framing relay. Native model states and suffix retained. Controlled-task held-out evidence only; no corpus OOD, independent generator, unique unit, static compression or quantization claim.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True);signal.alarm(0)


if __name__=='__main__':main()
