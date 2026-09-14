#!/usr/bin/env python3
# BQGATE:288bodyforwards;72prefixes;180seconds;no fitting.
"""pred_a native/R anchors<=1e-5; live source recomposition<=1e-10.
pred_b current-only remainder cue error<=.20 EACHfamily.
pred_c separate-source effect composition<=.05 EACHfamily/readout.
Null: inherited source is material or native suffix prevents additive response.
Price288bodyforwards,72prefixes,180seconds; original model weights retained.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from run_even_value_factorial_native_v1 import setup,cpu_control
from remainder_value_sources_v1 import split
from sparse_path_stability_atlas_v1 import digest
from regional_cue_row_check_v1 import validate
STEM='REMAINDER_SOURCES_NATIVE_V1'


@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
    assert all(digest(k)==v for k,v in binding.items())
    rows=json.loads((P/'SRO_FRESH_CUE_V1_ROWS.json').read_text())['rows']
    assert len(rows)==72
    validate(rows[:24]);validate(rows[24:48],expected_token_differences=2);validate(rows[48:])
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        assert max(json.loads((P/'REMAINDER_VALUE_SOURCES_V1_CPU_CONTROL.json').read_text())['errors'].values())<=1e-10
        print('288bodyforwards;72prefixes; source CPUcontrol',cpu_control());return
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
        current=args[0];first=args[1].reshape(*current.shape[:2],9,128)[:,:,8]
        rc,rf=split(graph,current,first)
        reference=graph.state(current,first_values=first)[1]
        checks.append(rel(rc+rf,reference))
        delta=graph.write([rc,rf],[arm in (1,3),arm in (2,3)])
        return output[0]-delta.to(output[0].dtype),output[1]
    handle=model.transformer.h[9].attn.register_forward_hook(hook)
    values=torch.zeros(4,72,2,dtype=torch.float64);count=0
    try:
        for i,row in enumerate(rows):
            ids=torch.tensor([row['ids']],device='cuda')
            for arm in range(4):
                context['arm']=arm
                x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
                for block in model.transformer.h:x,v1=block(x,v1,x0)
                scores=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
                values[arm,i,0]=(scores[row['uk_id']]-scores[row['us_id']]).cpu()
                values[arm,i,1]=(scores[row['control_ids'][0]]-scores[row['control_ids'][1]]).cpu()
                count+=1
    finally:handle.remove()
    assert count==288
    old=torch.load(P/'SRO_FRESH_CUE_V1_ARTIFACT.pt',weights_only=True)['cube']
    anchors=dict(native=rel(values[0],old[0]),remainder=rel(values[3],old[2]))
    effect=values-values[0];cells=[]
    for family in range(3):
        ix=[i for i,r in enumerate(rows) if r['family']==family]
        cue=effect[:,ix[::2],0]-effect[:,ix[1::2],0]
        control=effect[:,ix,1]
        current_error=rel(cue[1],cue[3])
        composition=[rel(cue[1]+cue[2],cue[3]),rel(control[1]+control[2],control[3])]
        cells.append(dict(family=family,current_only_cue_error=current_error,
                          source_cue_norms=[float(x.norm()) for x in cue[1:]],
                          source_control_norms=[float(x.norm()) for x in control[1:]],
                          composition_errors=composition,
                          current_pass=current_error<=.2,composition_pass=max(composition)<=.05))
    torch.save(dict(values=values,effects=effect),artifact)
    result={'pred_a':max(anchors.values())<=1e-5 and max(checks)<=1e-10,
            'pred_b':all(c['current_pass'] for c in cells),'pred_c':all(c['composition_pass'] for c in cells),
            'anchors':anchors,'max_source_recomposition_error':max(checks),'families':cells,
            'body_forwards':count,'seconds':time.perf_counter()-start,'artifact_sha256':digest(artifact),
            'source_shas':binding,'arm_order':['native','remove_current_remainder','remove_inherited_remainder','remove_R'],
            'scope':'Source localization on reused fresh panel; two graph calls plus third reference call per instrumented arm. All native weights/context remain; no new semantic identification or static compression.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True)


if __name__=='__main__':main()
