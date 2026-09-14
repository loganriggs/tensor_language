#!/usr/bin/env python3
# BQGATE:192bodyforwards;48prefixes;180seconds;no fitting.
"""pred_a native anchor<=1e-5; live source recomposition<=1e-10.
pred_b city-only cue error<=.20 AND inherited/fullhead cue>=.01 EACHfamily.
pred_c separate-source effect composition<=.05 EACHfamily/readout.
Null: inherited source is material or native suffix prevents additive response.
Price192bodyforwards,48prefixes,180seconds; original model weights retained.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from run_even_value_factorial_native_v1 import setup,cpu_control
from remainder_value_sources_v1 import split
from inherited_source_positions_v1 import paired_masks,inherited_at
from sparse_path_stability_atlas_v1 import digest
from regional_cue_row_check_v1 import validate
STEM='INHERITED_SOURCE_POSITIONS_NATIVE_V1'


@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
    assert all(digest(k)==v for k,v in binding.items())
    rows=json.loads((P/'SRO_ARTICLE_CORRECTION_V1_ROWS.json').read_text())['rows']
    assert len(rows)==48
    validate(rows);masks=paired_masks(rows)
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        assert max(json.loads((P/'REMAINDER_VALUE_SOURCES_V1_CPU_CONTROL.json').read_text())['errors'].values())<=1e-10
        print('192bodyforwards;48prefixes; source CPUcontrol',cpu_control());return
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
        city=inherited_at(graph,current,first,context['source_mask'])
        other=inherited_at(graph,current,first,~context['source_mask'])
        reference=split(graph,current,first)[1]
        checks.append(rel(city+other,reference))
        delta=graph.write([city,other],[arm in (1,3),arm in (2,3)])
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
    anchors=dict(native=rel(values[0],old[0]))
    effect=values-values[0];cells=[]
    for family in range(2):
        ix=[i for i,r in enumerate(rows) if r['family']==family]
        cue=effect[:,ix[::2],0]-effect[:,ix[1::2],0]
        control=effect[:,ix,1]
        current_error=rel(cue[1],cue[3])
        head=old[7]-old[0]
        headcue=head[ix[::2],0]-head[ix[1::2],0]
        materiality=float(cue[3].norm()/headcue.norm().clamp_min(1e-8))
        composition=[rel(cue[1]+cue[2],cue[3]),rel(control[1]+control[2],control[3])]
        cells.append(dict(family=family,city_only_cue_error=current_error,inherited_to_fullhead_cue=materiality,
                          source_cue_norms=[float(x.norm()) for x in cue[1:]],
                          source_control_norms=[float(x.norm()) for x in control[1:]],
                          composition_errors=composition,
                          current_pass=current_error<=.2 and materiality>=.01,composition_pass=max(composition)<=.05))
    torch.save(dict(values=values,effects=effect),artifact)
    result={'pred_a':max(anchors.values())<=1e-5 and max(checks)<=1e-10,
            'pred_b':all(c['current_pass'] for c in cells),'pred_c':all(c['composition_pass'] for c in cells),
            'anchors':anchors,'max_source_recomposition_error':max(checks),'families':cells,
            'body_forwards':count,'seconds':time.perf_counter()-start,'artifact_sha256':digest(artifact),
            'source_shas':binding,'arm_order':['native','remove_city_inherited','remove_other_inherited','remove_all_inherited'],
            'scope':'Inherited value-source localization on corrected panel; six shared-graph calls per instrumented arm including independent recomposition reference. All native weights/context remain; no new semantic identification or static compression.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True)


if __name__=='__main__':main()
