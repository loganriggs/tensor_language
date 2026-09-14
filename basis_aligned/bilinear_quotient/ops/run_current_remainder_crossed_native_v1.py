#!/usr/bin/env python3
# BQGATE:288bodyforwards;48prefixes;180seconds;no fitting.
"""pred_a native/joint anchors<=1e-5; component/field identities<=1e-10.
pred_b value-only cue error<=.20 and joint/fullhead cue>=.01 EACHfamily.
pred_c three-effect sum error<=.05 EACHfamily/readout.
pred_d routing+value effect sum error<=.05 EACHfamily/readout.
Null: joint routing/value computation or mixed effect is indispensable.
Price288bodyforwards,48prefixes,180seconds; native weights/context plus FP64 ports.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from current_remainder_crossed_v1 import CurrentRemainder
from sparse_path_stability_atlas_v1 import digest
from inherited_source_positions_v1 import paired_masks
from regional_cue_row_check_v1 import validate
STEM='CURRENT_REMAINDER_CROSSED_NATIVE_V1'


@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
    assert all(digest(k)==v for k,v in binding.items())
    rows=json.loads((P/'SRO_ARTICLE_CORRECTION_V1_ROWS.json').read_text())['rows']
    assert len(rows)==48;validate(rows);paired_masks(rows)
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        check=json.loads((P/'CURRENT_REMAINDER_CROSSED_V1_CPU_CONTROL.json').read_text())
        assert max(check['errors'].values())<=1e-10 and check['mixed_to_full_change']>=1e-4
        print('288bodyforwards;48prefixes; native weights CPU mixed identity passed');return
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_ARTIFACT.pt')
    assert not out.exists() and not artifact.exists()
    start=time.perf_counter();signal.alarm(180);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval()
    p={k:v.cuda() for k,v in torch.load(P/'EVEN_VALUE_SHARED_GRAPH_PACKED_V1_PROGRAM.pt',weights_only=True).items()}
    graph=CurrentRemainder(p);context={};ports=[];checks=[];identities=[];count=0
    rel=lambda x,y:float((x-y).norm()/y.norm().clamp_min(1e-8))
    def hook(module,args,output):
        if context['capture']:
            current=args[0];g,z=graph.routing(current),graph.values(current)
            ref=graph.reference.state(current,first_values=torch.zeros(*current.shape[:2],128,device='cuda'))[1]
            checks.append(rel(g@z,ref));ports.append((g,z))
            return output
        return output[0]+context['delta'].to(output[0].dtype),output[1]
    handle=model.transformer.h[9].attn.register_forward_hook(hook)
    values=torch.zeros(6,48,2,dtype=torch.float64)
    def forward(row):
        nonlocal count
        ids=torch.tensor([row['ids']],device='cuda')
        x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
        for block in model.transformer.h:x,v1=block(x,v1,x0)
        scores=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
        count+=1
        return torch.stack([scores[row['uk_id']]-scores[row['us_id']],scores[row['control_ids'][0]]-scores[row['control_ids'][1]]]).double().cpu()
    try:
        context['capture']=True
        for i,row in enumerate(rows):values[0,i]=forward(row)
        context['capture']=False
        for i,row in enumerate(rows):
            changes=graph.changes(*ports[i],*ports[i^1])
            composed=changes['routing']+changes['values']+changes['mixed']
            identities.append(rel(composed,changes['joint']))
            for arm,channels in enumerate([changes['routing'],changes['values'],changes['joint'],changes['mixed'],composed],1):
                context['delta']=graph.write(channels);values[arm,i]=forward(row)
    finally:handle.remove()
    assert count==288
    old=torch.load(P/'SRO_ARTICLE_CORRECTION_V1_ARTIFACT.pt',weights_only=True)['cube']
    anchors=dict(native=rel(values[0],old[0]),joint_recomposed=rel(values[5],values[3]))
    effect=values-values[0];head=old[7]-old[0];cells=[]
    for family in range(2):
        ix=[i for i,r in enumerate(rows) if r['family']==family]
        cue=effect[:,ix[::2],0]-effect[:,ix[1::2],0];control=effect[:,ix,1]
        materiality=float(cue[3].norm()/(head[ix[::2],0]-head[ix[1::2],0]).norm().clamp_min(1e-8))
        value_error=rel(cue[2],cue[3])
        three=[rel(x[1]+x[2]+x[4],x[3]) for x in [cue,control]]
        two=[rel(x[1]+x[2],x[3]) for x in [cue,control]]
        cells.append(dict(family=family,value_only_cue_error=value_error,joint_to_fullhead_cue=materiality,
                          three_effect_errors=three,two_effect_errors=two,
                          cue_effect_norms=[float(x.norm()) for x in cue[1:]],
                          control_effect_norms=[float(x.norm()) for x in control[1:]]))
    port_bytes=sum(t.numel()*t.element_size() for pair in ports for t in pair)
    torch.save(dict(values=values,effects=effect,ports=[tuple(t.cpu() for t in pair) for pair in ports]),artifact)
    result={'pred_a':max(anchors.values())<=1e-5 and max(checks+identities)<=1e-10,
            'pred_b':all(c['value_only_cue_error']<=.2 and c['joint_to_fullhead_cue']>=.01 for c in cells),
            'pred_c':all(max(c['three_effect_errors'])<=.05 for c in cells),
            'pred_d':all(max(c['two_effect_errors'])<=.05 for c in cells),
            'anchors':anchors,'max_component_error':max(checks),'max_field_identity_error':max(identities),
            'families':cells,'port_bytes':port_bytes,'body_forwards':count,'seconds':time.perf_counter()-start,
            'artifact_sha256':digest(artifact),'source_shas':binding,
            'arm_order':['native','routing','values','joint','mixed','recomposed'],
            'scope':'Paired interchange in full current even remainder on corrected rows; original native weights/prefix/suffix and measured ports retained. No refit, semantic role or static model compression.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True)


if __name__=='__main__':main()
