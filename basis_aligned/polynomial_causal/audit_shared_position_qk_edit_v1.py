"""Native fixed-adapter replay and reproducible head-resolved common-space audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,time
from pathlib import Path
import torch
from audit_joint_qk_value_ports_v1 import CK,digest
from shared_position_qk_edit_v1 import compile_static,prepare_from_head_vectors
from compiled_qk_source_edit_v1 import predict
from folded_normalized_router_v1 import direct,EPS

P=Path(__file__).resolve().parent


def redteam(result):
    heads=result['heads'];rows=[]
    for h in heads:
        rows.append(dict(head=h['head'],retention_missed=h['common_to_separate_ratio']<.9,validation_mean_missed=h['validation_mean_touch']<.45,validation_minimum_missed=h['validation_minimum_touch']<.3,discovery_upper_bound_below_45_percent=h['discovery_mean_touch_upper_bound']<.45,discovery_upper_bound=h['discovery_mean_touch_upper_bound'],discovery_touch=h['discovery_mean_touch'],validation_touch=h['validation_mean_touch'],train_validation_touch_gap=h['validation_mean_touch']-h['discovery_mean_touch'],common_to_separate_ratio=h['common_to_separate_ratio']))
    return dict(original_predictions=result['predictions'],head_audit=rows,validation_mean_touch=sum(h['validation_mean_touch'] for h in heads)/9,validation_mean_inside=sum(h['validation_inside_mean'] for h in heads)/9,validation_mean_mixed=sum(h['validation_mixed_mean'] for h in heads)/9,mean_retention=sum(h['common_to_separate_ratio'] for h in heads)/9,maximum_split_touch_gap=max(abs(x['train_validation_touch_gap']) for x in rows),scope='Executed head-resolved and same-rank influence-bound red-team. Discovery bounds do not certify validation or32sample targets. Influence selection is not true-touch optimization. Original B/C failures preserved; no broad negative about shared source structure.')


def main():
    start=time.perf_counter();torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    assert json.loads((P/'SHARED_POSITION_QK_EDIT_V1_CONTROL.json').read_text())['instrument_passed']
    result=json.loads((P/'POSITION_SHARED_QK_SOURCE_V1_RESULT.json').read_text());cache=result['cache']
    assert redteam(result)==json.loads((P/'POSITION_SHARED_QK_SOURCE_V1_REDTEAM.json').read_text())
    assert digest(cache['path'])==cache['sha256'];frames=torch.load(cache['path'],weights_only=True,map_location='cpu')['frames']
    prior=json.loads((P/'JOINT_SHARED_READER_RANK16_V1_RESULT.json').read_text());mc=prior['cache']
    assert digest(mc['path'])==mc['sha256'];a=torch.load(mc['path'],weights_only=True,map_location='cpu')['programs'][prior['best']]['reader']
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    maps=[sd[f'transformer.h.17.attn.{key}.weight'].double().reshape(9,128,1152) for key in ['c_q','c_k','c_q2','c_k2']]
    o=sd['transformer.h.17.attn.c_proj.weight'].double();vc=sd['transformer.h.17.attn.c_v.weight'].double()
    vb=sd['transformer.h.0.attn.c_v.weight'].double();mix=float(sd['transformer.h.17.attn.lamb'])
    torch.manual_seed(1303);normalize=lambda x:torch.nn.functional.rms_norm(x,(1152,),eps=EPS)
    query,source,base=[normalize(torch.randn(64,1152)) for _ in range(3)]
    edits={'full':torch.eye(17),'mixed':.2*torch.randn(17,17)};rows=[];errors=[];prices=[]
    for h in range(9):
        sl=slice(128*h,128*(h+1));weights=[w[h] for w in maps]
        gate=(1-mix)*(a@o[:,sl])@vc[sl];gb=mix*(a@o[:,sl])@vb[sl]
        compiled=compile_static(weights,frames[h],gate)
        prices.append(compiled['basis'].numel()+compiled['gate_core'].numel()+sum(x.numel() for x in compiled['key_inside']+compiled['norm_cores']))
        qheads=[query@weights[j].T for j in [0,2]];kheads=[source@weights[j].T for j in [1,3]]
        for t,s in [(511,0),(511,1),(511,255),(511,510),(255,13)]:
            ports=prepare_from_head_vectors(compiled,source,qheads,kheads,source@gate+base@gb,t,s)
            for name,edit in edits.items():
                observed=predict(ports,compiled['norm_cores'],compiled['gate_core'],edit)
                changed=source-(source@frames[h]@edit.T)@frames[h].T
                expected=direct(weights,query,changed,t,s)*(changed@gate+base@gb)
                error=float((observed['contribution']-expected).norm()/expected.norm());errors.append(error)
                rows.append(dict(head=h,query_position=t,source_position=s,edit=name,relative_error=error))
    assert all(n==24531 for n in prices)
    output=dict(instrument_passed=max(errors)<1e-10,rows=rows,maximum_replay_error=max(errors),
                wall_seconds=time.perf_counter()-start,source_cache_sha256=cache['sha256'],body_forwards=0,corpus_access=False,
                price=dict(adapter_numbers_per_head=24531,all_nine_heads=9*24531,previous_adapter_numbers_per_head_and_position=98515,
                           dynamic_predictor_port_numbers=92,native_head_vectors_and_full_scores_norms_values_required=True),
                scope='One fixed source-edit adapter per head reused across five position pairs, including a different query position. Exact conditional local scalar response, not natural-data validation or a whole-model saving.')
    with (P/'SHARED_POSITION_QK_EDIT_V1_RESULT.json').open('x') as f:json.dump(output,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in output.items() if k!='rows'},indent=2));assert output['instrument_passed']


if __name__=='__main__':main()
