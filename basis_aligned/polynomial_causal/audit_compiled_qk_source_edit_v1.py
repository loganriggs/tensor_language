"""Native replay of compiled baseline-only source edits; CPU, no text data."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,time
from pathlib import Path
import torch
from audit_joint_qk_value_ports_v1 import CK,digest
from compiled_qk_source_edit_v1 import compile_maps,prepare_ports,predict,compose
from folded_normalized_router_v1 import direct,EPS

P=Path(__file__).resolve().parent


def main():
    start=time.perf_counter();torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    assert json.loads((P/'COMPILED_QK_SOURCE_EDIT_V1_CONTROL.json').read_text())['instrument_passed']
    spectral=json.loads((P/'JOINT_QK_SOURCE_BOUND_V1_RESULT.json').read_text());cache=spectral['cache']
    assert digest(cache['path'])==cache['sha256'];spaces=torch.load(cache['path'],weights_only=True,map_location='cpu')
    prior=json.loads((P/'JOINT_SHARED_READER_RANK16_V1_RESULT.json').read_text());mc=prior['cache']
    assert digest(mc['path'])==mc['sha256']
    a=torch.load(mc['path'],weights_only=True,map_location='cpu')['programs'][prior['best']]['reader']
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    maps=[sd[f'transformer.h.17.attn.{key}.weight'].double().reshape(9,128,1152) for key in ['c_q','c_k','c_q2','c_k2']]
    o=sd['transformer.h.17.attn.c_proj.weight'].double();vc=sd['transformer.h.17.attn.c_v.weight'].double()
    vb=sd['transformer.h.0.attn.c_v.weight'].double();mix=float(sd['transformer.h.17.attn.lamb'])
    torch.manual_seed(1277);normalize=lambda x:torch.nn.functional.rms_norm(x,(1152,),eps=EPS)
    query,source,base=[normalize(torch.randn(64,1152)) for _ in range(3)]
    rotation=torch.linalg.qr(torch.randn(17,17)).Q
    first=torch.diag(torch.tensor([1.]*8+[0.]*9));second=.7*rotation@first@rotation.T
    edits={'zero':torch.zeros(17,17),'full':torch.eye(17),'first8':first,
           'amplify':-.25*torch.eye(17),'composed':compose(first,second),'reversed':compose(second,first)}
    rows=[];errors=[];composition_errors=[]
    for h in range(9):
        sl=slice(128*h,128*(h+1));weights=[w[h] for w in maps]
        gate=(1-mix)*(a@o[:,sl])@vc[sl];base_gate=mix*(a@o[:,sl])@vb[sl]
        full_value=source@gate+base@base_gate
        for position in [7,0]:
            basis=spaces[f'{h}:{position}'];compiled=compile_maps(weights,basis,gate,8,position)
            ports=prepare_ports(weights,compiled,query,source,full_value)
            predictions={};local=[]
            for name,edit in edits.items():
                pred=predict(ports,compiled['norm_cores'],compiled['gate_core'],edit)
                edited=source-(source@basis@edit.T)@basis.T
                expected_route=direct(weights,query,edited,8,position)
                expected=expected_route*(edited@gate+base@base_gate)
                error=float((pred['contribution']-expected).norm()/expected.norm())
                route_error=float((pred['routing']-expected_route).norm()/expected_route.norm())
                errors.extend([error,route_error]);predictions[name]=pred['contribution']
                local.append(dict(edit=name,contribution_relative_error=error,routing_relative_error=route_error))
            sequential=source-(source@basis@first.T)@basis.T
            sequential=sequential-(sequential@basis@second.T)@basis.T
            composed_source=source-(source@basis@edits['composed'].T)@basis.T
            sequential_error=float((sequential-composed_source).norm()/sequential.norm());composition_errors.append(sequential_error)
            changed=compile_maps(weights,basis@rotation,gate,8,position)
            changed_ports=prepare_ports(weights,changed,query,source,full_value)
            rotated=predict(changed_ports,changed['norm_cores'],changed['gate_core'],rotation.T@edits['composed']@rotation)
            gauge=float((rotated['contribution']-predictions['composed']).norm()/predictions['composed'].norm());errors.append(gauge)
            rows.append(dict(head=h,source_position=position,edits=local,gauge_relative_error=gauge,
                             sequential_source_relative_error=sequential_error,
                             order_effect_relative_to_composed=float((predictions['composed']-predictions['reversed']).norm()/predictions['composed'].norm())))
    output=dict(predictions={'pred_a_compiled_replay':max(errors)<1e-10,'pred_b_composition':max(composition_errors)<1e-10},
                rows=rows,maximum_replay_error=max(errors),maximum_composition_error=max(composition_errors),
                wall_seconds=time.perf_counter()-start,body_forwards=0,corpus_access=False,formal_probes=64,seed=1277,
                price=dict(dynamic_port_numbers=92,small_core_numbers=595,projection_map_numbers=97920,
                           native_score_norm_value_ports_required=True,per_head_and_position=True),
                source_sha256=digest(__file__),predictor_sha256=digest(P/'compiled_qk_source_edit_v1.py'),
                scope='Conditional exact normalized routing/scalar OV response from baseline-only ports. Six finite edits and gauge/order tests. Native background still required; no semantic selectivity, natural-data prediction or full-model adoption.')
    with (P/'COMPILED_QK_SOURCE_EDIT_V1_RESULT.json').open('x') as f:json.dump(output,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in output.items() if k!='rows'},indent=2));assert all(output['predictions'].values())


if __name__=='__main__':main()
