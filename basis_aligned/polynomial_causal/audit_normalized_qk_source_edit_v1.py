"""CPU native audit of frozen QK source removals with full head normalization."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,time
from pathlib import Path
import torch
from audit_joint_qk_value_ports_v1 import CK,digest
from normalized_qk_source_edit_v1 import response,normalizer_ports
from folded_normalized_router_v1 import direct,EPS

P=Path(__file__).resolve().parent


def control():
    torch.manual_seed(1237)
    weights=[torch.randn(4,7) for _ in range(4)]
    q,x=torch.randn(11,7),torch.randn(11,7);basis=torch.linalg.qr(torch.randn(7,2)).Q
    y=x-(x@basis)@basis.T
    r=response(weights,q,x,y,8,7)
    f,g=direct(weights,q,x,8,7),direct(weights,q,y,8,7)
    errors=[float((r['full']-f).norm()/f.norm()),float((r['numerator']+r['normalizer']-(f-g)).norm()/(f-g).norm())]
    projector=basis@basis.T;outside=torch.eye(7)-projector
    for key in [weights[1],weights[3]]:
        gram=key.T@key/4
        raw={'inside':projector@gram@projector,'outside':outside@gram@outside,
             'mixed':projector@gram@outside+outside@gram@projector}
        observed=normalizer_ports(key,basis)
        errors.extend(abs(observed[k]-float(v.square().sum()/gram.square().sum())) for k,v in raw.items())
    return max(errors)


def main():
    start=time.perf_counter();torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    fixture=control();assert fixture<1e-10
    spectral=json.loads((P/'JOINT_QK_SOURCE_BOUND_V1_RESULT.json').read_text());cache=spectral['cache']
    assert digest(cache['path'])==cache['sha256'];spaces=torch.load(cache['path'],weights_only=True,map_location='cpu')
    prior=json.loads((P/'JOINT_SHARED_READER_RANK16_V1_RESULT.json').read_text());mc=prior['cache']
    assert digest(mc['path'])==mc['sha256']
    a=torch.load(mc['path'],weights_only=True,map_location='cpu')['programs'][prior['best']]['reader']
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    maps=[sd[f'transformer.h.17.attn.{key}.weight'].double().reshape(9,128,1152) for key in ['c_q','c_k','c_q2','c_k2']]
    o=sd['transformer.h.17.attn.c_proj.weight'].double();vc=sd['transformer.h.17.attn.c_v.weight'].double()
    vb=sd['transformer.h.0.attn.c_v.weight'].double();mix=float(sd['transformer.h.17.attn.lamb'])
    torch.manual_seed(1249)
    normalize=lambda x:torch.nn.functional.rms_norm(x,(1152,),eps=EPS)
    q,x,base=[normalize(torch.randn(64,1152)) for _ in range(3)]
    rows=[];errors=[fixture]
    for h in range(9):
        sl=slice(h*128,(h+1)*128);weights=[w[h] for w in maps]
        g=(1-mix)*(a@o[:,sl])@vc[sl];gb=mix*(a@o[:,sl])@vb[sl]
        for position in [7,0]:
            basis=spaces[f'{h}:{position}'];edited=x-(x@basis)@basis.T
            r=response(weights,q,x,edited,8,position)
            native=direct(weights,q,x,8,position);other=direct(weights,q,edited,8,position)
            effect=native-other
            replay=float((r['numerator']+r['normalizer']-effect).norm()/effect.norm())
            value=x@g+base@gb;edited_value=edited@g+base@gb
            gate_effect=native*value-other*edited_value
            pieces=[r['numerator']*edited_value,r['normalizer']*edited_value,native*(value-edited_value)]
            gate_replay=float((sum(pieces)-gate_effect).norm()/gate_effect.norm())
            errors.extend([replay,gate_replay,float((r['full']-native).norm()/native.norm())])
            rows.append(dict(head=h,source_position=position,
                             normalizer_coefficient_ports=[normalizer_ports(weights[i],basis) for i in [1,3]],
                             numerator_only_routing_effect_relative_error=float((r['numerator']-effect).norm()/effect.norm()),
                             normalized_routing_effect_to_original_norm=float(effect.norm()/native.norm()),
                             denominator_edited_to_original_median=float((r['edited_denominator']/r['full_denominator']).median()),
                             gate_current_value_reader_projected_energy_fraction=float((g@basis).square().sum()/g.square().sum()),
                             gate_effect_piece_norm_ratios=[float(v.norm()/gate_effect.norm()) for v in pieces],
                             routing_effect_replay_relative_error=replay,gate_effect_replay_relative_error=gate_replay))
    valid=max(errors)<1e-10
    output=dict(predictions={'pred_a_instrument':valid,'pred_b_numerator_only_sufficient':valid and all(x['numerator_only_routing_effect_relative_error']<=.1 for x in rows)},
                rows=rows,fixture_error=fixture,maximum_instrument_error=max(errors),wall_seconds=time.perf_counter()-start,
                source_cache_sha256=cache['sha256'],component_cache_sha256=mc['sha256'],body_forwards=0,corpus_access=False,
                formal_probes=64,seed=1249,
                scope='Exact post-input-normalization current-source edit, head key RMS recomputed, query/base value fixed. Gaussian formal probes are numerical stress diagnostics, not FineWeb/OOD behavioral evidence. Scalar OV gate response is local, not the full downstream MLP or model.')
    with (P/'NORMALIZED_QK_SOURCE_EDIT_V1_RESULT.json').open('x') as f:json.dump(output,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in output.items() if k!='rows'},indent=2));assert valid


if __name__=='__main__':main()
