"""Exact local source edit program using unedited routing/value ports only."""
import json
from pathlib import Path
import torch
from folded_normalized_router_v1 import rotary,EPS,direct


def compile_maps(weights,basis,gate,query_position,source_position):
    width=weights[0].shape[0];rotation=rotary(query_position,width).to(basis).T@rotary(source_position,width).to(basis)
    query_maps=[];norm_maps=[];cores=[]
    for q,k in [weights[:2],weights[2:]]:
        ke=k@basis
        query_maps.append(q.T@rotation@ke/width)
        norm_maps.append(k.T@ke/width)
        cores.append(ke.T@ke/width)
    return dict(basis=basis,query_maps=query_maps,norm_maps=norm_maps,
                norm_cores=cores,gate_core=gate@basis,rotation=rotation)


def prepare_ports(weights,maps,query,source,full_value):
    width=weights[0].shape[0];scores=[];query_norms=[];key_norms=[]
    for q,k in [weights[:2],weights[2:]]:
        qx=query@q.T;ks=source@k.T
        scores.append(((qx@maps['rotation'])*ks).sum(-1)/width)
        query_norms.append(qx.square().mean(-1)+EPS)
        key_norms.append(ks.square().mean(-1)+EPS)
    return dict(source_coordinates=source@maps['basis'],
                query_coordinates=[query@m for m in maps['query_maps']],
                norm_coordinates=[source@m for m in maps['norm_maps']],
                scores=scores,query_norms=query_norms,key_norms=key_norms,full_value=full_value)


def predict(ports,norm_cores,gate_core,edit):
    """Needs no raw states, native weights, edited endpoint or large compiled maps."""
    delta=ports['source_coordinates']@edit.T
    scores=[];key_norms=[]
    for j in range(2):
        scores.append(ports['scores'][j]-(ports['query_coordinates'][j]*delta).sum(-1))
        key_norms.append(ports['key_norms'][j]-2*(ports['norm_coordinates'][j]*delta).sum(-1)+((delta@norm_cores[j])*delta).sum(-1))
    denominator_squared=ports['query_norms'][0]*ports['query_norms'][1]*key_norms[0]*key_norms[1]
    if not bool((denominator_squared>0).all()):raise ValueError('Nonpositive compiled norm: invalid numerical evaluation')
    routing=scores[0]*scores[1]/denominator_squared.sqrt()
    value=ports['full_value']-delta@gate_core
    return dict(routing=routing,value=value,contribution=routing*value)


def compose(first,second):
    """Apply first then second, in the same orthonormal source coordinates."""
    return first+second-second@first


def control():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(1259)
    weights=[torch.randn(4,7) for _ in range(4)];basis=torch.linalg.qr(torch.randn(7,3)).Q
    query,source=torch.randn(13,7),torch.randn(13,7);gate=torch.randn(7);base=torch.randn(13)
    maps=compile_maps(weights,basis,gate,8,7);ports=prepare_ports(weights,maps,query,source,source@gate+base)
    first=torch.randn(3,3)*.2;second=torch.randn(3,3)*.2
    combined=compose(first,second)
    original_edited=source-(source@basis@first.T)@basis.T
    original_edited=original_edited-(original_edited@basis@second.T)@basis.T
    predicted=predict(ports,maps['norm_cores'],maps['gate_core'],combined)
    expected=direct(weights,query,original_edited,8,7)*(original_edited@gate+base)
    error=float((predicted['contribution']-expected).norm()/expected.norm())
    rotation=torch.linalg.qr(torch.randn(3,3)).Q
    changed=compile_maps(weights,basis@rotation,gate,8,7)
    changed_ports=prepare_ports(weights,changed,query,source,source@gate+base)
    rotated=predict(changed_ports,changed['norm_cores'],changed['gate_core'],rotation.T@combined@rotation)
    gauge=float((rotated['contribution']-predicted['contribution']).norm()/expected.norm())
    reverse=predict(ports,maps['norm_cores'],maps['gate_core'],compose(second,first))
    order_difference=float((reverse['contribution']-predicted['contribution']).norm()/expected.norm())
    out=dict(instrument_passed=max(error,gauge)<1e-10 and order_difference>.001,
             sequential_edit_relative_error=error,gauge_change_relative_error=gauge,
             noncommuting_order_effect=order_difference,
             scope='Conditional exact local response from baseline-only ports, including key RMS and scalar value. No edited endpoint supplied to predictor; no semantic circuit or whole-model claim.')
    Path(__file__).with_name('COMPILED_QK_SOURCE_EDIT_V1_CONTROL.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2));assert out['instrument_passed']


if __name__=='__main__':control()
