"""Full head-value vector extension of the baseline-port source edit predictor."""
import json
from pathlib import Path
import torch
from compiled_qk_source_edit_v1 import predict
from shared_position_qk_edit_v1 import compile_static,prepare_from_head_vectors
from folded_normalized_router_v1 import direct


def compile_vector(weights,basis,value_weight,value_mix):
    compiled=compile_static(weights,basis,torch.zeros(basis.shape[0],dtype=basis.dtype,device=basis.device))
    compiled.pop('gate_core')
    compiled['value_core']=(1-value_mix)*(value_weight@basis).T
    return compiled


def predict_vector(ports,compiled,edit):
    scalar_ports=dict(ports,full_value=torch.zeros_like(ports['scores'][0]))
    rank=edit.shape[0]
    routing=predict(scalar_ports,compiled['norm_cores'],torch.zeros(rank,dtype=edit.dtype,device=edit.device),edit)['routing']
    delta=ports['source_coordinates']@edit.T
    value=ports['full_value']-delta@compiled['value_core']
    return dict(routing=routing,value=value,contribution=routing[:,None]*value)


def control():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(1319)
    weights=[torch.randn(4,7) for _ in range(4)];basis=torch.linalg.qr(torch.randn(7,3)).Q
    value_weight=torch.randn(4,7);q,s=torch.randn(11,7),torch.randn(11,7);base=torch.randn(11,4);mix=-.1
    compiled=compile_vector(weights,basis,value_weight,mix)
    qh=[q@weights[j].T for j in [0,2]];kh=[s@weights[j].T for j in [1,3]]
    value=(1-mix)*(s@value_weight.T)+mix*base
    ports=prepare_from_head_vectors(compiled,s,qh,kh,value,127,32)
    errors=[]
    for edit in [torch.zeros(3,3),torch.eye(3),.2*torch.randn(3,3)]:
        observed=predict_vector(ports,compiled,edit)
        edited=s-(s@basis@edit.T)@basis.T
        expected=direct(weights,q,edited,127,32)[:,None]*((1-mix)*(edited@value_weight.T)+mix*base)
        errors.append(float((observed['contribution']-expected).norm()/expected.norm()))
    out=dict(instrument_passed=max(errors)<1e-10,maximum_vector_replay_error=max(errors),
             scope='Full vector value contribution from baseline-only ports, including signed base/current mixing. CPU algebra control, no text validation yet.')
    Path(__file__).with_name('COMPILED_QK_VECTOR_EDIT_V1_CONTROL.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2));assert out['instrument_passed']


if __name__=='__main__':control()
