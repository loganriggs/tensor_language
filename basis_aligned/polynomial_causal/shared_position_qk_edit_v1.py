"""Position-independent source edit adapter, reusing native query/key vectors."""
import json
from pathlib import Path
import torch
from folded_normalized_router_v1 import rotary,EPS,direct
from compiled_qk_source_edit_v1 import predict


def compile_static(weights,basis,gate):
    width=weights[0].shape[0];inside=[weights[j]@basis for j in [1,3]]
    return dict(basis=basis,key_inside=inside,norm_cores=[k.T@k/width for k in inside],gate_core=gate@basis)


def prepare_from_head_vectors(compiled,source,query_heads,key_heads,full_value,query_position,source_position):
    width=query_heads[0].shape[-1]
    rt=rotary(query_position,width).to(source);rs=rotary(source_position,width).to(source)
    query_coordinates=[];norm_coordinates=[];scores=[];query_norms=[];key_norms=[]
    for q,k,inside in zip(query_heads,key_heads,compiled['key_inside']):
        # Native RMS moments are before the BF16-rounded rotary tables, whose
        # finite-precision matrices are not exactly orthogonal.
        qr=q@rt.T;kr=k@rs.T
        scores.append((qr*kr).sum(-1)/width)
        query_coordinates.append(qr@(rs@inside)/width)
        norm_coordinates.append(k@inside/width)
        query_norms.append(q.square().mean(-1)+EPS)
        key_norms.append(k.square().mean(-1)+EPS)
    return dict(source_coordinates=source@compiled['basis'],query_coordinates=query_coordinates,
                norm_coordinates=norm_coordinates,scores=scores,query_norms=query_norms,key_norms=key_norms,full_value=full_value)


def control():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(1301)
    weights=[torch.randn(4,7) for _ in range(4)];basis=torch.linalg.qr(torch.randn(7,3)).Q;gate=torch.randn(7)
    q,x=torch.randn(11,7),torch.randn(11,7);base=torch.randn(11)
    compiled=compile_static(weights,basis,gate);edit=torch.randn(3,3)*.2
    query_heads=[q@weights[j].T for j in [0,2]];key_heads=[x@weights[j].T for j in [1,3]]
    errors=[]
    for t,s in [(8,7),(255,13),(511,0)]:
        ports=prepare_from_head_vectors(compiled,x,query_heads,key_heads,x@gate+base,t,s)
        observed=predict(ports,compiled['norm_cores'],compiled['gate_core'],edit)
        edited=x-(x@basis@edit.T)@basis.T
        expected=direct(weights,q,edited,t,s)*(edited@gate+base)
        errors.append(float((observed['contribution']-expected).norm()/expected.norm()))
    out=dict(instrument_passed=max(errors)<1e-10,maximum_position_replay_error=max(errors),
             scope='One fixed adapter reused across three position pairs; native pre-RoPE query/key vectors remain inputs to port preparation. No native shared-frame result yet.')
    Path(__file__).with_name('SHARED_POSITION_QK_EDIT_V1_CONTROL.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2));assert out['instrument_passed']


if __name__=='__main__':control()
