"""Exact bilinear finite response; normalization is recomputed before this call."""
import torch
import json
from pathlib import Path


def response(native_normalized,edited_normalized,left,right,down):
    u=native_normalized.double();d=edited_normalized.double()-u
    l=left.double();r=right.double();w=down.double()
    lu=u@l.T;ru=u@r.T;ld=d@l.T;rd=d@r.T
    cross=(lu*rd+ld*ru)@w.T
    quadratic=(ld*rd)@w.T
    exact=(((lu+ld)*(ru+rd))-lu*ru)@w.T
    return cross,quadratic,exact


def controls():
    rng=torch.Generator().manual_seed(910813);records=[]
    for width in (8,32,128):
        x=torch.randn(32,width,generator=rng,dtype=torch.float64)
        # A finite source write plus native residual scale changes a raw consumer input.
        source=torch.randn(x.shape,generator=rng,dtype=torch.float64)*.4
        left=torch.randn(2*width,width,generator=rng,dtype=torch.float64)/width**.5
        right=torch.randn(2*width,width,generator=rng,dtype=torch.float64)/width**.5
        down=torch.randn(width,2*width,generator=rng,dtype=torch.float64)/(2*width)**.5
        u=torch.nn.functional.rms_norm(x,(width,),eps=1.1920928955078125e-7)
        v=torch.nn.functional.rms_norm(x-1.3*source,(width,),eps=1.1920928955078125e-7)
        c,q,e=response(u,v,left,right,down)
        direct=((v@left.T)*(v@right.T)-(u@left.T)*(u@right.T))@down.T
        error=float((c+q-direct).abs().max());assert error<1e-12
        omitted=float((c-direct).norm()/direct.norm());assert omitted>.05
        records.append({'width':width,'closure_max_abs':error,'quadratic_omission_relative_error':omitted})
    return {'passed':True,'records':records,'scope':'Exact normalized finite-response algebra; cross term is linear in normalized delta, not in raw source perturbation. No native consumer dominance claimed.'}

if __name__=='__main__':
    torch.set_num_threads(2);r=controls();f=Path(__file__).with_name('BILINEAR_NORMALIZED_SOURCE_RESPONSE_V1_CONTROLS.json')
    with f.open('x') as out:json.dump(r,out,indent=2);out.write('\n')
    print(json.dumps(r))
