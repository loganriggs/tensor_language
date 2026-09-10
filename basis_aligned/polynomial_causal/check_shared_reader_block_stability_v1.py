"""Exact shared-reader counterexample to individual-term matching as block stability."""
import json
import hashlib
from pathlib import Path
import torch
from joint_quadratic_fit_v1 import product_cross

def main():
    dtype=torch.float64
    a=torch.tensor([[1.,0.,0.],[1.,0.,0.]],dtype=dtype)
    b=torch.tensor([[0.,1.,0.],[0.,0.,1.]],dtype=dtype)
    w=torch.eye(2,dtype=dtype)
    rotation=torch.tensor([[1.,-1.],[1.,1.]],dtype=dtype)/(2**.5)
    rotated_b=rotation.T@b;rotated_w=w@rotation
    def tensor(aa,bb,ww):
        forms=(aa[:,:,None]*bb[:,None,:]+bb[:,:,None]*aa[:,None,:])/2
        return torch.einsum('vj,jab->vab',ww,forms)
    original=tensor(a,b,w);alternate=tensor(a,rotated_b,rotated_w)
    err=float((alternate-original).norm()/original.norm())
    cross=product_cross(a,b,a,rotated_b);ga=product_cross(a,b,a,b);gb=product_cross(a,rotated_b,a,rotated_b)
    norms_a=((w.T@w).diag()*ga.diag()).sqrt();norms_b=((rotated_w.T@rotated_w).diag()*gb.diag()).sqrt()
    cos=(w.T@rotated_w)*cross/norms_a[:,None]/norms_b[None,:]
    span=float(torch.trace(torch.linalg.solve(ga,cross)@torch.linalg.solve(gb,cross.T))/2)
    assert err<1e-12 and float(cos.max())<.95 and abs(span-1)<1e-12
    result=dict(original_program=['z0=x0*x1','z1=x0*x2'],alternate_program=['f0=x0*(x1+x2)/sqrt(2)','f1=x0*(-x1+x2)/sqrt(2)','z0=(f0-f1)/sqrt(2)','z1=(f0+f1)/sqrt(2)'],
        tensor_relative_error=err,individual_full_term_cosines=cos.tolist(),input_function_span_overlap=span,
        native_forwards=0,scope='Constructive counterexample only. Identical shared-reader block can fail every individual-term .95 match. It does not establish such a block in the checkpoint or rescore the existing fixed screen.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out=Path(__file__).with_name('SHARED_READER_BLOCK_STABILITY_V1_CONTROL.json')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
