"""Exact quadratic coefficient pullback through a linear attention output map.

Input is the pre-c_proj concatenated head output, not token embeddings or values.
Residual context and normalization remain separate in the full block formula.
"""
import torch
from joint_quadratic_fit_v1 import product_cross


def cross_port_energy(la,ra,lb,rb,output_gram):
    aa=la@la.T;bb=ra@ra.T;ab=la@ra.T
    cc=lb@lb.T;dd=rb@rb.T;cd=lb@rb.T
    return ((aa*dd+bb*cc+2*ab*cd.T)*output_gram).sum()/4


def head_energies(l,r,output_gram,heads):
    width=l.shape[1]//heads
    assert width*heads==l.shape[1]
    blocks=[]
    for i in range(heads):
        a=l[:,i*width:(i+1)*width];b=r[:,i*width:(i+1)*width]
        blocks.append((a@a.T,b@b.T,a@b.T))
    result=l.new_zeros(heads,heads)
    for i,(aa,bb,ab) in enumerate(blocks):
        for j in range(i,heads):
            cc,dd,cd=blocks[j]
            value=((aa*dd+bb*cc+2*ab*cd.T)*output_gram).sum()/4
            result[i,j]=value;result[j,i]=value
    return result


def output_spectrum(l,r,metric_writer):
    gram=product_cross(l,r,l,r)
    covariance=metric_writer@gram@metric_writer.T
    covariance=(covariance+covariance.T)/2
    values=torch.linalg.eigvalsh(covariance).flip(0)
    return values,covariance.trace()


def dense_forms(l,r,w):
    native=(l[:,:,None]*r[:,None,:]+r[:,:,None]*l[:,None,:])/2
    return torch.einsum('vk,kij->vij',w,native)


def controls():
    torch.manual_seed(771);dt=torch.float64
    l,r=torch.randn(8,6,dtype=dt),torch.randn(8,6,dtype=dt)
    w=torch.randn(7,8,dtype=dt);o=torch.randn(6,6,dtype=dt)
    q=dense_forms(l,r,w);pull=torch.einsum('ia,vij,jb->vab',o,q,o)
    folded=dense_forms(l@o,r@o,w)
    fold=float((pull-folded).norm()/pull.norm())
    pair=head_energies(l@o,r@o,w.T@w,3)
    direct=torch.stack([torch.stack([pull[:,i*2:(i+1)*2,j*2:(j+1)*2].square().sum() for j in range(3)]) for i in range(3)])
    pair_error=float((pair-direct).norm()/direct.norm())
    cross=cross_port_energy(l,r,l@o,r@o,w.T@w)
    direct_cross=(q@o).square().sum()
    cross_error=float(abs(cross/direct_cross-1))
    p,_=torch.linalg.qr(torch.randn(6,6,dtype=dt))
    first,_=output_spectrum(l@o,r@o,w)
    rotated,_=output_spectrum(l@o@p,r@o@p,w)
    spectrum_error=float((first-rotated).norm()/first.norm())
    b=torch.randn(12,6,dtype=dt);z=torch.randn_like(b);x=b+z@o.T
    sigma=x.square().mean(1)+torch.finfo(torch.float32).eps
    direct_values=((x@l.T)*(x@r.T))@w.T/sigma[:,None]
    bb=torch.einsum('ni,vij,nj->nv',b,q,b)
    bz=2*torch.einsum('ni,vij,nj->nv',b,q,z@o.T)
    zz=torch.einsum('ni,vij,nj->nv',z,pull,z)
    replay=float((direct_values-(bb+bz+zz)/sigma[:,None]).norm()/direct_values.norm())
    return dict(fold_relative_error=fold,head_energy_relative_error=pair_error,
                mixed_port_relative_error=cross_error,orthogonal_rotation_spectrum_error=spectrum_error,
                residual_attention_normalized_replay_error=replay,
                passed=max(fold,pair_error,cross_error,spectrum_error,replay)<=1e-10)

if __name__=='__main__':
    import json
    from pathlib import Path
    torch.set_num_threads(2);r=controls();print(json.dumps(r,indent=2))
    Path(__file__).with_name('ATTENTION_OUTPUT_PULLBACK_V1_CONTROL.json').write_text(json.dumps(r,indent=2)+'\n')
    assert r['passed']
