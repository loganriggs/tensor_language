"""Exact scalar-times-vector mixed-input partition with explicit RMS epsilon."""
import torch


def partition(raw,projections,epsilon):
    x=raw.double();g=torch.rsqrt(x.square().mean(-1,keepdim=True)+epsilon)
    xs=torch.einsum('kij,jtd->kitd',projections,x)
    gs=torch.einsum('kij,jtd->kitd',projections,g)
    inherited=gs[0]*xs[3]
    normalization=gs[3]*xs[0]+gs[1]*xs[2]+gs[2]*xs[1]
    normalized=g*x
    full=torch.einsum('ij,jtd->itd',projections[3],normalized)
    error=float((full-inherited-normalization).norm()/full.norm().clamp_min(1e-30))
    return {'raw':inherited,'normalization':normalization,'full':full,
            'normalized':normalized,'closure_relative':error}


def controls(corners):
    from symmetric_factor_interaction_v1 import projectors
    p=projectors(corners);o=torch.tensor([c[2] for c in corners],dtype=torch.float64)
    h=torch.tensor([c[4] for c in corners],dtype=torch.float64)
    # No raw mixed term: normalization alone can create one.
    raw=torch.stack((2+o,.5+h),-1)[:,None,:]
    a=partition(raw,p,1e-7)
    assert a['raw'].norm()==0 and a['normalization'].norm()>.01
    # Constant norm with a signed mixed component: all interaction is inherited.
    raw=torch.stack((o*h,torch.ones_like(o)),-1)[:,None,:]
    b=partition(raw,p,1e-7)
    assert b['normalization'].norm()==0 and b['raw'].norm()>1
    assert max(a['closure_relative'],b['closure_relative'])<1e-12
    return {'passed':True,'normalization_only_control':True,'inherited_only_control':True,
            'closure_max_relative':max(a['closure_relative'],b['closure_relative'])}
