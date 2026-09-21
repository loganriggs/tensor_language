"""Exact uniform-alpha squared polynomial response objective on fixed donor paths.
Five-point Gauss-Legendre is exact through degree9; quartic residual square is8.
Only polynomial numerators: nonlinear normalized/logit endpoints are not covered.
"""
import numpy as np
import torch
from empirical_quartic_dictionary import features
from quartic_finite_response import response_features

def rule(reference,count=5):
    nodes,weights=np.polynomial.legendre.leggauss(count)
    return (torch.as_tensor((nodes+1)/2,device=reference.device,dtype=reference.dtype),
            torch.as_tensor(weights/2,device=reference.device,dtype=reference.dtype))

def design(x,y,donor_delta,responses,u,v,response_weight=1.):
    """responses shape[5,N,V], exact teacher F(x+alpha*delta)-F(x).
    Joint objective: relative value error + weight * integrated response error
    divided by integrated teacher response energy. This is one global response
    denominator; nodes with small responses are not separately upweighted.
    """
    if response_weight<0:raise ValueError('negative response weight')
    nodes,weights=rule(x)
    if responses.shape[:2]!=(5,len(x)):raise ValueError('expected five response panels')
    energy=(responses.square().sum((1,2))*weights).sum();value_energy=y.square().sum()
    if energy<=0 or value_energy<=0:raise ValueError('undefined relative energy')
    blocks=[features(x,u,v)/value_energy.sqrt()];targets=[y/value_energy.sqrt()]
    for a,w,r in zip(nodes,weights,responses):
        scale=(response_weight*w/energy).sqrt()
        blocks.append(response_features(x,a*donor_delta,u,v)*scale);targets.append(r*scale)
    return torch.cat(blocks),torch.cat(targets)

def controls():
    records=[]
    for seed in range(5):
        torch.manual_seed(8120+seed);d=3+seed
        x=torch.randn(19,d,dtype=torch.float64);delta=torch.randn_like(x)
        u=torch.randn(3,2,d,dtype=torch.float64,requires_grad=True);v=torch.randn_like(u,requires_grad=True);c=torch.randn(6,4,dtype=torch.float64)
        # Independent random degree-four vector polynomial along alpha, arbitrary
        # constant cancels when comparing changes from alpha0.
        target=torch.randn(4,19,4,dtype=torch.float64)
        def response(a):return sum(a**(j+1)*target[j] for j in range(4))
        def loss(count):
            nodes,weights=rule(x,count);return sum(w*(response_features(x,a*delta,u,v)@c-response(a)).square().sum() for a,w in zip(nodes,weights))
        l5,l9=loss(5),loss(9);g5=torch.autograd.grad(l5,(u,v),retain_graph=True);g9=torch.autograd.grad(l9,(u,v),retain_graph=True)
        # Independently recover the polynomial coefficients by evaluation using
        # ordinary dense quadratic forms; integrate their product analytically.
        mats=torch.einsum('mkd,mke->mde',u,v)
        q=lambda z:torch.einsum('nd,mde,ne->nm',z,mats,z)
        i,j=torch.triu_indices(3,3);q0=q(x);base=(q0[:,i]*q0[:,j])@c
        grid=torch.tensor([.2,.4,.7,1.],dtype=x.dtype);ys=[]
        for a in grid:
            qa=q(x+a*delta);ys.append((qa[:,i]*qa[:,j])@c-base-response(a))
        vand=grid[:,None]**torch.arange(1,5,dtype=x.dtype)[None,:]
        coefficients=torch.linalg.solve(vand,torch.stack(ys).flatten(1)).reshape(4,19,4)
        integral=sum((coefficients[a]*coefficients[b]).sum()/(a+b+3) for a in range(4) for b in range(4))
        ga=torch.autograd.grad(integral,(u,v),retain_graph=True)
        nodes,weights=rule(x);rs=torch.stack([response(a) for a in nodes]);y=torch.randn(19,4,dtype=x.dtype);dd,tt=design(x,y,delta,rs,u,v,.7)
        response_energy=sum(w*response(a).square().sum() for a,w in zip(nodes,weights))
        expected=(features(x,u,v)@c-y).square().sum()/y.square().sum()+.7*l5/response_energy
        ratios=dict(quadrature=float(abs((l5-l9)/l9).detach()),coefficient_integral=float(abs((l5-integral)/integral).detach()),gradient=max(float((a-b).norm()/b.norm()) for a,b in zip(g5,g9)),coefficient_gradient=max(float((a-b).norm()/b.norm()) for a,b in zip(g5,ga)),normalized_design=float(abs(((dd@c-tt).square().sum()-expected)/expected).detach()))
        records.append(dict(seed=seed,**ratios))
    assert max(max(v for k,v in r.items() if k!='seed') for r in records)<1e-10
    return records

if __name__=='__main__':
    import json
    from pathlib import Path
    torch.set_num_threads(2);result=dict(controls=controls(),scope='Five algebraic controls, independent dense coefficient integrals and gradients. Not native fit or behavioral evidence.')
    Path(__file__).with_name('QUARTIC_PATH_OBJECTIVE_CONTROLS_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
