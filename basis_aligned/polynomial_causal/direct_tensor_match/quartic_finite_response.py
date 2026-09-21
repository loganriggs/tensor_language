"""Shared quartic dictionary finite differences and joint relative-energy design."""
import torch
from empirical_quartic_dictionary import features

def response_features(x,delta,u,v):
    n=len(x);m,k,d=u.shape
    ux=(x@u.flatten(0,1).T).reshape(n,m,k)
    vx=(x@v.flatten(0,1).T).reshape(n,m,k)
    ud=(delta@u.flatten(0,1).T).reshape(n,m,k)
    vd=(delta@v.flatten(0,1).T).reshape(n,m,k)
    q=(ux*vx).sum(-1)
    dq=(ux*vd+ud*vx+ud*vd).sum(-1)
    i,j=torch.triu_indices(m,m,device=x.device)
    return q[:,i]*dq[:,j]+dq[:,i]*q[:,j]+dq[:,i]*dq[:,j]

def joint_design(x,y,delta,response,u,v,response_weight=1.):
    """One response family; sum relative squared value and weighted response errors.
    Zero target energy is undefined and rejected, rather than silently rescaled.
    Caller chooses ridge/profiled optimization; no held-out targets belong here.
    """
    if response_weight<0:raise ValueError('response_weight must be nonnegative')
    value_energy=y.square().sum();response_energy=response.square().sum()
    if value_energy<=0 or response_energy<=0:raise ValueError('target energies must be positive')
    a=value_energy.rsqrt();b=(response_weight/response_energy).sqrt()
    return torch.cat([features(x,u,v)*a,response_features(x,delta,u,v)*b]),torch.cat([y*a,response*b])

def controls():
    records=[]
    for seed in range(5):
        torch.manual_seed(6200+seed);d=seed+3
        x=torch.randn(23,d,dtype=torch.float64);delta=torch.randn_like(x)*.2
        u=torch.randn(3,2,d,dtype=torch.float64,requires_grad=True);v=torch.randn_like(u,requires_grad=True)
        # Independent dense quadratic matrices, followed by explicit products.
        matrices=torch.einsum('mkd,mke->mde',u,v)
        q=lambda z:torch.einsum('nd,mde,ne->nm',z,matrices,z)
        i,j=torch.triu_indices(3,3);before,after=q(x),q(x+delta)
        reference=after[:,i]*after[:,j]-before[:,i]*before[:,j]
        actual=response_features(x,delta,u,v)
        error=float(((actual-reference).norm()/reference.norm()).detach())
        probe=torch.randn_like(actual);ga=torch.autograd.grad((actual*probe).sum(),(u,v),retain_graph=True);gr=torch.autograd.grad((reference*probe).sum(),(u,v),retain_graph=True)
        gradient=max(float((a-b).norm()/b.norm()) for a,b in zip(ga,gr))
        assert torch.count_nonzero(response_features(x,torch.zeros_like(delta),u,v))==0
        y=torch.randn(23,4,dtype=torch.float64);response=torch.randn_like(y);writer=torch.randn(6,4,dtype=torch.float64)
        design,target=joint_design(x,y,delta,response,u,v,.7)
        expected=(features(x,u,v)@writer-y).square().sum()/y.square().sum()+.7*(reference@writer-response).square().sum()/response.square().sum()
        objective_error=float(abs(((design@writer-target).square().sum()-expected)/expected).detach())
        records.append(dict(seed=seed,dimension=d,response_error=error,gradient_error=gradient,objective_error=objective_error))
    assert max(max(r[k] for k in ['response_error','gradient_error','objective_error']) for r in records)<1e-12
    return records

if __name__=='__main__':
    import json
    from pathlib import Path
    torch.set_num_threads(2)
    result=dict(controls=controls(),scope='Five algebra/gradient controls; not five structural recovery results or native optimization.')
    Path(__file__).with_name('QUARTIC_FINITE_RESPONSE_CONTROLS_V1.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
