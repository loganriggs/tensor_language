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

def ensemble_design(x,y,deltas,responses,u,v,response_weight=1.):
    """Equal-mass donor families, one globally normalized response energy.
    deltas[K,N,d], responses[K,N,V]; repeated families leave objective unchanged.
    """
    if response_weight<0:raise ValueError('negative response weight')
    if deltas.ndim!=3 or responses.ndim!=3 or deltas.shape[:2]!=responses.shape[:2] or deltas.shape[1]!=len(x):raise ValueError('expected matching donor family panels')
    value_energy=y.square().sum();energy=responses.square().sum()
    if value_energy<=0 or energy<=0:raise ValueError('undefined target energy')
    scale=(response_weight/energy).sqrt()
    blocks=[features(x,u,v)/value_energy.sqrt()];targets=[y/value_energy.sqrt()]
    for delta,response in zip(deltas,responses):
        blocks.append(response_features(x,delta,u,v)*scale);targets.append(response*scale)
    return torch.cat(blocks),torch.cat(targets)


def fit(x,y,delta,response,initial,response_weight=1.,steps=300,rate=.01,optimizer='adam',design_builder=None):
    import math
    from empirical_quartic_dictionary import readout
    from shared_quadratic_bank import normalize_bank
    build=joint_design if design_builder is None else design_builder
    shape=initial[0].shape
    parameters=[torch.nn.Parameter(z.flatten(0,1).clone()) for z in initial]
    opt=torch.optim.Adam(parameters,lr=rate) if optimizer=='adam' else torch.optim.Muon(parameters,lr=rate,weight_decay=0.,adjust_lr_fn='match_rms_adamw')
    best=None;history=[]
    for step in range(steps+1):
        u,v=normalize_bank(*(z.reshape(shape) for z in parameters))
        design,target=build(x,y,delta,response,u,v,response_weight)
        with torch.no_grad():c,_,ridge=readout(design,target)
        scales=design.square().mean(0).sqrt().clamp_min(1e-12)
        residual=design/scales@c-target
        loss=residual.square().sum()+ridge*c.square().sum()
        value=float(loss.detach())
        if best is None or value<best[0]:best=(value,step,u.detach().clone(),v.detach().clone(),(c/scales[:,None]).T.detach().clone())
        if step%50==0 or step==steps:history.append(dict(step=step,objective=value))
        if step==steps:break
        opt.zero_grad();loss.backward();opt.step()
        for group in opt.param_groups:group['lr']=rate*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/steps)))
    value,step,u,v,c=best
    return dict(objective=value,selected_step=step,history=history),dict(U=u,V=v,C=c)


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
