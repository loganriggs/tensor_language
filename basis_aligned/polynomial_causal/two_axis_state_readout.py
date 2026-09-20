"""Shared quadratic features for a two-axis state curve and its quartic norm."""
from directional_state_jet import second_jet
from final_readout_field_program import evaluate_fields


def basis(coordinates):
    import torch
    s,t=coordinates.unbind(-1)
    return torch.stack([torch.ones_like(s),s,t,s*s,s*t,t*t],dim=-1)


def capture_state_coefficients(function,zero):
    """function maps [B,2] edit amplitudes to a final [B,D] state."""
    import torch
    jets=[]
    for v in [[1.,0.],[0.,1.],[1.,1.]]:
        vector=torch.tensor(v,dtype=zero.dtype,device=zero.device)
        jets.append(second_jet(lambda t:function(t[:,None]*vector),zero))
    a,b,total=jets
    states=torch.stack([a[:,0],a[:,1],b[:,1],a[:,2],total[:,2]-a[:,2]-b[:,2],b[:,2]],dim=1)
    return states,dict(baseline_replay=max(float((a[:,0]-b[:,0]).abs().max()),float((a[:,0]-total[:,0]).abs().max())),
                       first_derivative_additivity=float((total[:,1]-a[:,1]-b[:,1]).abs().max()))


def compile_core(states,rows):
    import torch
    assert states.shape[1]==6 and states.shape[-1]>=6
    r=torch.linalg.qr(states.transpose(1,2)/states.shape[-1]**.5,mode='reduced').R
    index=torch.triu_indices(6,6,device=states.device)
    return dict(numerator=torch.einsum('bkd,bopd->bkop',states,rows).flatten(2),
                norm_triangular=r[:,index[0],index[1]])


def execute(core,coordinates):
    import torch
    n=core['numerator'];coordinates=torch.as_tensor(coordinates,dtype=n.dtype,device=n.device).reshape(-1,2).expand(len(n),2)
    phi=basis(coordinates);numerator=torch.einsum('bk,bkv->bv',phi,n)
    r=torch.zeros(len(n),6,6,dtype=n.dtype,device=n.device)
    index=torch.triu_indices(6,6,device=n.device);r[:,index[0],index[1]]=core['norm_triangular']
    norm_features=torch.einsum('bij,bj->bi',r,phi)
    denominator=norm_features.square().sum(-1,keepdim=True)+torch.finfo(torch.float32).eps
    return evaluate_fields(torch.cat([numerator,denominator],dim=-1))
