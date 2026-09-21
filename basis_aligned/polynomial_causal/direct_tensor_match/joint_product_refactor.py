"""Local multi-output bilinear refactor with exact output variable projection."""
import itertools,math
import torch

def reconstruct(core,left,right):
    phi=torch.einsum('ri,rj->rij',left,right).flatten(1)
    target=core.permute(2,0,1).flatten(1)
    gram=phi@phi.T
    ridge=1e-10*gram.diag().mean().clamp_min(1e-30)
    cross=target@phi.T
    writer=torch.linalg.solve(gram+ridge*torch.eye(len(gram),dtype=gram.dtype),cross.T).T
    residual=target-writer@phi
    return residual.square().sum(),writer

def best_subset(core,left,right,width):
    best=None
    for ids in itertools.combinations(range(len(left)),width):
        a=left[list(ids)];b=right[list(ids)];loss,c=reconstruct(core,a,b)
        if best is None or float(loss)<best[0]:best=(float(loss),list(ids),a.clone(),b.clone(),c.clone())
    return best

def fit(core,width,optimizer,lr,seed,steps,initial=None):
    gen=torch.Generator().manual_seed(seed)
    d=core.shape[0]
    raw=[torch.nn.Parameter(torch.randn(width,d,dtype=core.dtype,generator=gen) if initial is None else v.clone()+1e-3*torch.randn(v.shape,dtype=v.dtype,generator=gen)) for v in ([None,None] if initial is None else initial)]
    opt=torch.optim.Adam(raw,lr=lr) if optimizer=='adam' else torch.optim.Muon(raw,lr=lr,weight_decay=0.,adjust_lr_fn='match_rms_adamw')
    norm=core.square().sum();best=None
    for step in range(steps+1):
        a,b=[v/v.norm(dim=1,keepdim=True).clamp_min(1e-20) for v in raw]
        loss,c=reconstruct(core,a,b)
        value=float(loss.detach()/norm)
        if best is None or value<best[0]:best=(value,a.detach().clone(),b.detach().clone(),c.detach().clone(),step)
        if step==steps:break
        opt.zero_grad();(loss/norm).backward();opt.step()
        for group in opt.param_groups:group['lr']=lr*(.05+.95*.5*(1+math.cos(math.pi*(step+1)/steps)))
    return best
