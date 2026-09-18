"""Native-weight MLP8 mediator from z8, intervention and token IDs; no supplied RMS."""
import torch
import torch.nn.functional as F

def prepare(program):
    runtime={k:(v.double() if v.is_floating_point() else v) for k,v in program.items()}
    runtime['folded_down']=runtime['value_reader']@runtime['down']
    return runtime

def execute(program,z,delta,token_ids,return_rho=False):
    if z.ndim!=3 or z.shape[0]!=1 or z.shape[-1]!=1152 or delta.shape!=z.shape or token_ids.shape!=z.shape[:2]:raise ValueError('Expected batch1 native z/delta and matching token IDs')
    if any(x.device.type!='cpu' for x in [z,delta,token_ids]):raise ValueError('CPU interface only')
    lookup={int(t):i for i,t in enumerate(program['token_ids'].tolist())}
    try:index=torch.tensor([[lookup[int(t)] for t in row] for row in token_ids.tolist()])
    except KeyError as exc:raise ValueError('Token outside frozen weight-derived table') from exc
    z,delta=z.double(),delta.double();edited=z+delta;eps=torch.finfo(torch.float32).eps
    h0=F.linear(z,program['left'])*F.linear(z,program['right'])/(z.square().mean(-1,keepdim=True)+eps)
    h1=F.linear(edited,program['left'])*F.linear(edited,program['right'])/(edited.square().mean(-1,keepdim=True)+eps)
    m1=F.linear(h1,program['down'])+program['bias'];lam=program['lambdas9'];initial=F.embedding(index,program['initial_table'])
    mixed=lam[0]*(edited+m1)+lam[1]*initial;rho=(mixed.square().mean(-1,keepdim=True)+eps).sqrt()
    value=(1-program['mixture'])*lam[0]/rho*F.linear(h1-h0,program['folded_down'])
    return (value,rho) if return_rho else value
