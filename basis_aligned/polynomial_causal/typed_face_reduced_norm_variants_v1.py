"""Generate the raw MLP8 context, retaining an explicit native RMS scale."""
import torch
import torch.nn.functional as F
import head8
from attention8_context_channels_v1 import channels

def execute(p,residual7,donor_city7,token_ids,recipient_token,donor_token,city,destination,strength=.5,mode="exact"):
    assert mode in ("exact","omit_correction","freeze_denominator")
    entry=p['reentry'];lookup={int(t):i for i,t in enumerate(entry['token_ids'].tolist())}
    try:
        idx=torch.tensor([[lookup[int(t)] for t in row] for row in token_ids.tolist()],device=residual7.device)
        donor_idx=lookup[int(donor_token)]
    except KeyError as e:raise ValueError('Token outside frozen initial-state table') from e
    initial=F.embedding(idx,entry['initial_table']).to(residual7.dtype);lam=entry['lambdas8']
    mixed=lam[0]*residual7+lam[1]*initial
    current8=F.rms_norm(mixed,(1152,))
    donor_initial=entry['initial_table'][donor_idx][None].to(donor_city7.dtype)
    donor_city8=F.rms_norm(lam[0]*donor_city7+lam[1]*donor_initial,(1152,))
    values=channels(p['context'],current8,token_ids)
    a=F.linear(values.reshape_as(current8),p['context']['output'])
    g=mixed+a
    sl=slice(256,384);head=dict(p['local']['head8']);context=p['context']
    for h,c in [('q1','q1'),('k1','k1'),('q2','q2'),('k2','k2'),('current_value','value')]:head[h]=context[c][sl]
    head['output']=context['output'][:,sl];head['mixture']=context['mixture']
    delta=strength*head8.execute(head,current8,donor_city8,recipient_token,donor_token,city,destination)
    a2=F.linear(values[:,:,2],context['output'][:,sl])
    mlp=p['local']['mlp8'];L=mlp['left'].double();R=mlp['right'].double();D=mlp['down'].double()
    z=g.double();d=delta.double();omitted=(a-a2).double();retained=z-omitted
    eps=torch.finfo(torch.float32).eps;s0=z.square().mean(-1,keepdim=True)+eps;s1=(z+d).square().mean(-1,keepdim=True)+eps
    ld=d@L.T;rd=d@R.T;lr=retained@L.T;rr=retained@R.T
    numerator=ld*rr+lr*rd+ld*rd
    hidden=numerator/(s0 if mode=='freeze_denominator' else s1)
    if mode=='exact':hidden=hidden+(1/s1-1/s0)*((z@L.T)*(z@R.T))
    return d+hidden@D.T
