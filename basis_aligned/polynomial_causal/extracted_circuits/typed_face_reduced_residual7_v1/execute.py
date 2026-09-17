"""Generate the raw MLP8 context, retaining an explicit native RMS scale."""
import torch
import torch.nn.functional as F
import coupled
import head8
from attention8_context_channels_v1 import channels
from head2_mlp8_cross_edit_v1 import retained_delta
import attention8

def execute(p,residual7,donor_city7,token_ids,recipient_token,donor_token,city,destination,strength=.5):
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
    a=attention8.execute(p['context'],current8,token_ids)
    g=mixed+a
    sl=slice(256,384);head=dict(p['local']['head8']);context=p['context']
    for h,c in [('q1','q1'),('k1','k1'),('q2','q2'),('k2','k2'),('current_value','value')]:head[h]=context[c][sl]
    head['output']=context['output'][:,sl];head['mixture']=context['mixture']
    local={'head8':head,'mlp8':p['local']['mlp8']}
    complete=coupled.execute(local,current8,donor_city8,g,recipient_token,donor_token,city,destination,strength)
    delta=strength*head8.execute(head,current8,donor_city8,recipient_token,donor_token,city,destination)
    a2=F.linear(channels(p['context'],current8,token_ids)[:,:,2],context['output'][:,sl])
    return retained_delta(complete,delta,a,a2,g,p['local']['mlp8'])
