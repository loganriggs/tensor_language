"""Single-head frozen-norm approximation, with two declared native inputs."""
import torch
import torch.nn.functional as F
import head8
from attention import channels

def execute(p,residual7,donor_city7,token_ids,recipient_token,donor_token,city,destination,strength=.5):
 entry=p['entry'];head=p['head8'];lookup={int(t):i for i,t in enumerate(entry['token_ids'].tolist())}
 try:
  idx=torch.tensor([[lookup[int(t)] for t in row] for row in token_ids.tolist()],device=residual7.device)
  donor_idx=lookup[int(donor_token)]
 except KeyError as e:raise ValueError('Token outside frozen initial-state table') from e
 initial=F.embedding(idx,entry['initial_table']).to(residual7.dtype);lam=entry['lambdas8']
 mixed=lam[0]*residual7+lam[1]*initial
 current8=F.rms_norm(mixed,(1152,))
 donor_initial=entry['initial_table'][donor_idx][None].to(donor_city7.dtype)
 donor8=F.rms_norm(lam[0]*donor_city7+lam[1]*donor_initial,(1152,))
 values=channels(head,current8,token_ids).reshape(*current8.shape[:2],128)
 g=mixed+F.linear(values,head['output'])
 d=(strength*head8.execute(head,current8,donor8,recipient_token,donor_token,city,destination)).double()
 z=g.double();mlp=p['mlp8'];L=mlp['left'].double();R=mlp['right'].double();D=mlp['down'].double()
 ld=d@L.T;rd=d@R.T;lr=z@L.T;rr=z@R.T
 s=z.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps
 return d+((ld*rr+lr*rd+ld*rd)/s)@D.T
