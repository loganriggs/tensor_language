"""Donor-free removal of one named native source; new counterfactual."""
import torch
import torch.nn.functional as F
import head8
from attention import channels

def source(p,residual7,token_ids,city,destination,strength=.5):
 entry=p['entry'];head=p['head8'];lookup={int(t):i for i,t in enumerate(entry['token_ids'].tolist())}
 idx=torch.tensor([[lookup[int(t)] for t in row] for row in token_ids.tolist()],device=residual7.device)
 initial=F.embedding(idx,entry['initial_table']).to(residual7.dtype);lam=entry['lambdas8']
 mixed=lam[0]*residual7+lam[1]*initial;current=F.rms_norm(mixed,(1152,))
 route=head8.routing(head,current,current[:,city],city)
 inherited=head8.inherited(head,int(token_ids[0,city])).to(current.dtype)
 delta=-strength*F.linear(route[...,None]*head['mixture']*inherited[:,None],head['output'])*destination[None,:,None]
 local=channels(head,current,token_ids).reshape(*current.shape[:2],128)
 return delta,mixed+F.linear(local,head['output'])

def execute(p,residual7,token_ids,city,destination,strength=.5):
 delta,g=source(p,residual7,token_ids,city,destination,strength)
 d=delta.double();z=g.double();m=p['mlp8'];L=m['left'].double();R=m['right'].double();D=m['down'].double()
 ld=d@L.T;rd=d@R.T;lr=z@L.T;rr=z@R.T
 return d+((ld*rr+lr*rd+ld*rd)/(z.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps))@D.T
