"""Exact original key-subspace coordinates in existing full key outputs."""
import torch
import torch.nn.functional as F


class KeySpanParent:
    def __init__(self,p):
        self.p=p
        self.adapters=[]
        for i in range(2):
            k=torch.cat([p['k1'][i],p['k2'][i]]).double()
            self.adapters.append((k@k.T)@p['key_coordinates'][i])

    def scalar(self,current,tokens,index):
        p=self.p;n=current.shape[1];x=current.double()
        inv=1/(10000**(torch.arange(0,128,2,dtype=torch.float32)/128))
        angle=torch.outer(torch.arange(n,dtype=torch.float32),inv)
        co=angle.cos().bfloat16().to(current.device);si=angle.sin().bfloat16().to(current.device)
        def rot(x):
            a,b=x.chunk(2,-1);return torch.cat((a*co+b*si,-a*si+b*co),-1)
        nums=[x@p[k][index].double().T for k in ['k1','k2']]
        shared=torch.cat(nums,-1)@p['key_coordinates'][index]
        full=[];reflected=[]
        for j,(qn,kn) in enumerate([('q1','k1'),('q2','k2')]):
            q=rot(F.rms_norm(F.linear(current,p[qn][index]),(128,),eps=torch.finfo(torch.float32).eps)).double()
            nk=F.linear(current,p[kn][index]);den=(nk.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt().double()
            inside=shared@self.adapters[index][j*128:(j+1)*128].T
            full.append(q@rot(nums[j]/den).transpose(-1,-2)/128)
            reflected.append(q@rot((nums[j]-2*inside)/den).transpose(-1,-2)/128)
        gamma=(full[0]*full[1]+reflected[0]*reflected[1])/2
        gamma=gamma.masked_fill(~torch.ones(n,n,dtype=torch.bool,device=current.device).tril(),0)
        value=p['first_token_value'][tokens] if index==0 else x@p['current_value_reader']
        return (gamma@value[...,None])[...,0]
