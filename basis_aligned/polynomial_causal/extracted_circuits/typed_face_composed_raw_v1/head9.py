"""PyTorch-only conditional head9 odd-value delta, without inherited values."""
import torch
import torch.nn.functional as F


class Program:
    def __init__(self,p):
        self.p=p
        keys=torch.cat([p['k1'],p['k2']]).double()
        self.adapters=((keys@keys.T)@p['key_coordinates']).reshape(2,128,64)

    def routing(self,current):
        p=self.p;n=current.shape[1]
        inv=1/(10000**(torch.arange(0,128,2,dtype=torch.float32)/128))
        angles=torch.outer(torch.arange(n,dtype=torch.float32),inv)
        co,si=angles.cos().bfloat16().to(current.device),angles.sin().bfloat16().to(current.device)
        def rotate(z):
            a,b=z.chunk(2,-1)
            return torch.cat((a*co+b*si,-a*si+b*co),-1)
        nums=[current.double()@p[k].double().T for k in ['k1','k2']]
        shared=torch.cat(nums,-1)@p['key_coordinates'];full=[];reflected=[]
        for j,(qn,kn) in enumerate([('q1','k1'),('q2','k2')]):
            q=rotate(F.rms_norm(F.linear(current,p[qn].to(current.dtype)),(128,),eps=torch.finfo(torch.float32).eps)).double()
            key=F.linear(current,p[kn].to(current.dtype));den=(key.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt().double()
            inside=shared@self.adapters[j].T
            full.append(q@rotate(nums[j]/den).transpose(-1,-2)/128)
            reflected.append(q@rotate((nums[j]-2*inside)/den).transpose(-1,-2)/128)
        odd=(full[0]*full[1]-reflected[0]*reflected[1])/2
        return odd.masked_fill(~torch.ones(n,n,dtype=torch.bool,device=current.device).tril(),0)

    def execute(self,raw_mixed9,delta8,lambda90,destination):
        current=F.rms_norm(raw_mixed9,(raw_mixed9.shape[-1],))
        changed=F.rms_norm(raw_mixed9+lambda90*delta8,(raw_mixed9.shape[-1],))
        value_delta=(1-self.p['mixture'].double())*((changed.double()-current.double())@self.p['current_value'].double().T)
        mask=torch.as_tensor(destination,device=current.device)
        if mask.ndim==1:mask=mask[None].expand(current.shape[0],-1)
        channels=self.routing(current)[...,None]*value_delta[:,None]*mask[:,None,:,None]
        return channels.sum(-2)@self.p['output'].double().T
