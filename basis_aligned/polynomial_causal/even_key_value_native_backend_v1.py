"""Full-value even/odd head9.8 components with shared native first values."""
import torch
import torch.nn.functional as F


class FullValueComponents:
    def __init__(self, program, device):
        self.p={k:v.to(device) for k,v in program.items()}
        p=self.p
        keys=torch.cat([p['k1'],p['k2']]).double()
        self.adapters=((keys@keys.T)@p['key_coordinates']).reshape(2,128,64)

    def __call__(self,current,first_values):
        p=self.p;x=current.double();n=x.shape[1]
        inv=1/(10000**(torch.arange(0,128,2,dtype=torch.float32)/128))
        angles=torch.outer(torch.arange(n,dtype=torch.float32),inv)
        co,si=angles.cos().bfloat16().to(x.device),angles.sin().bfloat16().to(x.device)
        def rotate(x):
            a,b=x.chunk(2,-1)
            return torch.cat((a*co+b*si,-a*si+b*co),-1)
        nums=[x@p[k].double().T for k in ['k1','k2']]
        shared=torch.cat(nums,-1)@p['key_coordinates']
        full,reflected=[],[]
        for j,(qn,kn) in enumerate([('q1','k1'),('q2','k2')]):
            q=rotate(F.rms_norm(F.linear(current,p[qn].to(current.dtype)),(128,),eps=torch.finfo(torch.float32).eps)).double()
            native_key=F.linear(current,p[kn].to(current.dtype))
            den=(native_key.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt().double()
            inside=shared@self.adapters[j].T
            full.append(q@rotate(nums[j]/den).transpose(-1,-2)/128)
            reflected.append(q@rotate((nums[j]-2*inside)/den).transpose(-1,-2)/128)
        gate=full[0]*full[1];ref=reflected[0]*reflected[1]
        even=(gate+ref)/2;odd=(gate-ref)/2
        mask=~torch.ones(n,n,dtype=torch.bool,device=x.device).tril()
        even=even.masked_fill(mask,0);odd=odd.masked_fill(mask,0)
        lam=p['mixture'].double()
        values=(1-lam)*(x@p['current_value'].double().T)+lam*first_values.double()
        output=p['output'].double()
        return (even@values)@output.T,(odd@values)@output.T
