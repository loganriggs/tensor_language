"""One conditional S/R/O graph with native-value coordinates and shared output.

S is the earlier cue-selective scalar, R the added even-value remainder, and O
the odd-key component. The native downstream function is not linearized.
"""
import torch
import torch.nn.functional as F


def compile_program(value_program, scalar_reader, scalar_writer):
    p={k:v.clone() for k,v in value_program.items()}
    # LAPACK's solution may view a padded RHS buffer; own only the stored coordinates.
    p['scalar_value_coordinates']=torch.linalg.lstsq(p['current_value'].double().T,scalar_reader.double()).solution.clone()
    p['scalar_output_coordinates']=torch.linalg.lstsq(p['output'].double(),scalar_writer.double()).solution.clone()
    return p


class SharedGraph:
    def __init__(self, p):
        self.p=p
        keys=torch.cat([p['k1'],p['k2']]).double()
        self.adapters=((keys@keys.T)@p['key_coordinates']).reshape(2,128,64)

    def state(self,current,initial):
        """Prepare three 128-channel branches from supplied normalized contexts."""
        p=self.p;x=current.double();n=x.shape[1]
        inv=1/(10000**(torch.arange(0,128,2,dtype=torch.float32)/128))
        angles=torch.outer(torch.arange(n,dtype=torch.float32),inv)
        co,si=angles.cos().bfloat16().to(x.device),angles.sin().bfloat16().to(x.device)
        def rotate(x):
            a,b=x.chunk(2,-1)
            return torch.cat((a*co+b*si,-a*si+b*co),-1)
        nums=[x@p[k].double().T for k in ['k1','k2']]
        shared=torch.cat(nums,-1)@p['key_coordinates'];full=[];reflected=[]
        for j,(qn,kn) in enumerate([('q1','k1'),('q2','k2')]):
            q=rotate(F.rms_norm(F.linear(current,p[qn].to(current.dtype)),(128,),eps=torch.finfo(torch.float32).eps)).double()
            key=F.linear(current,p[kn].to(current.dtype))
            den=(key.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt().double()
            inside=shared@self.adapters[j].T
            full.append(q@rotate(nums[j]/den).transpose(-1,-2)/128)
            reflected.append(q@rotate((nums[j]-2*inside)/den).transpose(-1,-2)/128)
        native_gate=full[0]*full[1];ref=reflected[0]*reflected[1]
        mask=~torch.ones(n,n,dtype=torch.bool,device=x.device).tril()
        even=((native_gate+ref)/2).masked_fill(mask,0)
        odd=((native_gate-ref)/2).masked_fill(mask,0)
        current_values=x@p['current_value'].double().T
        first_values=initial.double()@p['first_value'].double().T
        lam=p['mixture'].double();mixed=(1-lam)*current_values+lam*first_values
        value_read=current_values@p['scalar_value_coordinates']
        scalar=(even@value_read[...,None])*p['scalar_output_coordinates']
        even_channels=even@mixed
        return scalar,even_channels-scalar,odd@mixed

    def write(self,branches,gains=(1,1,1)):
        """Combine component writes in head space before one native output map."""
        channels=sum(float(g)*branch for g,branch in zip(gains,branches))
        return channels@self.p['output'].double().T
