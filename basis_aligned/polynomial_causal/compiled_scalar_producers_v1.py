"""Torch-only scalar producers with explicit native QK and token value lookup."""
import torch
import torch.nn.functional as F

def head_scalar(current,token_ids,p,index):
    """Inputs [batch,sequence,1152] and IDs; output [batch,sequence]."""
    width=p['q1'].shape[1];length=current.shape[1]
    projected=[F.rms_norm(F.linear(current,p[name][index]),(width,),eps=torch.finfo(torch.float32).eps) for name in ('q1','k1','q2','k2')]
    # Native tables are computed on CPU then rounded to BF16 before device transfer.
    inv=1.0/(10000**(torch.arange(0,width,2,dtype=torch.float32)/width))
    angle=torch.outer(torch.arange(length,dtype=torch.float32),inv)
    cos=angle.cos().bfloat16().to(current.device);sin=angle.sin().bfloat16().to(current.device)
    def rotate(x):
        a,b=x.chunk(2,-1)
        return torch.cat([a*cos+b*sin,-a*sin+b*cos],-1).double()
    q,k,q2,k2=[rotate(x) for x in projected]
    routing=(q@k.transpose(-1,-2)/width)*(q2@k2.transpose(-1,-2)/width)
    routing=routing.masked_fill(~torch.ones(length,length,dtype=torch.bool,device=current.device).tril(),0)
    value=current.double()@p['current_value_readers'][index]+p['first_token_values'][token_ids,index]
    return (routing@value[...,None])[...,0]

def execute(current8,current9,token_ids,p,masks=(1.,1.)):
    scalar=sum(float(masks[i])*p['output_coefficients'][i]*head_scalar(x,token_ids,p,i) for i,x in enumerate((current8,current9)))
    return scalar[...,None]*p['shared_output']
