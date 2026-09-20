"""Native-port adapter for the conditional fixed-query source-column executor."""
import torch
import torch.nn.functional as F
from fixed_query_source_column import source_column

def prepare(att,z,first):
    from jacclust.tt_model import apply_rotary_emb
    b,t,d=z.shape;h,e=att.n_head,att.head_dim
    cos,sin=att.rotary(torch.zeros(b,t,h,e,device=z.device,dtype=z.dtype))
    def project(name):
        v=getattr(att,name)(z).reshape(b,t,h,e)
        return apply_rotary_emb(F.rms_norm(v,(e,)),cos,sin)
    q1,q2,k1,k2=[project(name) for name in ['c_q','c_q2','c_k','c_k2']]
    v=(1-att.lamb)*att.c_v(z).reshape(b,t,h,e)+att.lamb*first.reshape(b,t,h,e)
    base,_=att(z,first)
    return dict(q1=q1,q2=q2,k1=k1,k2=k2,v=v,base=base,cos=cos,sin=sin,first=first)

def evaluate(att,z,positions,background):
    from jacclust.tt_model import apply_rotary_emb
    b,t,d=z.shape;h,e=att.n_head,att.head_dim;batch=torch.arange(b,device=z.device);local=z[batch,positions][:,None]
    cos=background['cos'][0,positions][:,None];sin=background['sin'][0,positions][:,None]
    def key(name):
        v=getattr(att,name)(local).reshape(b,1,h,e)
        return apply_rotary_emb(F.rms_norm(v,(e,)),cos,sin)[:,0]
    k11,k21=key('c_k'),key('c_k2')
    old=lambda name:background[name][batch,positions]
    v1=(1-att.lamb)*att.c_v(local).reshape(b,h,e)+att.lamb*background['first'].reshape(b,t,h,e)[batch,positions]
    # FP64 summation minimizes error from distributing the native output map.
    delta=source_column(background['q1'].double(),background['q2'].double(),old('k1').double(),old('k2').double(),old('v').double(),k11.double(),k21.double(),v1.double(),att.c_proj.weight.double(),positions)
    return (background['base'].double()+delta).to(z.dtype)
