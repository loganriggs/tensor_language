"""Native-port adapter for the exact single-token row+column executor."""
import torch
import torch.nn.functional as F
from fixed_query_source_column import source_column
from source_query_row import query_row
from native_fixed_query_source import prepare

def evaluate(att,z,positions,background):
    from jacclust.tt_model import apply_rotary_emb
    b,t,d=z.shape;h,e=att.n_head,att.head_dim;batch=torch.arange(b,device=z.device);local=z[batch,positions][:,None]
    cos=background['cos'][0,positions][:,None];sin=background['sin'][0,positions][:,None]
    def key(name):
        v=getattr(att,name)(local).reshape(b,1,h,e)
        return apply_rotary_emb(F.rms_norm(v,(e,)),cos,sin)[:,0]
    k11,k21=key('c_k'),key('c_k2');q11,q21=key('c_q'),key('c_q2')
    old=lambda name:background[name][batch,positions]
    v1=(1-att.lamb)*att.c_v(local).reshape(b,h,e)+att.lamb*background['first'].reshape(b,t,h,e)[batch,positions]
    # FP64 summation minimizes error from distributing the native output map.
    delta=source_column(background['q1'].double(),background['q2'].double(),old('k1').double(),old('k2').double(),old('v').double(),k11.double(),k21.double(),v1.double(),att.c_proj.weight.double(),positions)
    k1,k2,v=[background[name].double().clone() for name in ['k1','k2','v']]
    k1[batch,positions]=k11.double();k2[batch,positions]=k21.double();v[batch,positions]=v1.double()
    row=query_row(old('q1').double(),old('q2').double(),q11.double(),q21.double(),k1,k2,v,att.c_proj.weight.double(),positions)
    delta[batch,positions]+=row
    return (background['base'].double()+delta).to(z.dtype)
