"""Weight-only optimal common reader subspace for the two key consumers."""
import torch
from shared_key_reads_v1 import compile_program as full_compile
from contracted_qk_response_v1 import scalar as old_scalar

def compile_program(native,low,gain,rank):
    p=full_compile(native,low,gain)
    if not 1<=rank<=64:raise ValueError('rank must lie in [1,64]')
    adapter=p['inside_adapters'].reshape(256,64)
    consumer_norms=adapter.reshape(2,128,64).norm(dim=(1,2));balanced=(adapter.reshape(2,128,64)/consumer_norms[:,None,None]).reshape(256,64)
    u,s,vh=torch.linalg.svd(balanced,full_matrices=False)
    p['readers']=torch.cat([p['readers'][:512],vh[:rank]@p['readers'][512:576],p['readers'][-1:]])
    p['read_left']=p['readers']@p['left'];p['read_direction']=p['readers']@p['direction']
    p['inside_adapters']=(adapter@vh[:rank].T).reshape(2,128,rank)
    return p,float((adapter-(adapter@vh[:rank].T)@vh[:rank]).norm()/adapter.norm())

def scalar(reads,rho2,p):
    t=reads[...,512:-1]
    expanded=torch.cat([reads[...,:512],t@p['inside_adapters'][0].T,t@p['inside_adapters'][1].T,reads[...,-1:]],-1)
    return old_scalar(expanded,rho2)
