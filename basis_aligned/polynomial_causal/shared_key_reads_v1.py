"""Share the 64 source reads consumed by both reflected-key factors."""
import torch
from contracted_qk_response_v1 import compile_program as old_compile,features,scalar as old_scalar

def compile_program(native,low,gain):
    p=old_compile(native,low,gain)
    basis=native['key_basis'][1].double()
    p['readers']=torch.cat([p['readers'][:512],basis.T,p['readers'][-1:]],0)
    p['read_left']=p['readers']@p['left']
    p['read_direction']=p['readers']@p['direction']
    p['inside_adapters']=torch.stack([native[k][1].double()@basis for k in ['k1','k2']])
    return p

def expand(reads,p):
    shared=reads[...,512:576]
    return torch.cat([reads[...,:512],shared@p['inside_adapters'][0].T,
                      shared@p['inside_adapters'][1].T,reads[...,-1:]],-1)

def scalar(reads,rho2,p):
    # Expand only at the consumer boundary; no working-memory reduction claimed.
    return old_scalar(expand(reads,p),rho2)
