"""Weight-only optimal common reader subspace for the two key consumers."""
import torch
from pathlib import Path
from shared_key_reads_v1 import compile_program as full_compile
from contracted_qk_response_v1 import scalar as old_scalar

def compile_program(native,low,gain,rank):
    p=full_compile(native,low,gain)
    if not 1<=rank<=64:raise ValueError('rank must lie in [1,64]')
    adapter=p['inside_adapters'].reshape(256,64)
    assert rank==48
    rotation=torch.load(Path(__file__).resolve().parent/'JOINT_KEY_PRODUCT_FIT_V1_PROGRAM.pt',weights_only=True)['basis_rotation'];vh=rotation.T
    p['readers']=torch.cat([p['readers'][:512],vh[:rank]@p['readers'][512:576],p['readers'][-1:]])
    p['read_left']=p['readers']@p['left'];p['read_direction']=p['readers']@p['direction']
    p['inside_adapters']=(adapter@vh[:rank].T).reshape(2,128,rank)
    return p,float((adapter-(adapter@vh[:rank].T)@vh[:rank]).norm()/adapter.norm())

def scalar(reads,rho2,p):
    t=reads[...,512:-1]
    expanded=torch.cat([reads[...,:512],t@p['inside_adapters'][0].T,t@p['inside_adapters'][1].T,reads[...,-1:]],-1)
    return old_scalar(expanded,rho2)
