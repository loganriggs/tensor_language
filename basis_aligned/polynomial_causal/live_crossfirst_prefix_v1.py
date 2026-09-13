"""Generate the child/remainder ports from live native prefix states."""
from pathlib import Path
import importlib.util
import torch
import torch.nn.functional as F
from regional_even_routing_v1 import routing

P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('live_crossfirst_assembled',P/'extracted_circuits/crossfirst_state_executor_v1/execute.py')
assembled=importlib.util.module_from_spec(spec);spec.loader.exec_module(assembled)

@torch.no_grad()
def prepare(model,ids,weights):
    x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
    for block in model.transformer.h[:7]:x,v1=block(x,v1,x0)
    saved={}
    for index in (7,8,9):
        block=model.transformer.h[index]
        raw=block.lambdas[0]*x+block.lambdas[1]*x0
        norm=F.rms_norm(raw,(1152,));att,v1=block.attn(norm,v1)
        z=raw+att;mlp_input=F.rms_norm(z,(1152,));mlp=block.mlp(mlp_input)
        if index==7:saved['mlp7_input']=mlp_input
        if index==8:
            saved['attention8_input']=norm
            saved['rho8']=z.square().mean(-1).double()+torch.finfo(torch.float32).eps
        if index==9:
            rho9=(raw.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt().double()
            child=assembled.field(ids,saved['mlp7_input'],saved['attention8_input'],norm,saved['rho8'],rho9,weights)
            p=weights['routing']
            parent=(routing(norm,p,1)@(norm.double()@p['current_value_reader'])[...,None])[...,0]
            return dict(x0=x0,v1=v1,raw9=raw,att9=att,z9=z,m9=mlp,h9=z+mlp,
                        child=child[...,None],remainder=(parent-child)[...,None])
        x=z+mlp
