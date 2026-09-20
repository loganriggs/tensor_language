"""Capture the declared native background for last-branch scalar interventions."""
import torch.nn.functional as F

def capture(model,tokens):
    blocks=model.transformer.h;cache={};x=F.rms_norm(model.transformer.wte(tokens),(tokens.new_tensor(model.config.n_embd).item(),));x0=x;v1=None
    def pre16(module,args):cache['x16']=args[0].clone()
    def post16(module,args,value):cache['m16']=value.clone()
    def postattn(module,args,value):cache['attn17']=value[0].clone()
    handles=[blocks[16].mlp.register_forward_pre_hook(pre16),blocks[16].mlp.register_forward_hook(post16),blocks[17].attn.register_forward_hook(postattn)]
    try:
        for i,block in enumerate(blocks):
            if i==17:incoming=x.clone()
            x,v1=block(x,v1,x0)
    finally:
        for handle in handles:handle.remove()
    block=blocks[17];cache['h17']=block.lambdas[0]*incoming+block.lambdas[1]*x0+cache['attn17'];cache['final']=x
    return cache
