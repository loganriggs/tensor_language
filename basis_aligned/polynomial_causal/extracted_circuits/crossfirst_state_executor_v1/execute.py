"""All-source Q7/head8.2-first/MLP8/head9 interaction from explicit state ports.

No model forward hooks or cached intermediate tables are used by this executor.
It is conditional on native normalized prefix states and two RMS scalars.
Native replay of this assembled version is required before deployment.
"""
from pathlib import Path
import importlib.util
import torch
import torch.nn.functional as F

P = Path(__file__).resolve().parents[2]

def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_token = _module('crossfirst_token_inputs', P/'extracted_circuits/first_token_value_path_v1/execute.py')
_routing = _module('crossfirst_even_routing', P/'regional_even_routing_v1.py')


def load_weights(checkpoint, device='cpu'):
    """Read only declared checkpoint tensors; caller supplies checkpoint mapping.

    The embedding and MLP7 left/right matrices remain external native weights,
    explicitly charged. No copies of those matrices are hidden in a small file.
    """
    parent = torch.load(P/'MLP7_PHI_READERS_FOLD_V1_PROGRAM.pt', weights_only=True)
    token = torch.load(P/'extracted_circuits/first_token_value_path_v1/program.pt', weights_only=True)
    routing = torch.load(P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1/program.pt', weights_only=True)
    return {
        'embedding': checkpoint['transformer.wte.weight'].to(device),
        'left7': checkpoint['transformer.h.7.mlp.Left.weight'].to(device),
        'right7': checkpoint['transformer.h.7.mlp.Right.weight'].to(device),
        'product_coefficients': parent['product_coefficients'].to(device),
        'lambda8': parent['lambda8'][0].to(device),
        'token': {k: v.to(device) for k,v in token.items()},
        'routing': {k: v.to(device) for k,v in routing.items()},
    }


def full_joint_routing8(current, weights):
    """Full product of both normalized/RoPE QK factors for head8.2.

    current is [B,T,1152] native FP32 attention8 input. Match native rounded
    rotary tables; this head8 routing is not the even-key approximation.
    """
    n = current.shape[1]
    inv = 1 / (10000 ** (torch.arange(0,128,2,dtype=torch.float32) / 128))
    angle = torch.outer(torch.arange(n,dtype=torch.float32), inv)
    co = angle.cos().bfloat16().to(current.device)
    si = angle.sin().bfloat16().to(current.device)
    def rotate(x):
        a,b = x.chunk(2,-1)
        return torch.cat((a*co+b*si,-a*si+b*co),-1)
    scores=[]
    for qn,kn in [('q1','k1'),('q2','k2')]:
        q=rotate(F.rms_norm(F.linear(current,weights[qn][0]),(128,)))
        k=rotate(F.rms_norm(F.linear(current,weights[kn][0]),(128,)))
        scores.append((q @ k.transpose(-1,-2)) / 128)
    gamma=scores[0]*scores[1]
    return gamma.masked_fill(~torch.ones(n,n,dtype=torch.bool,device=current.device).tril(),0).double()


def field(token_ids, mlp7_input, attention8_input, attention9_input,
          rms8_squared, rms9, weights):
    """Return [B,T] interaction write along the frozen head9 writer.

    Inputs:
      token_ids [B,T]; three native normalized state arrays [B,T,1152];
      rms8_squared and rms9 [B,T]. All positions/sources are included.
    No donor, cue annotation, Q7 cache, or attention-pattern cache is accepted.
    """
    x=mlp7_input.double()
    q=((x @ weights['left7'].double().T)*(x @ weights['right7'].double().T)) @ weights['product_coefficients'].T
    q=q*weights['lambda8']
    token=weights['token']
    values=_token.token_readings(token_ids,weights['embedding'],token)
    h=full_joint_routing8(attention8_input,weights['routing']) @ values
    v=2*token['head9_gain']*(q*token['eigenvalues']*h).sum(-1)/(rms8_squared.double()*rms9.double())
    return (_routing.routing(attention9_input,weights['routing'],1) @ v[...,None])[...,0]


def residual_write(*args, **kwargs):
    weights=kwargs.get('weights',args[-1] if args else None)
    return field(*args,**kwargs)[...,None]*weights['routing']['writers'][1]
