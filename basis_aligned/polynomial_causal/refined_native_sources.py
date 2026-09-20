"""Split six parent source deltas into 23 native writes, preserving their gauges."""
import torch
NAMES=['embedding']+[f'{kind}{layer}' for layer in range(8) for kind in ['attention','mlp']]+['mlp8','mlp10','attention8','attention9','mlp9','attention10']
PARENTS=[0]+[1]*8+[2]*8+[3,4]+[5]*4

def expand(amplitudes):
    return amplitudes[...,PARENTS]

def collapse(values):
    # Channel dimension is the penultimate dimension, e.g. [B,23,D].
    return torch.stack([values[..., [i for i,p in enumerate(PARENTS) if p==parent],:].sum(-2) for parent in range(6)],dim=-2)

def refine(parents,explicit):
    result=explicit.clone();corrections=[]
    for parent in range(6):
        ids=[i for i,p in enumerate(PARENTS) if p==parent]
        remaining=parents[:,parent]-result[:,ids[:-1]].sum(1) if len(ids)>1 else parents[:,parent]
        correction=remaining-result[:,ids[-1]]
        corrections.append(dict(absolute=float(correction.abs().max()),relative=float(correction.norm()/result[:,ids[-1]].norm().clamp_min(1e-30))))
        result[:,ids[-1]]=remaining
    return result,dict(collapse_error=float((collapse(result)-parents).abs().max()),corrections=corrections)

def capture_explicit(model,initial,torch,F,capture):
    """Observe actual native writes in the existing prefix, then apply its lambdas."""
    writes={};hooks=[]
    def hook(name):
        def save(module,args,output):writes[name]=(output[0] if isinstance(output,tuple) else output).detach().clone()
        return save
    try:
        for layer in range(11):
            b=model.transformer.h[layer]
            hooks.extend([b.attn.register_forward_hook(hook(f'attention{layer}')),b.mlp.register_forward_hook(hook(f'mlp{layer}'))])
        result=capture(model,initial,torch,F)
    finally:
        for h in hooks:h.remove()
    assert len(writes)==22
    for name,value in list(writes.items()):
        layer=int(name.removeprefix('attention').removeprefix('mlp'))
        for later in range(layer+1,12):value=value*model.transformer.h[later].lambdas[0]
        writes[name]=value
    return result,writes

def build_refined(model,context,modal):
    from frozen_two_site_context import build
    import subject_number_sparse_graph_token_extraction_v1 as graph
    captures=[]
    def observed(model,initial,torch,F):
        result,writes=capture_explicit(model,initial,torch,F,graph._capture);captures.append(writes);return result
    c=build(model,context,modal,capture_fn=observed);checks=[]
    for index,role in enumerate(['subject','attractor']):
        pos,parents=c['source_components'][role];batch=c['batch'];base,edited=captures[0],captures[index+1]
        explicit=torch.stack([parents[:,0] if name=='embedding' else (edited[name]-base[name])[batch,pos].float().double() for name in NAMES],dim=1)
        children,check=refine(parents,explicit);checks.append(dict(role=role,**check));c['source_components'][role]=(pos,children)
    c['refinement_checks']=checks
    return c
