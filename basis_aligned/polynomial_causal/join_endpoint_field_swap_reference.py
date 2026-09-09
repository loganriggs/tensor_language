"""Explicit forward endpoint embedding paths under recipient gates and gains."""
import types
import torch
from join_contribution_context_reference import populations as two_chain_populations


def populations():
    out={}
    for pop,worlds in two_chain_populations(seeds=(23909,23910)).items():
        out[pop]=[]
        for w in worlds[::2]:
            tok=w['recipient'].clone();left=tok[:,26:28].clone()
            tok[:,26:28]=tok[:,42:44];tok[:,42:44]=left
            out[pop].append({**{k:v for k,v in w.items() if k!='donor'},'recipient':tok,'heads':{0:1,1:1}})
    return out


def state(model,tokens):
    e=model.embed(tokens);x=model.layers[1](model.layers[0](e));layer=model.layers[2]
    eps=torch.finfo(x.dtype).eps if layer.norm.eps is None else layer.norm.eps
    return e,x,torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)


def paths(model,tokens,masks,swap=False):
    e,x,gain=state(model,tokens);layer=model.layers[2];p=layer.pattern(x);out={}
    positions={j:int(torch.where(mask.any(0))[0][-1]) for j,mask in masks.items()}
    for j,mask in masks.items():
        source=positions[j];donor=positions[1-j] if swap else source
        v=layer.v(e[:,donor]*.25*gain[:,source]).reshape(len(tokens),layer.n_head,layer.d_head)
        z=p[:,1,:,source,None]*mask[:,source,None]*v[:,None,1]
        out[j]=(z@layer.o.weight[:,layer.d_head:2*layer.d_head].T)*.5
    return out


def native_hook_paths(model,tokens,masks,swap=False):
    e,x,gain=state(model,tokens);layer=model.layers[2];original=layer.pattern;out={}
    positions={j:int(torch.where(mask.any(0))[0][-1]) for j,mask in masks.items()}
    for j,mask in masks.items():
        source=positions[j];donor=positions[1-j] if swap else source
        replacement=torch.zeros_like(x);replacement[:,source]=e[:,donor]*.25*gain[:,source]
        def pattern(_self,inputs):
            selected=torch.zeros_like(original(inputs));selected[:,1]=original(inputs)[:,1]*mask
            return selected
        def replace(_module,_args):return (replacement,)
        layer.pattern=types.MethodType(pattern,layer);handle=layer.v.register_forward_pre_hook(replace)
        try:out[j]=layer(x)-x*.5
        finally:handle.remove();del layer.pattern
    return out


def controls():
    from deep_model import DeepModel
    data=populations();checks={}
    for pop,worlds in data.items():
        valid=True
        for w in worlds:
            tok=w['recipient'];f=torch.empty(24,dtype=torch.long).scatter_(0,tok[0,:48:2],tok[0,1:48:2])
            valid &= bool((tok[0,:48:2].sort().values==torch.arange(24)).all() and (tok[0,1:48:2].sort().values==torch.arange(24)).all())
            for row,answer in zip(tok,w['answers']):
                a=row[49]
                for _ in range(int(row[50])-25):a=f[a]
                valid &= bool(a==answer)
            for j,mask in w['masks'].items():
                source=int(torch.where(mask.any(0))[0][-1]);valid &= bool(tok[0,source]==w['answers'][4*j+3])
                valid &= not bool(mask.triu(1).any())
        checks[pop+'_maps_answers_endpoints_causality']=valid
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23908);model=DeepModel(29,16,4,['attn']*4,240,norm='rms').double().eval()
    w=data['iid'][0];tok=w['recipient'];masks=w['masks']
    with torch.inference_mode():
        baseline=model(tok)
        for swap in (False,True):
            direct=paths(model,tok,masks,swap);native=native_hook_paths(model,tok,masks,swap)
            checks['native_path_'+str(swap)]=all(torch.allclose(direct[j],native[j],atol=1e-9,rtol=1e-10) for j in masks)
        own=paths(model,tok,masks);changed=paths(model,tok,masks,True)
        checks['swaps_live']=all(float((changed[j]-own[j]).abs().max())>1e-12 for j in masks)
        checks['hooks_restored']=bool(torch.equal(baseline,model(tok)))
        checks['destinations_disjoint']=not bool((masks[0].any(-1)&masks[1].any(-1)).any())
    return {'passed':all(checks.values()),'checks':checks}
