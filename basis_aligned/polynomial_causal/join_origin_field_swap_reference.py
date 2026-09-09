"""Origin-value interchange with recipient path gates/gains and native complement."""
import types
import torch
from join_contribution_context_reference import populations as two_chain_populations


def populations():
    out={}
    for pop,worlds in two_chain_populations(seeds=(22909,22910)).items():
        chosen=[]
        for w in worlds[::2]:
            tok=w['recipient'].clone();left=tok[:,22:24].clone();tok[:,22:24]=tok[:,38:40];tok[:,38:40]=left
            converted={k:v for k,v in w.items() if k!='donor'}
            chosen.append({**converted,'recipient':tok,'heads':{0:2,1:2}})
        out[pop]=chosen
    return out


def state(model,tokens):
    x1=model.layers[0](model.embed(tokens));l1,l2=model.layers[1:3];x2=l1(x1)
    p1=l1.pattern(x1);v1=l1.v(l1.norm(x1)).reshape(len(tokens),tokens.shape[1],l1.n_head,l1.d_head)
    p2=l2.pattern(x2);eps=torch.finfo(x2.dtype).eps if l2.norm.eps is None else l2.norm.eps
    gain=torch.rsqrt(x2.square().mean(-1,keepdim=True)+eps)
    return x1,p1,v1,p2,gain


def send(model,y,p2,gain,mask):
    l2=model.layers[2]
    v=l2.v(y*.5*gain).reshape(len(y),y.shape[1],l2.n_head,l2.d_head)
    z=torch.einsum('bts,bsd->btd',p2[:,2]*mask,v[:,:,2])
    return (z@l2.o.weight[:,2*l2.d_head:3*l2.d_head].T)*.5


def paths(model,tokens,masks,swap=False):
    x1,p1,v1,p2,gain=state(model,tokens);l1=model.layers[1];out={}
    keys={j:int(torch.where(mask.any(0))[0][0]) for j,mask in masks.items()}
    for j,mask in masks.items():
        key=keys[j];donor=keys[1-j] if swap else key
        z=(p1[:,:,key+1,key,None]*v1[:,donor]).flatten(-2)
        y=torch.zeros_like(x1);y[:,key+1]=l1.o(z)
        out[j]=send(model,y,p2,gain,mask)
    return out


def native_hook_paths(model,tokens,masks,swap=False):
    x1,_,_,p2,gain=state(model,tokens);l1=model.layers[1];original=l1.pattern;captured={}
    keys={j:int(torch.where(mask.any(0))[0][0]) for j,mask in masks.items()}
    local=torch.zeros(tokens.shape[1],tokens.shape[1],device=tokens.device,dtype=torch.bool)
    for key in keys.values():local[key+1,key]=True
    def pattern(_self,x):return original(x)*local
    def values(_module,_args,output):
        if not swap:return output
        changed=output.clone()
        for j,key in keys.items():changed[:,key]=output[:,keys[1-j]]
        return changed
    def capture(_module,_args,output):captured['y']=output
    l1.pattern=types.MethodType(pattern,l1)
    handles=[l1.v.register_forward_hook(values),l1.o.register_forward_hook(capture)]
    try:l1(x1)
    finally:
        for h in handles:h.remove()
        del l1.pattern
    return {j:send(model,captured['y'],p2,gain,mask) for j,mask in masks.items()}


def controls():
    from deep_model import DeepModel
    checks={};data=populations()
    for pop,worlds in data.items():
        valid=True
        for w in worlds:
            tok=w['recipient'];f=torch.empty(24,dtype=torch.long).scatter_(0,tok[0,:48:2],tok[0,1:48:2])
            valid &= bool((tok[0,:48:2].sort().values==torch.arange(24)).all() and (tok[0,1:48:2].sort().values==torch.arange(24)).all())
            for row,answer in zip(tok,w['answers']):
                e=row[49]
                for _ in range(int(row[50])-25):e=f[e]
                valid &= bool(e==answer)
        checks[pop+'_valid_map_answers']=valid
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(22908);model=DeepModel(29,16,4,['attn']*4,240,norm='rms').double().eval()
    w=data['iid'][0];tokens=w['recipient'];masks=w['masks']
    with torch.inference_mode():
        base=model(tokens)
        for swap in (False,True):
            explicit=paths(model,tokens,masks,swap);native=native_hook_paths(model,tokens,masks,swap)
            checks['native_hook_correspondence_'+str(swap)]=all(torch.allclose(v,native[j],atol=1e-9,rtol=1e-10) for j,v in explicit.items())
        own=paths(model,tokens,masks);changed=paths(model,tokens,masks,True)
        checks['swap_live']=all(float((changed[j]-own[j]).abs().max())>1e-12 for j in own)
        checks['hooks_restored']=bool(torch.equal(model(tokens),base))
        checks['disjoint_destinations']=not bool((masks[0].any(-1)&masks[1].any(-1)).any())
    return {'passed':all(checks.values()),'checks':checks}
