"""Explicit L1 previous-key path feeding the L2 backward join's value input."""
import torch
from join_contribution_context_reference import populations, contributions, intervene
from join_value_producer_reference import producer_writes


def origin_write(model, tokens, masks, heads, all_sources=False):
    e = model.embed(tokens); x1 = model.layers[0](e)
    l1, l2 = model.layers[1:3]; x2 = l1(x1)
    p1 = l1.pattern(x1)
    v1 = l1.v(l1.norm(x1)).reshape(len(tokens),tokens.shape[1],l1.n_head,l1.d_head)
    p2 = l2.pattern(x2)
    eps = torch.finfo(x2.dtype).eps if l2.norm.eps is None else l2.norm.eps
    gain = torch.rsqrt(x2.square().mean(-1,keepdim=True)+eps)
    assert l1.scale == .5 and l2.scale == .5 and l1.residual == l2.residual == 'lerp'
    out = {}
    for j, mask in masks.items():
        if heads[j] != 2: continue
        source_positions = torch.where(mask.any(0))[0]
        assert len(source_positions) == 2 and int(source_positions[0])%2 == 0
        key, value = source_positions.tolist()
        local = torch.ones_like(mask) if all_sources else torch.zeros_like(mask)
        if not all_sources: local[value,key] = True
        y = l1.o(torch.einsum('bhts,bshd->bthd',p1*local,v1).flatten(-2))
        v2 = l2.v(y*.5*gain).reshape(len(tokens),tokens.shape[1],l2.n_head,l2.d_head)
        z = torch.einsum('bts,bsd->btd',p2[:,2]*mask,v2[:,:,2])
        out[j] = (z @ l2.o.weight[:,2*l2.d_head:3*l2.d_head].T)*.5
    return out


def controls():
    from deep_model import DeepModel
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19908)
        model = DeepModel(29,16,4,['attn']*4,240,norm='rms').double().eval()
    w = populations()['iid'][0]; tok=w['recipient']; masks,heads=w['masks'],w['heads']
    checks={}
    with torch.inference_mode():
        full=origin_write(model,tok,masks,heads,all_sources=True)
        parts,_=producer_writes(model,tok,masks,heads)
        checks['all_source_Y1_identity']=all(torch.allclose(v,parts[j]['Y1'],atol=1e-9,rtol=1e-9) for j,v in full.items())
        local=origin_write(model,tok,masks,heads)
        checks['local_path_live']=all(float(v.abs().max())>1e-12 for v in local.values())
        own=contributions(model,tok,masks,heads); back=next(iter(local)); forward=1-back
        native=model(tok)
        with intervene(model,masks,heads,(back,),own): restored=model(tok)
        with intervene(model,masks,heads,(forward,)): target=model(tok)
        with intervene(model,masks,heads,(back,forward),{back:own[back],forward:torch.zeros_like(own[forward])}): joint=model(tok)
        checks['single_restore']=bool(torch.allclose(native,restored,atol=1e-9,rtol=1e-9))
        checks['conditional_joint_restore']=bool(torch.allclose(target,joint,atol=1e-9,rtol=1e-9))
        changed=tok.clone(); changed[:,44:48]=(changed[:,44:48]+1)%24
        after=origin_write(model,changed,masks,heads)
        checks['future_binding_invariant']=all(torch.equal(v,after[j]) for j,v in local.items())
    return {'passed':all(checks.values()),'checks':checks}
