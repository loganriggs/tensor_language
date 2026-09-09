"""Shared endpoint messages with token-parsed consumers and explicit native gates."""
import itertools
import types
import torch


def joins(tokens,n_entities=24):
    """mask[b,t,s]: earlier edge v->w supplies endpoint w to later edge u->v."""
    b,t=tokens.shape;mask=torch.zeros(b,t,t,dtype=torch.bool,device=tokens.device)
    records=[]
    for i,row in enumerate(tokens):
        pairs=row[:2*n_entities].reshape(n_entities,2)
        assert torch.equal(pairs[:,0].sort().values,torch.arange(n_entities,device=tokens.device))
        current=[]
        for target in range(n_entities):
            for source in range(target):
                if int(pairs[target,1])==int(pairs[source,0]):
                    mask[i,2*target:2*target+2,2*source+1]=True
                    current.append({'source':2*source+1,'destinations':[2*target,2*target+1],
                                    'origin':int(pairs[target,0]),'middle':int(pairs[target,1]),'endpoint':int(pairs[source,1])})
        records.append(current)
    return mask,records


def dictionary(model,n_entities=24):
    layer=model.layers[2];h=1;d=layer.d_head
    assert layer.scale==.5 and layer.residual=='lerp' and layer.v.bias is None and layer.o.bias is None
    return ((model.embed.weight[:n_entities]@layer.v.weight[h*d:(h+1)*d].T)
            @layer.o.weight[:,h*d:(h+1)*d].T)*.125


def inputs(model,tokens):
    e=model.embed(tokens);x=model.layers[1](model.layers[0](e));layer=model.layers[2]
    eps=torch.finfo(x.dtype).eps if layer.norm.eps is None else layer.norm.eps
    return e,x,torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)


def messages(model,tokens,mask,endpoint_map=None,n_entities=24,table=None):
    _,x,gain=inputs(model,tokens);p=model.layers[2].pattern(x)[:,1]
    if table is None:table=dictionary(model,n_entities)
    entities=tokens.clamp(max=n_entities-1)
    if endpoint_map is not None:entities=endpoint_map[entities]
    return torch.einsum('bts,bsd->btd',p*mask*gain.transpose(1,2),table[entities])


def native_messages(model,tokens,mask,endpoint_map=None,n_entities=24):
    e,x,gain=inputs(model,tokens);layer=model.layers[2];original=layer.pattern
    if endpoint_map is not None:
        mapped=tokens.clone();mapped[:,:2*n_entities]=endpoint_map[mapped[:,:2*n_entities]]
        e=model.embed(mapped)
    def pattern(_self,value):
        raw=original(value);selected=torch.zeros_like(raw);selected[:,1]=raw[:,1]*mask
        return selected
    def value_input(_module,_args):return (e*.25*gain,)
    layer.pattern=types.MethodType(pattern,layer);handle=layer.v.register_forward_pre_hook(value_input)
    try:return layer(x)-x*.5
    finally:handle.remove();del layer.pattern


def controls():
    from deep_model import DeepModel
    checks={};count=0;reuse=False
    for values in itertools.product(range(3),repeat=3):
        for order in itertools.permutations(range(3)):
            pairs=torch.tensor([[k,values[k]] for k in order]);tok=torch.cat((pairs.flatten(),torch.tensor([3,0,4])))[None]
            mask,records=joins(tok,3);expected=set()
            for source,key in enumerate(order):
                for target,origin in enumerate(order):
                    if source<target and values[origin]==key:expected.add((2*target,2*source+1));expected.add((2*target+1,2*source+1))
            assert set(map(tuple,mask[0].nonzero().tolist()))==expected
            for r in records[0]:assert values[values[r['origin']]]==r['endpoint']
            reuse |= len(records[0])>len({r['source'] for r in records[0]})
            count+=1
    checks['exhaustive_162_function_orders']=count==162;checks['shared_endpoint_multiple_consumers']=reuse
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(24908);model=DeepModel(7,16,4,['attn']*4,240,norm='rms').double().eval()
    # Nonbijective graph: the same early 2->0 endpoint supplies two later joins.
    tok=torch.tensor([[2,0,0,2,1,2,3,0,4]]);mask,_=joins(tok,3)
    with torch.inference_mode():
        before=model(tok)
        for mapping in (None,torch.tensor([1,2,0])):
            a=messages(model,tok,mask,mapping,3);b=native_messages(model,tok,mask,mapping,3)
            checks['native_message_'+str(mapping is not None)]=bool(torch.allclose(a,b,atol=1e-9,rtol=1e-10))
        identity=messages(model,tok,mask,torch.arange(3),3);own=messages(model,tok,mask,n_entities=3)
        changed=messages(model,tok,mask,torch.tensor([1,2,0]),3)
        checks['identity']=bool(torch.equal(identity,own));checks['live_edit']=float((changed-own).abs().max())>1e-12
        checks['hooks_restored']=bool(torch.equal(before,model(tok)))
    return {'passed':all(checks.values()),'checks':checks,'exhaustive_function_orders':count}
