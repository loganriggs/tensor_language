"""Literal raw embedding and compiled endpoint fields at postL2 positions."""
import torch
from forward_endpoint_random_layout_reference import populations as random_worlds


def populations(seeds=(25909,25910)):
    out=random_worlds(seeds=seeds)
    for worlds in out.values():
        for w in worlds:
            raw=w['sigma'];joined=raw[raw];w['raw_map']=raw;w['join_map']=joined
            for m in w['metadata']:
                join_consumer=m['hop']==3 and m['eligible'];raw_consumer=m['hop']>0 and not join_consumer
                m.update(join_consumer=join_consumer,raw_consumer=raw_consumer,
                         raw_desired=int(raw[m['answer']]) if raw_consumer else m['answer'],
                         join_desired=int(joined[m['answer']]) if join_consumer else m['answer'],
                         both_desired=int(joined[m['answer']]) if join_consumer else int(raw[m['answer']]) if raw_consumer else m['answer'])
    return out


def raw_field(model,tokens,mapping=None):
    mapped=tokens.clone()
    if mapping is not None:mapped[:,1:48:2]=mapping[mapped[:,1:48:2]]
    out=torch.zeros(len(tokens),tokens.shape[1],model.embed.embedding_dim,dtype=model.embed.weight.dtype,device=tokens.device)
    out[:,1:48:2]=model.embed(mapped[:,1:48:2])*.125
    return out


def prefix_identity(model,tokens):
    captured={};handles=[]
    for j in range(3):
        def capture(_module,_args,output,j=j):captured[j]=output
        handles.append(model.layers[j].o.register_forward_hook(capture))
    try:
        e=model.embed(tokens);x=e
        for layer in model.layers[:3]:x=layer(x)
    finally:
        for h in handles:h.remove()
    return x, e*.125+captured[0]*.125+captured[1]*.25+captured[2]*.5


def controls():
    from deep_model import DeepModel
    import forward_endpoint_program_reference as F
    data=populations();checks={};valid=True
    for worlds in data.values():
        for w in worlds:
            valid &= bool((w['raw_map']!=w['join_map']).all() and (w['raw_map']!=torch.arange(24)).all() and (w['join_map']!=torch.arange(24)).all())
            for m in w['metadata']:
                valid &= (int(m['raw_consumer'])+int(m['join_consumer']))==int(m['hop']>0)
    checks['distinct_maps_and_consumer_partition']=valid
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(25908);model=DeepModel(29,16,4,['attn']*4,240,norm='rms').double().eval()
    w=data['iid'][0];tok=w['tokens'][:8]
    with torch.inference_mode():
        before=model(tok);actual,expected=prefix_identity(model,tok)
        checks['prefix_producer_identity']=bool(torch.allclose(actual,expected,atol=1e-9,rtol=1e-10))
        own=raw_field(model,tok);identity=raw_field(model,tok,torch.arange(24));changed=raw_field(model,tok,w['raw_map'])
        checks['raw_identity']=bool(torch.equal(own,identity));checks['raw_live']=float((changed-own).abs().max())>1e-12
        mask=w['mask'][None].expand(len(tok),-1,-1)
        jdelta=F.messages(model,tok,mask,w['join_map'])-F.messages(model,tok,mask)
        checks['field_support_overlap']=bool(((changed-own).ne(0).any(-1)&jdelta.ne(0).any(-1)).any())
        checks['query_state_unchanged']=not bool((changed-own)[:,-3:].ne(0).any() or jdelta[:,-3:].ne(0).any())
        checks['hooks_restored']=bool(torch.equal(before,model(tok)))
    return {'passed':all(checks.values()),'checks':checks}
