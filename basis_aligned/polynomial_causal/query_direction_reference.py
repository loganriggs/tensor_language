"""Final-query Q1/Q2 input at the original RMS gain; sources and skip unchanged."""
from contextlib import contextmanager
import torch
import key_direction_gain_reference as K


def context(program, original, raw_query):
    x=original['x'];layer=program.background.layers[-1];pos=original['positions'][-1:]
    assert raw_query.shape==x[:,-1].shape
    normalized=raw_query[:,None]*K.gains(layer,x)[:,-1:];features={**original['features']}
    for name in ('q1','q2'):
        z=getattr(layer,name)(normalized).reshape(len(x),1,layer.n_head,layer.d_head)
        a,b=z.chunk(2,-1)
        rotated=z*layer.rotary.cos_cached[:,pos]+torch.cat((-b,a),-1)*layer.rotary.sin_cached[:,pos]
        features[name]=features[name].clone();features[name][:,-1:]=rotated
    return {**original,'features':features}


@contextmanager
def native_hooks(model, base, raw_query):
    layer=model.layers[-1];normalized=layer.norm(base).clone()
    normalized[:,-1]=raw_query*K.gains(layer,base)[:,-1];handles=[]
    try:
        for name in ('q1','q2'):
            module=getattr(layer,name);replacement=module(normalized)
            def replace(_module,_args,_output,replacement=replacement):return replacement
            handles.append(module.register_forward_hook(replace))
        yield
    finally:
        for h in handles:h.remove()


def controls():
    from deep_model import DeepModel
    from exact_source_edit_reference import SourceEditProgram
    import key_payload_interaction_reference as I
    import source_port_interchange_reference as P
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(44908);model=DeepModel(8,16,4,['attn']*4,32,norm='rms').double().eval()
        tok=torch.randint(8,(2,8));key=torch.zeros(2,8,16,dtype=torch.float64);payload=key.clone()
        key[:,2]=torch.randn(2,16,dtype=torch.float64)*.2;payload[:,2]=torch.randn(2,16,dtype=torch.float64)*.3
    program=SourceEditProgram(model);checks={};terms={}
    with torch.inference_mode():
        base=program.prepare(tok);before=model(tok)
        for scale in (0.,.5,1.):
            raw=base['x'][:,-1]*scale;changed=context(program,base,raw)
            actual=P.execute(program,changed,base['features'])
            with native_hooks(model,base['x'],raw):arms=I.native_arms(model,base['x'],key,payload)
            checks['native_Q_'+str(scale)]=torch.allclose(actual,arms[(False,False)],atol=1e-9,rtol=1e-10)
            terms[scale]=I.interaction(program,changed,key,payload)[:,-1]
            wanted=arms[(False,False)]+arms[(True,True)]-arms[(True,False)]-arms[(False,True)]
            checks['native_interaction_'+str(scale)]=torch.allclose(terms[scale],wanted[:,-1],atol=1e-9,rtol=1e-10)
        checks['zero_query_zero_route']=float(terms[0.].abs().max())<1e-12
        checks['planted_mixed_query_live']=float((terms[1.]-2*terms[.5]).abs().max())>1e-6
        checks['query_quadratic_scaling']=torch.allclose(terms[.5],terms[1.]*.25,atol=1e-12,rtol=1e-12)
        checks['hooks_restored']=torch.equal(before,model(tok))
    checks={k:bool(v) for k,v in checks.items()};return {'passed':all(checks.values()),'checks':checks}
