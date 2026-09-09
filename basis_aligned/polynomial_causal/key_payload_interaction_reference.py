"""Exact additive key-write x payload-write interaction in a native final read."""
import torch
import key_direction_gain_reference as K


def interaction(program, context, key_write, payload_write):
    x=context['x'];layer=program.background.layers[-1];positions=context['positions']
    value=layer.v(payload_write*K.gains(layer,x)).reshape(len(x),x.shape[1],layer.n_head,layer.d_head)
    native={**context['features'],'v':value}
    cut={**K.ports(program,context,key_write,True,False),'v':value}
    original=program.aggregate(context['features'],native,positions,positions)
    changed=program.aggregate(context['features'],cut,positions,positions)
    return .5*((original-changed)@program.folded.T)


def native_arms(model, base, key_write, payload_write):
    layer=model.layers[-1];out={}
    for value_cut in (False,True):
        replacement=layer.v((base-payload_write)*K.gains(layer,base));handle=None
        if value_cut:
            def replace(_module,_args,_output,replacement=replacement):return replacement
            handle=layer.v.register_forward_hook(replace)
        try:
            for key_cut in (False,True):
                out[(key_cut,value_cut)]=K.native(model,base,key_write,key_cut,False)
        finally:
            if handle is not None:handle.remove()
    return out


def controls():
    from deep_model import DeepModel
    from exact_source_edit_reference import SourceEditProgram
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(43908);model=DeepModel(8,16,4,['attn']*4,32,norm='rms').double().eval()
        tok=torch.randint(8,(2,8));key=torch.zeros(2,8,16,dtype=torch.float64);payload=key.clone()
        key[:,2]=torch.randn(2,16,dtype=torch.float64)*.2;payload[:,2]=torch.randn(2,16,dtype=torch.float64)*.3
    program=SourceEditProgram(model);checks={}
    with torch.inference_mode():
        context=program.prepare(tok);before=model(tok);term=interaction(program,context,key,payload)
        arms=native_arms(model,context['x'],key,payload);oracle=arms[(False,False)]-arms[(True,False)]-arms[(False,True)]+arms[(True,True)]
        checks['native_contrast']=torch.allclose(term,oracle,atol=1e-9,rtol=1e-10)
        checks['live_overlap']=float(term.abs().max())>1e-5
        checks['zero_key']=float(interaction(program,context,key*0,payload).abs().max())<1e-12
        checks['zero_payload']=float(interaction(program,context,key,payload*0).abs().max())<1e-12
        checks['disjoint_support']=float(interaction(program,context,key,payload.roll(2,1)).abs().max())<1e-12
        checks['linear_payload']=torch.allclose(interaction(program,context,key,2*payload),2*term,atol=1e-12,rtol=1e-12)
        checks['native_hooks_restored']=torch.equal(before,model(tok))
    checks={k:bool(v) for k,v in checks.items()};return {'passed':all(checks.values()),'checks':checks}
