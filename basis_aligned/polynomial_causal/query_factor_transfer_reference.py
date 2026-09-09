"""Query Q1/Q2 interchange for the explicit initial-binding readout only."""
import types
import torch


def read(program,recipient,donor,subset,binding_tokens=48):
    q={name:(donor if subset&(1<<bit) else recipient)['features'][name][:,-1:]
       for bit,name in enumerate(('q1','q2'))}
    source={name:value[:,:binding_tokens] for name,value in recipient['features'].items()}
    z=program.aggregate(q,source,recipient['positions'][-1:],recipient['positions'][:binding_tokens])
    return (.5*z@program.folded.T)[:,0]


def native(model,recipient,donor,subset,binding_tokens=48):
    layer=model.layers[-1];normalized=layer.norm(donor);handles=[];original=layer.pattern
    try:
        for bit,name in enumerate(('q1','q2')):
            if subset&(1<<bit):
                module=getattr(layer,name);value=module(normalized)[:,-1]
                def replace(_module,_args,output,value=value):
                    result=output.clone();result[:,-1]=value;return result
                handles.append(module.register_forward_hook(replace))
        def pattern(_self,x):
            p=original(x).clone();p[:,:,:,binding_tokens:]=0;return p
        layer.pattern=types.MethodType(pattern,layer)
        return model.head(layer(recipient)-recipient*.5)[:,-1]
    finally:
        for handle in handles:handle.remove()
        del layer.pattern


def controls():
    from deep_model import DeepModel
    from exact_source_edit_reference import SourceEditProgram
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31908);model=DeepModel(8,16,4,['attn']*4,240,norm='rms').double().eval()
        old=torch.randint(8,(2,12));new=old.clone();new[:,-1]=(new[:,-1]+1)%8
    program=SourceEditProgram(model);checks={}
    with torch.inference_mode():
        before=model(old);a=program.prepare(old);b=program.prepare(new)
        checks['causal_binding_state_identity']=bool(torch.equal(a['x'][:,:8],b['x'][:,:8]))
        arms={i:read(program,a,b,i,8) for i in range(4)}
        for subset in range(4):
            checks['native_subset_'+str(subset)]=bool(torch.allclose(arms[subset],native(model,a['x'],b['x'],subset,8),atol=1e-9,rtol=1e-10))
            if subset:checks['live_subset_'+str(subset)]=float((arms[subset]-arms[0]).abs().max())>1e-12
        checks['both_equals_donor_read']=bool(torch.allclose(arms[3],read(program,b,b,0,8),atol=1e-9,rtol=1e-10))
        checks['own_query_identity']=bool(torch.equal(read(program,a,a,3,8),arms[0]))
        checks['read_is_not_full_donor_logits']=float((arms[3]-model(new)[:,-1]).abs().max())>1e-12
        checks['mixed_query_term_live']=float((arms[3]-arms[1]-arms[2]+arms[0]).abs().max())>1e-12
        checks['hooks_restored']=bool(torch.equal(before,model(old)))
    return {'passed':all(checks.values()),'checks':checks}
