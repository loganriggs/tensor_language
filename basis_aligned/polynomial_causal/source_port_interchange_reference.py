"""Source K1/K2/V interchange with every query and residual kept native."""
import torch

PORTS=('k1','k2','v')


def ports(program,context,delta,subset):
    changed=program.features(context['x']+delta,context['positions'])
    mixed={k:v for k,v in context['features'].items()}
    for bit,name in enumerate(PORTS):
        if subset&(1<<bit):mixed[name]=changed[name]
    return mixed


def execute(program,context,mixed):
    z=program.aggregate(context['features'],mixed,context['positions'],context['positions'])
    return .5*program.background.head(context['x'])+.5*z@program.folded.T


def native(model,base,delta,subset):
    layer=model.layers[-1];normalized=layer.norm(base+delta);handles=[]
    try:
        for bit,name in enumerate(PORTS):
            if subset&(1<<bit):
                module=getattr(layer,name);replacement=module(normalized)
                def replace(_module,_args,_output,replacement=replacement):return replacement
                handles.append(module.register_forward_hook(replace))
        return model.head(layer(base))
    finally:
        for handle in handles:handle.remove()


def controls():
    from deep_model import DeepModel
    from exact_source_edit_reference import SourceEditProgram
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(26908);model=DeepModel(8,16,4,['attn']*4,240,norm='rms').double().eval()
        tokens=torch.randint(8,(2,12));delta=torch.zeros(2,12,16,dtype=torch.float64)
        delta[:,[2,5]]=torch.randn(2,2,16,dtype=torch.float64)*.3
    program=SourceEditProgram(model);checks={}
    with torch.inference_mode():
        before=model(tokens);context=program.prepare(tokens)
        for subset in range(8):
            candidate=execute(program,context,ports(program,context,delta,subset))
            oracle=native(model,context['x'],delta,subset)
            checks['native_ports_'+str(subset)]=bool(torch.allclose(candidate,oracle,atol=1e-9,rtol=1e-10))
            if subset:checks['live_ports_'+str(subset)]=float((candidate-before).abs().max())>1e-12
            else:checks['empty_identity']=bool(torch.allclose(candidate,before,atol=1e-9,rtol=1e-10))
        full=execute(program,context,ports(program,context,delta,7));physical=program.edit(context,delta)
        checks['full_query_equals_physical']=bool(torch.allclose(full[:,-1],physical[:,-1],atol=1e-9,rtol=1e-10))
        checks['binding_output_scope_negative']=float((full[:,[2,5]]-physical[:,[2,5]]).abs().max())>1e-12
        checks['restored_hooks']=bool(torch.equal(before,model(tokens)))
    return {'passed':all(checks.values()),'checks':checks}
