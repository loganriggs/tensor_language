"""Whole-model linear payload execution conditioned on all native P/RMS gates."""
import torch
import frozen_payload_lineage_reference as L


def cache(program,tokens):
    out=L.prepare(program.background,tokens);x=out['native_prefix'];last=program.background.layers[-1]
    eps=torch.finfo(x.dtype).eps if last.norm.eps is None else last.norm.eps
    out['final_gate']={'pattern':last.pattern(x),'gain':torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)}
    return out


def execute(program,embedding,gates):
    z=L.propagate(program.background,embedding,gates);last=program.background.layers[-1];g=gates['final_gate']
    v=last.v(z*g['gain']).reshape(len(z),z.shape[1],last.n_head,last.d_head)
    summed=torch.einsum('bhts,bshd->bthd',g['pattern'],v).flatten(-2)
    return .5*program.background.head(z)+.5*summed@program.folded.T


def rename(tokens,permutation,n_entities=24):
    out=tokens.clone();selected=out<n_entities;out[selected]=permutation[out[selected]]
    return out


def factorial(program,tokens,renamed):
    old=cache(program,tokens);new=cache(program,renamed)
    return {'native':execute(program,old['embedding'],old),
            'payload_only':execute(program,new['embedding'],old),
            'gates_only':execute(program,old['embedding'],new),
            'renamed_native':execute(program,new['embedding'],new)},(old,new)


def controls():
    from deep_model import DeepModel
    from exact_source_edit_reference import SourceEditProgram
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(27908);model=DeepModel(8,16,4,['attn']*4,240,norm='rms').double().eval()
        tokens=torch.randint(8,(2,12));permutation=torch.tensor([1,2,3,0])
    program=SourceEditProgram(model);renamed=rename(tokens,permutation,4);checks={}
    with torch.inference_mode():
        outputs,(old,new)=factorial(program,tokens,renamed)
        checks['old_native_closure']=bool(torch.allclose(outputs['native'],model(tokens),atol=1e-9,rtol=1e-10))
        checks['new_native_closure']=bool(torch.allclose(outputs['renamed_native'],model(renamed),atol=1e-9,rtol=1e-10))
        delta_e=new['embedding']-old['embedding']
        payload=execute(program,delta_e,old)
        gate=execute(program,old['embedding'],new)-execute(program,old['embedding'],old)
        interaction=execute(program,delta_e,new)-execute(program,delta_e,old)
        checks['bilinear_factorial_closure']=bool(torch.allclose(outputs['native']+payload+gate+interaction,outputs['renamed_native'],atol=1e-9,rtol=1e-10))
        checks['payload_live']=float(payload.abs().max())>1e-12;checks['gate_live']=float(gate.abs().max())>1e-12
        checks['interaction_live']=float(interaction.abs().max())>1e-12
        checks['special_tokens_unchanged']=bool(torch.equal(tokens[tokens>=4],renamed[tokens>=4]))
        checks['identity_renaming']=bool(torch.equal(tokens,rename(tokens,torch.arange(4),4)))
    # Pure graph conjugation: a permutation of labels preserves every power.
    f=torch.tensor([1,2,3,0]);keys=torch.arange(4);fp=torch.empty_like(f).scatter_(0,permutation[keys],permutation[f])
    valid=True
    for q in range(4):
        a=q;b=int(permutation[q])
        for _ in range(4):valid &= b==int(permutation[a]);a=int(f[a]);b=int(fp[b])
    checks['graph_power_conjugation']=valid
    return {'passed':all(checks.values()),'checks':checks}
