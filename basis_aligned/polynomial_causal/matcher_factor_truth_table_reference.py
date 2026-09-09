"""Individual QK-factor truth tables on the already nominated join source pairs."""
import torch


def factors(model,tokens,mask,head):
    x=model.embed(tokens)
    for layer in model.layers[:2]:x=layer(x)
    layer=model.layers[2];h=layer.norm(x);values=[]
    for j in (1,2):
        q=layer.rotary(getattr(layer,'q'+str(j))(h).reshape(len(tokens),tokens.shape[1],layer.n_head,layer.d_head))
        k=layer.rotary(getattr(layer,'k'+str(j))(h).reshape(len(tokens),tokens.shape[1],layer.n_head,layer.d_head))
        score=torch.einsum('btd,bsd->bts',q[:,:,head],k[:,:,head])/layer.d_head
        values.append(score[mask].reshape(len(tokens),4))
    native=layer.pattern(x)[:,head][mask].reshape(len(tokens),4)
    return values,native


def truth_table(values,cases):
    rms={case:float(values[torch.tensor([c==case for c in cases],device=values.device)].square().mean().sqrt())
         for case in ('base','keys','values','both')}
    denominator=max(min(rms['base'],rms['both']),1e-8)
    ratios={case:rms[case]/denominator for case in ('keys','values')}
    return {'rms':rms,'mismatch_ratios':ratios,'individual_predicate_nominated':
        min(rms['base'],rms['both'])>1e-8 and all(v<=.10 for v in ratios.values())}


def controls():
    from deep_model import DeepModel
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(38908);model=DeepModel(8,16,4,['attn']*4,240,norm='rms').double().eval()
        tokens=torch.randint(8,(2,8));mask=torch.zeros(2,8,8,dtype=torch.bool);mask[:,6:8,2:4]=True
    checks={}
    with torch.inference_mode():
        values,native=factors(model,tokens,mask,1)
        checks['native_product']=bool(torch.allclose(values[0]*values[1],native,atol=1e-9,rtol=1e-10))
        checks['factors_live']=all(float(v.abs().max())>1e-8 for v in values)
        cases=['base','keys','values','both']
        positive=torch.tensor([[2.],[.01],[.01],[3.]],dtype=torch.float64)
        checks['predicate_positive']=truth_table(positive,cases)['individual_predicate_nominated']
        checks['constant_negative']=not truth_table(torch.ones_like(positive),cases)['individual_predicate_nominated']
        checks['dead_negative']=not truth_table(torch.zeros_like(positive),cases)['individual_predicate_nominated']
        asymmetric=positive.clone();asymmetric[1]=2.
        checks['both_mismatches_required']=not truth_table(asymmetric,cases)['individual_predicate_nominated']
    return {'passed':all(checks.values()),'checks':checks}
