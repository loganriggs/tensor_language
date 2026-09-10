"""Direct residual-lineage MLP change into a normalized downstream value map."""
import torch


def folded_mlp_change(n_base,n_donor,left,right,down,value_reader):
    """Fold the value reader into Down; exact midpoint secant of tied-input MLP."""
    middle=(n_base+n_donor)/2;delta=n_donor-n_base
    feature_delta=(middle@left.T)*(delta@right.T)+(delta@left.T)*(middle@right.T)
    return feature_delta@(value_reader@down).T


def value_change(base_residual,mlp_delta,transport,value_reader,eps):
    """Query/key and all other source writes fixed; this is only the value edge."""
    u=base_residual;change=transport*mlp_delta
    s0=(u.square().mean(-1,keepdim=True)+eps).sqrt()
    s1=((u+change).square().mean(-1,keepdim=True)+eps).sqrt()
    direct_content=(change@value_reader.T)/s1
    norm_change=(u@value_reader.T)*(1/s1-1/s0)
    return {'total':direct_content+norm_change,'direct_content':direct_content,'normalization':norm_change,'new_scale':s1}


def controls():
    torch.set_num_threads(2);g=torch.Generator().manual_seed(60925)
    rand=lambda *s:torch.randn(*s,generator=g,dtype=torch.float64)
    nb=rand(5,6);nd=rand(5,6);left=rand(9,6);right=rand(9,6);down=rand(8,9);reader=rand(3,8);bias=rand(8)
    mlp=lambda n:((n@left.T)*(n@right.T))@down.T+bias
    delta=mlp(nd)-mlp(nb);folded=folded_mlp_change(nb,nd,left,right,down,reader)
    u=rand(5,8);transport=-.3;eps=.02;parts=value_change(u,delta,transport,reader,eps)
    norm=lambda v:v/(v.square().mean(-1,keepdim=True)+eps).sqrt()
    direct=(norm(u+transport*delta)-norm(u))@reader.T
    error=float((parts['total']-direct).abs().max());fold_error=float((folded-delta@reader.T).abs().max())
    folded_content=transport*folded/parts['new_scale']
    content_error=float((folded_content-parts['direct_content']).abs().max())
    norm_live=float(parts['normalization'].norm())
    zero=value_change(u,torch.zeros_like(delta),transport,reader,eps)['total']
    checks={'normalized_value_delta':error<1e-11,'folded_bilinear_secant':fold_error<1e-11,'folded_direct_content':content_error<1e-11,'zero_delta_identity':bool((zero==0).all()),'normalization_term_live':norm_live>.1}
    return {'passed':all(checks.values()),'checks':checks,'value_error':error,'fold_error':fold_error,'content_error':content_error,'normalization_term_norm':norm_live,'model_forwards':0,'scope':'Planted direct residual-lineage source edge only. Native MLP4 intervention also changes later attention/MLP writes; those are not folded by this identity. Down bias cancels in MLP difference but remains in native residual context.'}


if __name__=='__main__':
    import json
    from pathlib import Path
    result=controls();assert result['passed']
    with Path(__file__).with_name('MLP_VALUE_LINEAGE_FOLD_V1_CONTROLS.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
