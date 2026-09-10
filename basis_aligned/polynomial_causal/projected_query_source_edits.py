"""Exact-real source-gain query algebra; FP64 fixtures, no native extraction claim."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from embedding_query_router import direct_coefficient


def source_coefficients(lambdas):
    """Pre-attention L: embedding coefficient and coefficients of writes at j<L."""
    return direct_coefficient(lambdas), [
        __import__('math').prod(float(row[0]) for row in lambdas[j+1:])
        for j in range(len(lambdas)-1)]


def prepare(sources, readers, head_width):
    projected=(sources@readers.T).unflatten(-1,(-1,head_width))
    gram=sources@sources.transpose(-1,-2)
    return projected,gram


def queries(projected,gram,gains,residual_width,eps_residual,eps_head):
    p=torch.einsum('bsfh,bs->bfh',projected,gains)
    rho=torch.einsum('bs,bst,bt->b',gains,gram,gains)
    # Gram cancellation can make a floating-point realization invalid. Do not
    # silently clip a negative computed squared norm and call it exact execution.
    assert bool((rho>=0).all())
    denominator=(p.square().mean(-1,keepdim=True)+eps_head*(rho[:,None,None]/residual_width+eps_residual)).sqrt()
    return p/denominator


def controls():
    torch.set_num_threads(2);g=torch.Generator().manual_seed(60923);dtype=torch.float64
    sources=torch.randn(3,5,12,generator=g,dtype=dtype)
    readers=.1*torch.randn(8,12,generator=g,dtype=dtype);projected,gram=prepare(sources,readers,4)
    masks=torch.tensor([[1,1,1,1,1],[0,1,1,1,1],[1,0,1,1,1],[0,0,1,1,1],[0,0,0,0,0],[1,-.5,.3,1,0]],dtype=dtype)
    errors=[];wrong=[]
    for mask in masks:
        gains=mask.expand(3,-1);u=(sources*gains[:,:,None]).sum(1)
        native=F.rms_norm(F.linear(F.rms_norm(u,(12,),eps=.02),readers).unflatten(-1,(2,4)),(4,),eps=.03)
        compiled=queries(projected,gram,gains,12,.02,.03)
        errors.append(float((compiled-native).abs().max()))
        diag=torch.diag_embed(torch.diagonal(gram,dim1=-2,dim2=-1))
        wrong.append(float((queries(projected,diag,gains,12,.02,.03)-native).norm()))
    # Exact scalar recurrence check includes separate attention and MLP writes.
    lam=[(2.,.5),(.25,-.1),(.7,.3)];alpha,coeff=source_coefficients(lam)
    e=torch.tensor([2.,-1.],dtype=dtype);a=[torch.tensor([1.,2.],dtype=dtype),torch.tensor([3.,1.],dtype=dtype)];m=[v.flip(0) for v in a]
    r=e
    for i in range(2):r=lam[i][0]*r+lam[i][1]*e+a[i]+m[i]
    direct=lam[2][0]*r+lam[2][1]*e;expanded=alpha*e+sum(c*(av+mv) for c,av,mv in zip(coeff,a,m))
    sequential=torch.ones(3,5,dtype=dtype);sequential[:,0]=0;sequential[:,1]=0
    joint=masks[3].expand(3,-1)
    checks={'source_lineage_closure':bool(torch.allclose(direct,expanded,atol=1e-12,rtol=1e-12)),
        'all_single_joint_signed_and_zero_gains':max(errors)<=1e-12,
        'zero_source_query_is_zero':bool(torch.equal(queries(projected,gram,torch.zeros(3,5,dtype=dtype),12,.02,.03),torch.zeros(3,2,4,dtype=dtype))),
        'joint_source_deletion_orders_agree':bool(torch.equal(queries(projected,gram,sequential,12,.02,.03),queries(projected,gram,joint,12,.02,.03))),
        'cross_source_norm_terms_are_live':max(wrong)>.01}
    return {'passed':all(checks.values()),'checks':checks,'max_query_replay_error':max(errors),'discard_cross_gram_max_error':max(wrong),'control_design_note':'Initial unscaled reader gave cross-Gram effect .007922, below the fixed .01 liveness bar. Scaled the planted reader by .1 to exercise norm coupling; no native source-bank run or scientific bar changed.'}


if __name__=='__main__':
    import hashlib
    result=controls();assert result['passed']
    path=Path(__file__).with_name('BILIN18_L9_EMBEDDING_QUERY_V1_RESULT.json')
    native=json.loads(path.read_text());assert native['predictions']['pred_a_instrument']
    alpha,coeff=source_coefficients(native['lambda_coefficients'])
    assert alpha==native['alpha_direct_embedding']
    result.update({'native_source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'native_direct_embedding_coefficient':alpha,
        'native_attention_and_mlp_coefficients_by_layer':coeff,'source_count':19,'projected_channels':512,
        'intervention_state_scalars_per_endpoint':19*512+19*20//2,
        'implemented_dense_gram_scalars':19*19,
        'model_forwards':0,'scope':'Native coefficient audit plus planted FP64 query-source algebra. No native source bank captured yet, no circuit identification, no native weight reduction; Gram-based norm may suffer cancellation.'})
    with Path(__file__).with_name('PROJECTED_QUERY_SOURCE_EDITS_V1_CONTROLS.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
