"""Two-MLP reader contraction and a weight-only binary unembedding hierarchy.

The two-layer fold is conditional on the intervening attention output.
It retains dense native weights and exact normalization; it is not compression.
"""
import torch


def hierarchy(weights, depth=4, iterations=20):
    unit=weights/weights.norm(dim=1,keepdim=True).clamp_min(1e-30)
    labels=torch.zeros(len(weights),dtype=torch.long,device=weights.device)
    tree=[]
    for level in range(depth):
        new=labels.clone()
        for parent in range(2**level):
            ids=torch.where(labels==parent)[0];x=unit[ids]
            assert len(ids)>=2, 'Cannot split a singleton weight cluster'
            mean=x.mean(0);a=x[(x-mean).square().sum(-1).argmax()]
            b=x[(x-a).square().sum(-1).argmax()]
            for _ in range(iterations):
                assignment=(x@b>x@a).long()
                assert bool((assignment==0).any()) and bool((assignment==1).any()), 'Empty weight cluster'
                a=x[assignment==0].mean(0);b=x[assignment==1].mean(0)
                a=a/a.norm().clamp_min(1e-30);b=b/b.norm().clamp_min(1e-30)
            new[ids]=2*parent+assignment
            tree.append({'depth':level,'node':parent,'size':len(ids),
                         'children_sizes':[int((assignment==0).sum()),int((assignment==1).sum())]})
        labels=new
    means=torch.stack([weights[labels==j].mean(0) for j in range(2**depth)])
    return labels,means,tree


def compile_maps(readers, left17, right17, down17, down16, bias17):
    return {'left_from_products16':left17@down16,
            'right_from_products16':right17@down16,
            'reader_from_products16':readers@down16,
            'reader_from_products17':readers@down17,
            'reader_bias17':readers@bias17}


def folded_state(background, products16, coefficient, readers, maps,
                 left17, right17, down17, down16, bias17, eps):
    raw17=background+coefficient*(products16@down16.T)
    left=background@left17.T+coefficient*(products16@maps['left_from_products16'].T)
    right=background@right17.T+coefficient*(products16@maps['right_from_products16'].T)
    products17=left*right/(raw17.square().mean(-1,keepdim=True)+eps)
    final=raw17+products17@down17.T+bias17
    rho=(final.square().mean(-1,keepdim=True)+eps).sqrt()
    numerator=(background@readers.T+coefficient*(products16@maps['reader_from_products16'].T)
               +products17@maps['reader_from_products17'].T+maps['reader_bias17'])
    return final,numerator/rho


def controls():
    torch.set_num_threads(2);gen=torch.Generator().manual_seed(9111340)
    rand=lambda *s:torch.randn(*s,generator=gen,dtype=torch.float64)
    background,products,readers=rand(6,5),rand(6,8),rand(17,5)
    left,right,down,d16,bias=rand(9,5),rand(9,5),rand(5,9),rand(5,8),rand(5)
    eps=torch.finfo(torch.float32).eps;coefficient=-.7
    maps=compile_maps(readers,left,right,down,d16,bias)
    final,score=folded_state(background,products,coefficient,readers,maps,left,right,down,d16,bias,eps)
    r=background+coefficient*products@d16.T;n=torch.nn.functional.rms_norm(r,(5,),eps=eps)
    expected=r+((n@left.T)*(n@right.T))@down.T+bias
    truth=torch.nn.functional.rms_norm(expected,(5,),eps=eps)@readers.T
    errors={'state':float((final-expected).abs().max()),'readers':float((score-truth).abs().max())}
    # Group + token residual is exact before softcap; capped scores need not add.
    labels,means,tree=hierarchy(rand(128,5),depth=2,iterations=5)
    assert len(set(labels.tolist()))==4
    residual=readers-means[torch.arange(17)%4]
    errors['reader_linearity']=float((final@readers.T-final@means[torch.arange(17)%4].T-final@residual.T).abs().max())
    assert max(errors.values())<1e-10
    return {'passed':True,'errors':errors,'tree_nodes':len(tree),'gpu_accessed':False}
