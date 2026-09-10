"""Weight-only shared-subterm geometry; FP64 controls, no claim of circuit identity."""
import torch

def input_product_gram(left,right):
    left=left.double();right=right.double()
    cross=left@right.T
    return ((left@left.T)*(right@right.T)+cross*cross.T)/2

def full_output_code_grams(unembedding,down,chunk=512):
    """Consume EVERY unembedding row without allocating vocabulary x products.
    Return full and vocabulary-mean-centered product output-profile Gram matrices.
    Mean subtraction is a diagnostic only: final softcap need not preserve it.
    """
    d=down.shape[0];dev=down.device
    second=torch.zeros(d,d,dtype=torch.float64,device=dev);total=torch.zeros(d,dtype=torch.float64,device=dev)
    for rows in unembedding.split(chunk):
        rows=rows.double();second.add_(rows.T@rows);total.add_(rows.sum(0))
    down=down.double();mean=total/len(unembedding);common=mean@down
    gram=down.T@second@down
    return gram,gram-len(unembedding)*common[:,None]*common[None,:],common

def nearest_signed_profile(gram):
    norms=gram.diag().clamp_min(0).sqrt();valid=norms>1e-12
    cosine=gram/norms[:,None].clamp_min(1e-30)/norms[None,:].clamp_min(1e-30)
    score=cosine.abs();score.fill_diagonal_(-1);score[:,~valid]=-1;score[~valid]=-1
    best,j=score.max(1);ix=torch.arange(len(gram),device=gram.device)
    scale=gram[ix,j]/gram[j,j].clamp_min(1e-30)
    return dict(partner=j,cosine=cosine[ix,j],scale=scale,error=(1-best.clamp(0,1).square()).clamp_min(0).sqrt(),valid=valid)

def joint_score(q1,k1,q2,k2):
    return (q1*k1).sum(-1)*(q2*k2).sum(-1)
