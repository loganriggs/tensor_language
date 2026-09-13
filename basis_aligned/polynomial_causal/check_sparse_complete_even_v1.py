"""Explicit tensor and derivative controls, followed by native CPU pricing.
Pred_a explicit coefficient loss replay <=1e-10 normalized.
Pred_b finite-difference derivative error <=1e-6 normalized.
Pred_c original-basis zero loss <=1e-10 normalized.
"""
import json,time
from pathlib import Path
import torch
from sparse_complete_even_v1 import reflect,query_grams,inner,loss
from shared_query_product_objective_v1 import rotation
from sparse_parent_reader_v1 import unpack


def tensor(a,b):
    t=torch.einsum('ij,kl->ikjl',a,b)
    t=(t+t.transpose(0,1))/2
    return (t+t.transpose(2,3))/2


def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    gen=torch.Generator().manual_seed(7131344);rows=[]
    for seed in range(4):
        l1,l2=[torch.randn(5,3,generator=gen) for _ in range(2)]
        k1,k2=[torch.randn(3,7,generator=gen) for _ in range(2)]
        b=torch.linalg.qr(torch.randn(7,2,generator=gen))[0]
        s=torch.randn(7,2,generator=gen,requires_grad=True)
        grams=query_grams(l1,l2);o1,o2=reflect(k1,b),reflect(k2,b)
        norm=inner(grams,k1,k2,k1,k2)
        original=(tensor(l1@k1,l2@k2)+tensor(l1@o1,l2@o2))/2
        candidate=(tensor(l1@k1,l2@k2)+tensor(l1@reflect(k1,s),l2@reflect(k2,s)))/2
        direct=(candidate-original).square().sum();value=loss(s,grams,k1,k2,o1,o2,norm)
        grad=torch.autograd.grad(value,s)[0];direction=torch.randn(s.shape,generator=gen);step=1e-5
        fd=(loss(s+step*direction,grams,k1,k2,o1,o2,norm)-loss(s-step*direction,grams,k1,k2,o1,o2,norm))/(2*step)
        rows.append(dict(coefficient_error=float(abs(direct-value).detach()/norm),
            gradient_error=float(abs(fd-(grad*direction).sum()).detach()/norm),
            identity_error=float(abs(loss(b,grams,k1,k2,o1,o2,norm))/norm)))
    p=Path(__file__).resolve().parent
    n=torch.load(p/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1/program.pt',weights_only=True)
    b=n['key_basis'][1].double();k1,k2=[n[k][1].double() for k in ('k1','k2')]
    l1,l2=[torch.stack([n[q][1].double().T@rotation(pos).T for pos in (1,4,16,63)]) for q in ('q1','q2')]
    grams=query_grams(l1,l2);o1,o2=reflect(k1,b),reflect(k2,b);norm=inner(grams,k1,k2,k1,k2)
    candidates={}
    for name,file in [('original_frame','SPARSE_PARENT_READER_V1_PROGRAM.pt'),('rotated_frame','SPARSE_READER_ROTATION_V1_PROGRAM.pt')]:
        item=torch.load(p/file,weights_only=True)['0.25'];s,_,_=unpack(item);s.requires_grad_(True)
        t=time.perf_counter();value=loss(s,grams,k1,k2,o1,o2,norm);grad=torch.autograd.grad(value,s)[0]
        candidates[name]=dict(normalized_squared_error=float(value.detach()/norm.sum()),
                             evaluation_gradient_seconds=time.perf_counter()-t,
                             gradient_rms=float(grad.square().mean().sqrt()))
    result=dict(pred_a=max(x['coefficient_error'] for x in rows)<=1e-10,
        pred_b=max(x['gradient_error'] for x in rows)<=1e-6,
        pred_c=max(x['identity_error'] for x in rows)<=1e-10,
        controls=rows,candidates=candidates,
        scope=__doc__+' Native benchmark uses four positional numerator coefficients; no normalized behavioral guarantee.')
    (p/'SPARSE_COMPLETE_EVEN_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
