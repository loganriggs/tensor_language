"""One shared input factor with unrestricted output-linear partners.

For unit a, project Q onto {sym(a b.T)}: P Q+Q P-P Q P.
This is the exact post-normalization input-removal contribution.
"""
import json
from pathlib import Path
import torch
from congruence_block_operator_v1 import sandwich,factor_forms


def value_gradient(a,left,right,k,second):
    la=left@a;ra=right@a;h=la*ra;kh=k@h
    value=2*(a@second@a)-h@kh
    grad=4*(second@a)-2*(left.T@(ra*kh)+right.T@(la*kh))
    return value,grad


def native_partner(a,left,right,down):
    la=left@a;ra=right@a
    return down@(ra[:,None]*left+la[:,None]*right)-(down@(la*ra))[:,None]*a[None,:]


def optimize(initial,left,right,k,second,max_steps=2000,tolerance=1e-8):
    a=initial/initial.norm();total=second.trace();history=[];converged=False
    initial_rate=min(1000.,float(total/(4*torch.linalg.eigvalsh(second)[-1])))
    # Projected gradient ascent with Armijo backtracking and normalized retraction.
    for step in range(max_steps):
        value,grad=value_gradient(a,left,right,k,second)
        grad=(grad-a*(a@grad))/total;score=float(value/total);station=float(grad.norm())
        if station<=tolerance:converged=True;break
        rate=initial_rate
        for search in range(60):
            candidate=a+rate*grad;candidate/=candidate.norm()
            new,_=value_gradient(candidate,left,right,k,second)
            if float(new/total)>=score+1e-4*rate*station**2:break
            rate*=.5
        else:break
        a=candidate
        if step%20==0:history.append(dict(step=step,capture=score,tangent_stationarity=station,rate=rate))
    value,grad=value_gradient(a,left,right,k,second)
    station=float((grad-a*(a@grad)).norm()/total)
    return a,dict(converged=converged or station<=tolerance,steps=step+1,
                  capture=float(value/total),tangent_stationarity=station,history=history)


def control():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(1087)
    forms=torch.randn(5,7,7);forms=(forms+forms.transpose(1,2))/2
    l,r,w=factor_forms(forms);k=w.T@w;second=sandwich(l,r,k,torch.eye(7))
    a=torch.randn(7);a/=a.norm();p=a[:,None]*a[None,:]
    star=p@forms+forms@p-p@forms@p
    score,grad=value_gradient(a,l,r,k,second)
    score_error=abs(float(score-star.square().sum()))/float(forms.square().sum())
    variable=a.clone().requires_grad_();v,_=value_gradient(variable,l,r,k,second)
    automatic=torch.autograd.grad(v,variable)[0]
    gradient_error=float((automatic-grad).norm()/automatic.norm())
    x=torch.randn(19,7);removed=x-(x@a)[:,None]*a
    original=torch.einsum('ni,vij,nj->nv',x,forms,x)
    counter=torch.einsum('ni,vij,nj->nv',removed,forms,removed)
    partner=native_partner(a,l,r,w)
    predicted=(x@a)[:,None]*(x@partner.T)
    deletion_error=float((original-counter-predicted).norm()/original.norm())
    b=torch.randn(7);b-=a*(a@b);b/=b.norm();pb=b[:,None]*b[None,:]
    star_b=pb@forms+forms@pb-pb@forms@pb
    cross=2*torch.einsum('i,vij,j->v',a,forms,b)
    joint_removed=x-(x@a)[:,None]*a-(x@b)[:,None]*b
    joint_effect=original-torch.einsum('ni,vij,nj->nv',joint_removed,forms,joint_removed)
    composed=predicted+torch.einsum('ni,vij,nj->nv',x,star_b,x)-(x@a)[:,None]*(x@b)[:,None]*cross
    composition_error=float((joint_effect-composed).norm()/original.norm())
    planted_a=torch.randn(7);planted_a/=planted_a.norm()
    partners=torch.randn(12,7);partners-=((partners@planted_a)[:,None]*planted_a)
    planted=(planted_a[None,:,None]*partners[:,None,:]+partners[:,:,None]*planted_a[None,None,:])/2
    ll,rr,ww=factor_forms(planted);kk=ww.T@ww;ss=sandwich(ll,rr,kk,torch.eye(7))
    initial=torch.linalg.eigh(ss).eigenvectors[:,-1]
    found,fit=optimize(initial,ll,rr,kk,ss)
    random_found,random_fit=optimize(torch.randn(7),ll,rr,kk,ss)
    result=dict(instrument_passed=max(score_error,gradient_error,deletion_error,composition_error)<1e-10
                and fit['converged'] and fit['capture']>1-1e-10 and random_fit['converged'] and random_fit['capture']>1-1e-10,
                score_identity_error=score_error,gradient_error=gradient_error,
                exact_deletion_identity_error=deletion_error,orthogonal_two_reader_composition_error=composition_error,
                planted_capture=fit['capture'],planted_reader_cosine=float(abs(found@planted_a)),
                planted_fit=fit,random_planted_fit=random_fit,
                scope='Local quadratic input edit after normalization, without renormalizing. Composition includes the shared cross term. No behavioral/circuit identity claim.')
    Path(__file__).with_name('SHARED_INPUT_FACTOR_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));assert result['instrument_passed']


if __name__=='__main__':control()
