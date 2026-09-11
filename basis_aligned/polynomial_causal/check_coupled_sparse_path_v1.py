"""A dense projection/gradient and sparse selection <=1e-9. B planted known-basis reconstruction<=1e-9.
Cold-start recovery/convergence descriptive, not guaranteed or folded into A/B.
"""
import json,itertools
from pathlib import Path
import torch
from coupled_sparse_path_v1 import coefficients,evaluate,fit_fixed
from chunked_bilinear_coefficient_v1 import dense


def features(bank,mode):
    n,r=bank.shape[1:];result=[]
    def add(a,b):
        # a,b orthogonal unless exactly the same basis column.
        same=bool(torch.equal(a,b));result.append((torch.outer(a,b)+torch.outer(b,a))/(2 if same else 2**.5))
    zero=bank.new_zeros(n,r)
    if mode=='joint':
        e=torch.cat([torch.cat([bank[0],zero],1),torch.cat([zero,bank[1]],1)],0)
        for i,j in zip(*torch.triu_indices(2*r,2*r)):add(e[:,i],e[:,j])
    else:
        es=[torch.cat([b,zero],0) if i<2 else torch.cat([zero,b],0) for i,b in enumerate(bank)]
        for i,j in zip(*torch.triu_indices(r,r)):add(es[0][:,i],es[0][:,j])
        for i in range(r):
            for j in range(r):add(es[1][:,i],es[2][:,j])
        for i,j in zip(*torch.triu_indices(r,r)):add(es[3][:,i],es[3][:,j])
    return torch.stack(result)


def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.manual_seed(11231)
    p=Path(__file__).parent;out=p/'COUPLED_SPARSE_PATH_V1_CONTROL.json';assert not out.exists()
    left,right=torch.randn(2,9,5),torch.randn(2,9,5);writer=torch.randn(4,9)
    target=dense(torch.cat(list(left),1),torch.cat(list(right),1),writer);total=target.square().sum();errors=[];rows=[]
    for mode,count in [('joint',2),('independent',4)]:
        bank=torch.linalg.qr(torch.randn(count,5,2)).Q
        c=coefficients(left,right,writer,bank,mode);feat=features(bank,mode)
        ref=torch.einsum('oij,eij->oe',target,feat);errors.append(float((c-ref).norm()/ref.norm()))
        gram=torch.einsum('eij,fij->ef',feat,feat);errors.append(float((gram-torch.eye(len(feat))).abs().max()))
        support=c.square().sum(0).topk(3).indices
        predicted=torch.einsum('oe,eij->oij',c[:,support],feat[support]);expected=(predicted-target).square().sum()/total
        loss,_=evaluate(left,right,writer,bank,mode,total,3,support);errors.append(abs(float(expected-1-loss)))
        best=max(float(c[:,inds].square().sum()) for inds in itertools.combinations(range(c.shape[1]),3))
        errors.append(abs(float(c[:,support].square().sum())-best)/best)
        x=bank.clone().requires_grad_();loss,_=evaluate(left,right,writer,x,mode,total,3,support);grad=torch.autograd.grad(loss,x)[0]
        d=torch.randn_like(x);inner=x.transpose(-1,-2)@d;d=d-x@((inner+inner.transpose(-1,-2))/2);d=d/d.norm()
        # Retraction with QR signs fixed to reference prevents sign/gauge flips.
        def retract(z):
            q,r=torch.linalg.qr(z);return q*torch.diagonal(r,dim1=-2,dim2=-1).sign().unsqueeze(-2)
        h=1e-5;vals=[]
        for sign in (1,-1):
            b=retract(bank+sign*h*d);ff=features(b,mode);cc=torch.einsum('oij,eij->oe',target,ff)
            pred=torch.einsum('oe,eij->oij',cc[:,support],ff[support]);vals.append((pred-target).square().sum()/total)
        fd=(vals[0]-vals[1])/(2*h);analytic=(grad*d).sum();err=float(abs(fd-analytic)/abs(analytic).clamp_min(1e-10));errors.append(err)
        rows.append(dict(mode=mode,coefficient_columns=c.shape[1],finite_difference_relative_error=err))
    # Exactly planted three cross-port products with independent output vectors.
    basis=torch.linalg.qr(torch.randn(2,5,2)).Q
    l=torch.zeros(3,10);r=torch.zeros_like(l)
    l[0,:5]=basis[0,:,0];r[0,5:]=basis[1,:,0]
    l[1,:5]=basis[0,:,1];r[1,5:]=basis[1,:,1]
    l[2,:5]=basis[0,:,0];r[2,5:]=basis[1,:,1]
    w=torch.randn(4,3);ll=torch.stack(l.chunk(2,1));rr=torch.stack(r.chunk(2,1));tt=dense(l,r,w);tn=tt.square().sum()
    loss,detail=evaluate(ll,rr,w,basis,'joint',tn,3);planted=abs(float(1+loss));cold=[]
    for seed in (301,302):
        torch.manual_seed(seed);b=torch.linalg.qr(torch.randn(2,5,2)).Q;history=[]
        for cycle in range(3):
            _,d=evaluate(ll,rr,w,b,'joint',tn,3);support=d['support']
            b,report=fit_fixed(ll,rr,w,b,'joint',tn,support,seconds=3,tolerance=1e-8);history.append(report)
        _,end=evaluate(ll,rr,w,b,'joint',tn,3)
        cold.append(dict(seed=seed,capture=float(end['capture']),fixed_support_report=history[-1],recovered=float(end['capture'])>=1-1e-6))
    result=dict(pred_a=max(errors)<=1e-9,pred_b=planted<=1e-9,maximum_error=max(errors),checks=rows,known_basis_planted_error=planted,
        cold_start=cold,scope='Exact full-output projection, tangent gradient and support oracle; tiny synthetic cold-start outcomes do not establish native optimization or absent structure.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['pred_a'] and result['pred_b']

if __name__=='__main__':main()
