"""Dense multihomogeneous controls for every background/producer degree."""
from pathlib import Path
import itertools,json,hashlib,math
import torch
from mixed_repeated_contraction_v1 import contract,matchings
P=Path(__file__).resolve().parent
def main():
    torch.set_num_threads(2);torch.manual_seed(73200);dtype=torch.float64;din=2;dout=3;m=4
    left=torch.randn(m,din,dtype=dtype);right=torch.randn(m,din,dtype=dtype);down=torch.randn(dout,m,dtype=dtype,requires_grad=True)
    forms=torch.randn(3,dout,dout,dtype=dtype,requires_grad=True);a=(forms+forms.transpose(-1,-2))/2
    raw=torch.einsum('ab,jcd->jabcd',a[0],a[1:]);c=sum(raw.permute(0,*[i+1 for i in perm]) for perm in itertools.permutations(range(4)))/24
    producer=torch.einsum('ah,hi,hj->aij',down,left,right);producer=(producer+producer.transpose(-1,-2))/2
    errors=[];oracle_sum=0.;dense_sum=0.
    for grade in range(5):
        nb=4-grade;output='abcd'[:nb]+'efghijkl'[:2*grade];terms=['zabcd'];args=[c]
        for j in range(grade):terms.append('abcd'[nb+j]+'efghijkl'[2*j:2*j+2]);args.append(producer)
        tensor=torch.einsum(','.join(terms)+'->z'+output,*args);sym=torch.zeros_like(tensor)
        for matching in matchings(tuple(range(2*grade))):
            order=[i for pair in matching for i in pair];inverse=[order.index(i) for i in range(2*grade)]
            sym=sym+tensor.permute(0,*range(1,nb+1),*[1+nb+i for i in inverse])
        sym=sym*(math.comb(4,grade)/len(matchings(tuple(range(2*grade)))))
        b=torch.randn(5,nb,dout,dtype=dtype);x=torch.randn(5,2*grade,din,dtype=dtype)
        vectors=list(b.unbind(1))+list(x.unbind(1));actual=torch.einsum('z'+output+','+','.join('n'+i for i in output)+'->nz',sym,*vectors)
        predicted=contract(b,x,left,right,down,forms,grade)
        permuted=contract(b.flip(1),x.flip(1),left,right,down,forms,grade)
        errors.append(dict(grade=grade,dense=float(((predicted-actual).norm()/actual.norm()).detach()),permutation=float(((predicted-permuted).norm()/predicted.norm()).detach())))
        probe=torch.randn_like(predicted);oracle_sum=oracle_sum+(predicted*probe).sum();dense_sum=dense_sum+(actual*probe).sum()
    gradients=torch.autograd.grad(oracle_sum,(down,forms),retain_graph=True);reference=torch.autograd.grad(dense_sum,(down,forms))
    ge=[float((g-r).norm()/r.norm()) for g,r in zip(gradients,reference)]
    b=torch.randn(5,dout,dtype=dtype);x=torch.randn(5,din,dtype=dtype);values=[]
    for grade in range(5):values.append(contract(b[:,None,:].expand(-1,4-grade,-1),x[:,None,:].expand(-1,2*grade,-1),left,right,down,forms,grade))
    y=b+((x@left.T)*(x@right.T))@down.T;q=torch.einsum('ni,kij,nj->nk',y,a,y);direct=q[:,0,None]*q[:,1:];diagonal=float(((sum(values)-direct).norm()/direct.norm()).detach())
    assert max([diagonal]+ge+[r[k] for r in errors for k in ('dense','permutation')])<1e-10
    result=dict(grade_errors=errors,gradient_errors=ge,diagonal_sum_error=diagonal,source_sha=hashlib.sha256((P/'mixed_repeated_contraction_v1.py').read_bytes()).hexdigest())
    out=P/'MIXED_REPEATED_CONTRACTION_V1_CONTROL.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
