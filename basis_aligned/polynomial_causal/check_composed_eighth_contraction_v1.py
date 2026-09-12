"""Dense eighth-order controls at input dimension2; no native weights or text."""
from pathlib import Path
import itertools,json,hashlib
import torch
from composed_eighth_contraction_v1 import contract
P=Path(__file__).resolve().parent
def main():
    torch.set_num_threads(2);torch.manual_seed(73180);dtype=torch.float64;d=2;m=4;o=3
    l=torch.randn(m,d,dtype=dtype);r=torch.randn(m,d,dtype=dtype);down=torch.randn(o,m,dtype=dtype)
    forms=torch.randn(3,o,o,dtype=dtype);forms=((forms+forms.transpose(-1,-2))/2).requires_grad_();scale=.7
    producer=torch.einsum('ah,hi,hj->aij',down,l,r)*scale;producer=(producer+producer.transpose(-1,-2))/2
    raw=torch.einsum('kab,aij,blm->kijlm',forms,producer,producer)
    quartic=sum(raw.permute(0,*[i+1 for i in order]) for order in itertools.permutations(range(4)))/24
    raw8=torch.einsum('abcd,jefgh->jabcdefgh',quartic[0],quartic[1:]);dense=torch.zeros_like(raw8)
    for subset in itertools.combinations(range(8),4):
        order=list(subset)+[i for i in range(8) if i not in subset];inverse=[order.index(i) for i in range(8)]
        dense=dense+raw8.permute(0,*[i+1 for i in inverse])/70
    x=torch.randn(7,8,d,dtype=dtype);pred=contract(x,l,r,down,forms,scale)
    actual=torch.einsum('oabcdefgh,na,nb,nc,nd,ne,nf,ng,nh->no',dense,*x.unbind(1))
    permutation=contract(x[:,[7,2,0,5,3,1,6,4]],l,r,down,forms,scale)
    same=torch.randn(5,d,dtype=dtype);v=((same@l.T)*(same@r.T))@down.T*scale;q=torch.einsum('ni,kij,nj->nk',v,forms,v)
    diagonal=contract(same[:,None,:].expand(-1,8,-1),l,r,down,forms,scale);direct=q[:,0,None]*q[:,1:]
    probe=torch.randn_like(pred);g=torch.autograd.grad((pred*probe).sum(),forms,retain_graph=True)[0];gd=torch.autograd.grad((actual*probe).sum(),forms)[0]
    errors=dict(dense=float(((pred-actual).norm()/actual.norm()).detach()),permutation=float(((permutation-pred).norm()/pred.norm()).detach()),
                diagonal=float(((diagonal-direct).norm()/direct.norm()).detach()),gradient=float((g-gd).norm()/gd.norm()))
    assert max(errors.values())<1e-10
    result=dict(errors=errors,source_sha=hashlib.sha256((P/'composed_eighth_contraction_v1.py').read_bytes()).hexdigest(),scope='Exact homogeneous pure-producer eighth-degree numerator. No native fit, mixed-source or behavioral claim.')
    out=P/'COMPOSED_EIGHTH_CONTRACTION_V1_CONTROL.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
