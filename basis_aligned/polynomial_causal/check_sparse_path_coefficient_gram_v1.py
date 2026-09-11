"""Exact dense edge tensor inner products and gauge replay<=1e-10."""
import json
from pathlib import Path
import torch
from sparse_path_coefficient_gram_v1 import gram,feature_bank
from sparse_path_program_v1 import edges


def dense(program,u):
    v=feature_bank(program);a,b=edges(program);den=torch.where(a==b,2.,2**.5).to(v)
    h=(torch.einsum('ik,jk->kij',v[:,a],v[:,b])+torch.einsum('ik,jk->kij',v[:,b],v[:,a]))/den[:,None,None]
    return torch.einsum('ok,kij->koij',u@program['physical_writer'],h)


def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.manual_seed(11601)
    out=Path(__file__).with_name('SPARSE_PATH_COEFFICIENT_GRAM_V1_CONTROL.json');assert not out.exists();u=torch.randn(8,4);programs=[]
    for mode,k in [('joint',2),('independent',4)]:
        bank=torch.linalg.qr(torch.randn(k,5,2)).Q
        programs.append(dict(mode=mode,bank=bank,support=torch.arange(10),physical_writer=torch.randn(4,10),source_scale=1.))
    errors=[]
    for a in programs:
        for b in programs:
            actual=gram(a,b,u.T@u)[0];aa=dense(a,u).flatten(1);bb=dense(b,u).flatten(1);expected=aa@bb.T
            errors.append(float((actual-expected).norm()/expected.norm()))
    result=dict(passed=max(errors)<=1e-10,relative_errors=errors,scope='Full output and symmetric-source coefficient Gram; no native stability claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['passed']

if __name__=='__main__':main()
