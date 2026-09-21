"""Retain matrix replay tolerance; change only an equivalent output-pair basis."""
import math,json
from pathlib import Path
import torch
from quadratic_pair_blocks import compile_pair
P=Path(__file__).resolve().parent


def reconstruct(pair):
    T=pair['input_transform'];i,j,kind=pair['product_indices'];u,v=T[:,i],T[:,j];a=torch.where((kind==1)[None,:],u+v,u);b=torch.where((kind==1)[None,:],u-v,v);raw=torch.einsum('pk,ip,jp->kij',pair['product_weights'],a,b)
    return (raw+raw.transpose(-1,-2))/2


def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    source=torch.load(P/'QUARTIC_RANK_BLOCK_16_V1.pt',weights_only=True);E=source['root_eigenvectors'].double();v=source['root_eigenvalues'].double();forms=(E*v[:,None,:])@E.transpose(-1,-2);A,B=forms[4:6];records=[]
    for angle in [0.,math.pi/8,math.pi/4,3*math.pi/8]:
        c,s=math.cos(angle),math.sin(angle);R=torch.tensor([[c,s],[-s,c]])
        try:
            pair=compile_pair(c*A+s*B,-s*A+c*B);pair['product_weights']=pair['product_weights']@R;rebuilt=reconstruct(pair);errors=[float((a-b).norm()/a.norm()) for a,b in zip([A,B],rebuilt)];assert max(errors)<1e-10;records.append(dict(angle=angle,success=True,original_matrix_replay=errors,diagnostics=pair['diagnostics']))
        except ValueError as e:records.append(dict(angle=angle,success=False,reason=str(e)))
    result=dict(records=records,prediction_equivalent_basis_repairs=any(r['success'] for r in records[1:]),scope='Same two native root quadratic forms; orthogonal output-coordinate rotation, unchanged1e-10matrix replay tolerance. No data fitting or relaxation. Doesnotguarantee all defective/singular pencils work.')
    (P/'PAIR_BASIS_ROTATION_AUDIT_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
