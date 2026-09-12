"""Weight-only full cross-head Gram of frozen shared-source component.

Positions0,1,2,4,8,16 fixed before evaluation. Include cross-head cancellation.
These coefficient functions retain reference normalization, not native gates.
No semantic identification, sparsity threshold, fit or native selection claim.
"""
import json
from pathlib import Path
import torch
from shared_cubic_source_projection_v1 import atom_gram,cross_factors,query_output_gram
from folded_normalized_router_v1 import rotary
P=Path(__file__).resolve().parent


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    binding=json.loads((P/'REGIONAL_SPECTRAL_NORMALIZERS_V1_BINDING.json').read_text())['files']
    sd=torch.load(next(x for x in binding if x.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
    def weight(name):return sd['transformer.h.17.attn.'+name+'.weight'].double().reshape(9,128,1152)
    q1,k1,q2,k2=[weight(n) for n in ('c_q','c_k','c_q2','c_k2')]
    lam=float(sd['transformer.h.17.attn.lamb']);v=torch.cat([(1-lam)*weight('c_v'),lam*sd['transformer.h.0.attn.c_v.weight'].double().reshape(9,128,1152)],-1)
    out=sd['transformer.h.17.attn.c_proj.weight'].double().reshape(1152,9,128)
    g0=1/(q1.flatten(1).norm(dim=1)*q2.flatten(1).norm(dim=1)*k1.flatten(1).norm(dim=1)*k2.flatten(1).norm(dim=1))
    atoms=torch.load(P/'COMMON_QUADRATIC_SOURCE_BLOCK_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['atoms'].double()
    g=atom_gram(atoms);dual=torch.linalg.inv(g)[-2:];source=g[-2:,-2:];cells=[]
    for position in (0,1,2,4,8,16):
        rotation=rotary(position,128);keys=[torch.cat([torch.einsum('ab,hbd->had',rotation,k),torch.zeros_like(k)],-1) for k in (k1,k2)]
        factors=cross_factors(atoms,q1,keys[0],q2,keys[1],v,out*g0[None,:,None])
        # Treat head and source index as one bank, preserving every cross-head pair.
        flattened=tuple(t.reshape(1,9*16,6,1152) for t in factors)
        kernel=query_output_gram(flattened)[0].reshape(9,16,9,16)
        gram=torch.einsum('ir,jt,ij,hrkt->hk',dual,dual,source,kernel)
        symmetry=float((gram-gram.T).abs().max()/gram.abs().max())
        eigen=torch.linalg.eigvalsh((gram+gram.T)/2);assert symmetry<1e-10 and eigen[0]>-1e-10*eigen[-1]
        diagonal=gram.diag();order=torch.argsort(diagonal,descending=True);total=gram.sum()
        curves=[]
        for count in (1,2,3,5,8,9):
            omit=order[count:];error=gram[omit][:,omit].sum().clamp_min(0)
            curves.append(dict(retained=count,heads=order[:count].tolist(),relative_coefficient_error=float((error/total).sqrt())))
        cells.append(dict(position=position,gram=gram.tolist(),diagonal_energy_fraction=(diagonal/diagonal.sum()).tolist(),
            cross_head_sum_over_diagonal=float((total-diagonal.sum())/diagonal.sum()),
            normalized_gram_min_eigenvalue=float(eigen[0]/eigen[-1]),symmetry_error=symmetry,curves=curves))
    result=dict(cells=cells,scope='Exact reference-normalized coefficient Gram of frozen two-source block; head-indexed private consumers include cross terms. Native context-dependent gates can change relative importance. Energy ordering is descriptive, not causal unit selection.')
    (P/'SHARED_PRIVATE_CONSUMER_GRAM_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([dict(position=c['position'],fractions=c['diagonal_energy_fraction'],cross=c['cross_head_sum_over_diagonal'],curves=c['curves']) for c in cells],indent=2))


if __name__=='__main__':main()
