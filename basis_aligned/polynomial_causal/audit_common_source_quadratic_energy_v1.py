"""Complete centered source-energy accounting with the exact output mean."""
from pathlib import Path
import json,torch
from shared_source_attention_quadratic_v1 import symmetry_energies_low_rank
P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')

def main():
    torch.set_num_threads(2);sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    u=sd['lm_head.weight'];mean=u.double().mean(0)
    l,r,d=[sd[f'transformer.h.17.mlp.{k}.weight'].double() for k in ['Left','Right','Down']]
    raw=(l.T*(mean@d))@r;q=(raw+raw.T)/2
    o=sd['transformer.h.17.attn.c_proj.weight'].double();vc=sd['transformer.h.17.attn.c_v.weight'].double();vb=sd['transformer.h.0.attn.c_v.weight'].double();mix=float(sd['transformer.h.17.attn.lamb']);w=torch.cat(((1-mix)*vc,mix*vb),dim=1)
    generator=torch.Generator().manual_seed(33417)
    def permute(a):
        perm=torch.randperm(a.shape[1],generator=generator);sign=2*torch.randint(0,2,(a.shape[1],),generator=generator)-1
        return a[:,perm]*sign
    _=permute(w);control=torch.cat([permute(x) for x in w.split(128)])
    ref=json.loads((P/'FULL_SOURCE_QUADRATIC_ENERGY_V1_RESULT.json').read_text());rows={}
    for name,values in [('native',w),('independent_source_permutations',control)]:
        common={k:len(u)*float(v) for k,v in symmetry_energies_low_rank(q,o,values,9).items()}
        full={k:ref['maps'][name][k]+common[k] for k in common}
        rows[name]=dict(common=common,full_unembedding=full,common_fraction=common['total']/full['total'],full_antisymmetric_fraction=full['antisymmetric']/full['total'])
    # Orthogonal output-mean split is preserved by every linear input fold.
    torch.manual_seed(182);x=torch.randn(16,1152,dtype=torch.float64)
    direct=((x@l.T)*(x@r.T))@(mean@d)
    replay=torch.einsum('ni,ij,nj->n',x,q,x)
    bridge=float((direct-replay).norm()/direct.norm())
    total_error=abs(rows['native']['full_unembedding']['total']/rows['independent_source_permutations']['full_unembedding']['total']-1)
    result=dict(maps=rows,scalar_fold_relative_error=bridge,permutation_total_invariance=total_error,passed=max(bridge,total_error)<=1e-10,scope='Exact all-U completion: centered and vocabulary-common outputs remain orthogonal after folding. Includes the common function; no natural-state weighting, radial simplification or causal importance claim.')
    (P/'COMMON_SOURCE_QUADRATIC_ENERGY_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['passed']
if __name__=='__main__':main()
