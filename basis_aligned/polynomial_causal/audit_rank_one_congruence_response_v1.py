"""Distinguish aligned all-token derivative responses from weak input directions."""
import json,hashlib,time
from pathlib import Path
import torch
P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')


def main():
    torch.set_num_threads(2);start=time.perf_counter()
    receipt=json.loads((P/'NATIVE_CONGRUENCE_SPECTRUM_V1_RESULT.json').read_text())
    cache=Path(receipt['witness_cache']['path'])
    assert hashlib.sha256(cache.read_bytes()).hexdigest()==receipt['witness_cache']['sha256']
    vectors=torch.load(cache,weights_only=True,map_location='cpu')['vectors']
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    unembedding=sd['lm_head.weight'].double();metric=unembedding.T@unembedding
    l,r,d=[sd[f'transformer.h.17.mlp.{key}.weight'].double() for key in ['Left','Right','Down']]
    rows=[];total=receipt['native_total'];width=l.shape[1]
    for index,z in enumerate(vectors):
        left,s,right=torch.linalg.svd(z,full_matrices=False)
        u,v=left[:,0],right[0]
        bank=((r@u)[:,None]*l+(l@u)[:,None]*r)/2
        residual_map=d@bank
        moment=residual_map.T@metric@residual_map;moment=(moment+moment.T)/2
        energy=float(moment.trace());aligned=float(v@moment@v)
        eigen=torch.linalg.eigvalsh(moment)
        dot=float(u@v)
        dyad_cost=2*(energy-aligned)/(total*(1-dot*dot/width))
        # Independent token-form response check on fixed output rows.
        ids=torch.tensor([0,1,10,100,1000,10000,30000,50256])
        coeff=unembedding[ids]@d
        direct=[]
        for c in coeff:
            q=(l.T*c)@r;q=(q+q.T)/2;direct.append(q@u)
        direct=torch.stack(direct);folded=unembedding[ids]@residual_map
        replay=float((direct-folded).norm()/direct.norm())
        rows.append(dict(index=index,leading_singular_energy=float(s[0].square()/z.square().sum()),
            dyad_left_right_dot=dot,response_energy=energy,
            input_direction_energy_over_isotropic_mean=energy/(total/width),
            witness_response_alignment=aligned/energy,best_response_alignment=float(eigen[-1])/energy,
            dyad_tracefree_normalized_cost=dyad_cost,full_witness_cost=receipt['solver']['eigenvalues'][index],
            dyad_cost_over_full_witness=dyad_cost/receipt['solver']['eigenvalues'][index],
            fixed_token_response_replay_relative_error=replay))
    result=dict(instrument_passed=max(x['fixed_token_response_replay_relative_error'] for x in rows)<1e-10,
                rows=rows,wall_seconds=time.perf_counter()-start,
                scope='Exact full-U quadratic response moments at weight-selected directions. No activation sampling, task meaning, or causal circuit validation.')
    (P/'RANK_ONE_CONGRUENCE_RESPONSE_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));assert result['instrument_passed']


if __name__=='__main__':main()
