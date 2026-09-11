"""Known common-output contribution to the weight-only response alignment."""
import json,hashlib
from pathlib import Path
import torch
from audit_rank_one_congruence_response_v1 import CK
P=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2)
    receipt=json.loads((P/'NATIVE_CONGRUENCE_SPECTRUM_V1_RESULT.json').read_text())
    cache=Path(receipt['witness_cache']['path'])
    assert hashlib.sha256(cache.read_bytes()).hexdigest()==receipt['witness_cache']['sha256']
    vectors=torch.load(cache,weights_only=True,map_location='cpu')['vectors']
    reference=json.loads((P/'RANK_ONE_CONGRUENCE_RESPONSE_V1_AUDIT.json').read_text())
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    unembedding=sd['lm_head.weight'].double();metric=unembedding.T@unembedding;mean=unembedding.mean(0)
    l,r,d=[sd[f'transformer.h.17.mlp.{key}.weight'].double() for key in ['Left','Right','Down']]
    rows=[]
    for index,z in enumerate(vectors):
        left,_,right=torch.linalg.svd(z,full_matrices=False);u,v=left[:,0],right[0]
        bank=((r@u)[:,None]*l+(l@u)[:,None]*r)/2;mapping=d@bank
        full=mapping.T@metric@mapping;full=(full+full.T)/2
        common_direction=mean@mapping
        common=len(unembedding)*common_direction[:,None]*common_direction[None,:]
        centered=full-common
        values=torch.linalg.eigvalsh(centered)
        rows.append(dict(index=index,common_response_energy_fraction=float(common.trace()/full.trace()),
            centered_best_response_alignment=float(values[-1]/centered.trace()),
            centered_witness_response_alignment=float(v@centered@v/centered.trace()),
            full_energy_replay_error=abs(float(full.trace())/reference['rows'][index]['response_energy']-1),
            centered_min_eigenvalue_fraction=float(values[0]/centered.trace())))
    passed=all(x['full_energy_replay_error']<=1e-10 and x['centered_min_eigenvalue_fraction']>=-1e-10 for x in rows)
    result=dict(instrument_passed=passed,rows=rows,scope='Exact common/centered split of full-U directional quadratic response moments. Common output remains in the physical pre-tanh computation.')
    (P/'CONGRUENCE_COMMON_RESPONSE_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));assert passed


if __name__=='__main__':main()
