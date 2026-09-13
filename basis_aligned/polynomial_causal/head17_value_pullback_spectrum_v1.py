"""Known value interface pulled through the selected-output mixed tensor.

This folds linear value production, not score dependence or normalization.
Full-row-rank value maps do not restrict the set of arbitrary head writes.
"""
import json,time
from pathlib import Path
import torch
from head17_output_block_objective_v1 import build
from head17_source_interface_v1 import CHECKPOINT

P=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2);torch.manual_seed(9561);started=time.perf_counter()
    t,ids=build()
    sd=torch.load(CHECKPOINT,map_location='cpu',weights_only=True,mmap=True)
    def value(layer):
        return sd[f'transformer.h.{layer}.attn.c_v.weight'].reshape(9,128,1152)[2].double()
    mu=float(sd['transformer.h.17.attn.lamb'])
    current=(1-mu)*value(17)
    mixed=torch.cat((current,mu*value(0)),dim=1)
    tensors={'individual_tokens':t,'pair_contrasts':t[::2]-t[1::2]}
    records=[]
    for name,f in [('head_port',torch.eye(128,dtype=torch.float64)),('current_value',current),('mixed_value',mixed)]:
        gram=f@f.T
        root=torch.linalg.cholesky((gram+gram.T)/2)
        singular=torch.linalg.svdvals(f)
        row=dict(producer=name,shape=list(f.shape),row_rank=int(torch.linalg.matrix_rank(f)),
                 singular_condition=float(singular[0]/singular[-1]),targets={})
        for target,tensor in tensors.items():
            flat=tensor.flatten(0,1)
            pulled=flat@root
            spectrum=torch.linalg.svdvals(pulled).square()
            total=float(spectrum.sum())
            # Small output/input slice directly materializes the expanded producer.
            test=flat[:19]
            direct=test@f
            replay=abs(float((test@root).square().sum()/direct.square().sum())-1)
            assert replay<1e-10
            row['targets'][target]=dict(coefficient_energy=total,direct_slice_energy_replay=replay,
                shared_rank_relative_errors={str(k):float((spectrum[k:].sum()/total).sqrt()) for k in [8,16,32,64,96,120]},
                ranks_for_relative_error={str(e):next(k for k in range(129) if float(spectrum[k:].sum()/total)<=e*e) for e in [.1,.05,.02]})
        # Evaluate the source pullback on arbitrary vectors against the port interface.
        z=torch.randn(7,1152,dtype=torch.float64)
        source=torch.randn(7,f.shape[1],dtype=torch.float64)
        h=source@f.T
        port=torch.einsum('ni,oia,na->no',z,t,h)
        source_reader=torch.einsum('ni,oia->noa',z,t)@f
        native=torch.einsum('noj,nj->no',source_reader,source)
        row['source_interface_replay']=float((port-native).norm()/port.norm())
        assert row['source_interface_replay']<1e-10
        records.append(row)
    result=dict(schema='head17.value.pullback.spectrum.v1',token_ids=ids,mixture=mu,
                producers=records,wall_seconds=time.perf_counter()-started,
                scope='Exact shared-head-mode spectral restriction after linear V pullback only. '
                'Arbitrary independent normalized source inputs; no QK product folding, shared-source '
                'polynomial constraint, normalized circuit fit, or causal validation. Known V interface reused.')
    with (P/'HEAD17_VALUE_PULLBACK_SPECTRUM_V1_RESULT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
