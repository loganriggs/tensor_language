"""Compare port-fit and producer-fit at identical folded target/norm/rank."""
import json
from pathlib import Path
import torch
from head17_output_block_objective_v1 import build
from head17_source_interface_v1 import CHECKPOINT

P=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2)
    t,ids=build();sd=torch.load(CHECKPOINT,map_location='cpu',weights_only=True,mmap=True)
    mu=float(sd['transformer.h.17.attn.lamb'])
    def value(layer):return sd[f'transformer.h.{layer}.attn.c_v.weight'].reshape(9,128,1152)[2].double()
    f=torch.cat(((1-mu)*value(17),mu*value(0)),1)
    gram=f@f.T;h=torch.linalg.cholesky((gram+gram.T)/2)
    records=[]
    for name,tensor in [('individual_tokens',t),('pair_contrasts',t[::2]-t[1::2])]:
        flat=tensor.flatten(0,1)
        u,s,vh=torch.linalg.svd(flat,full_matrices=False)
        weighted=flat@h
        a,b,c=torch.linalg.svd(weighted,full_matrices=False)
        total=weighted.square().sum()
        cells=[]
        for rank in [16,32,64,96,120]:
            unweighted_fit=(u[:,:rank]*s[:rank])@vh[:rank]
            folded_fit=(a[:,:rank]*b[:rank])@c[:rank]
            port_error=float((((unweighted_fit-flat)@h).square().sum()/total).sqrt())
            folded_error=float(((folded_fit-weighted).square().sum()/total).sqrt())
            assert folded_error<=port_error+1e-12
            # Compile the same rank-k reader back into head coordinates.
            reader=torch.linalg.solve_triangular(h.T,c[:rank].T,upper=True).T
            recovered=(a[:,:rank]*b[:rank])@reader
            reconstruction=float(((recovered-flat)@h).square().sum()/total)
            assert abs(reconstruction-folded_error**2)<1e-10
            cells.append(dict(rank=rank,port_fit_folded_error=port_error,producer_fit_folded_error=folded_error,
                squared_error_gain=1-(folded_error/port_error)**2,
                folded_reader_scalars=rank*(flat.shape[0]+f.shape[1]),
                head_reader_scalars_if_F_supplied=rank*(flat.shape[0]+128)))
        records.append(dict(target=name,cells=cells))
    result=dict(schema='head17.value.pullback.matched.metric.v1',targets=records,
        scope='Both fits evaluated on identical T composed with mixed current/first value F. '
        'Fixed shared-head-mode rank; no full QK/RMS/retained-path or native behavior claim. '
        'Folding gains compared without changing denominator or output scope.')
    with (P/'HEAD17_VALUE_PULLBACK_MATCHED_V1_RESULT.json').open('x') as out:
        json.dump(result,out,indent=2);out.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
