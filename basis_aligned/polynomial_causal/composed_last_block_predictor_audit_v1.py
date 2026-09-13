"""Scope and per-example audit of the composed conditional block predictor."""
from pathlib import Path
import json
import torch
P=Path(__file__).resolve().parent


def main():
    new=torch.load(P/'COMPOSED_LAST_BLOCK_PREDICTOR_V1_ARTIFACT.pt',weights_only=True)
    reference=torch.load(P/'ADDITIVE_HEAD_RAW_PORTS_NATIVE_V1_ARTIFACT.pt',weights_only=True)
    cells=[];partitions=[]
    for group in range(5):
        sl=slice(24*group,24*(group+1))
        z=new['readouts'][sl,:,0].double();m=reference['measures'][sl,:,0].double()
        local=m[:,4]-z[:,0];total=m[:,2]-m[:,1]-m[:,3]+m[:,0]
        inherited=m[:,2]-m[:,4]
        readout=z[:,0]-m[:,1]-m[:,3]+m[:,0]
        assert float((inherited+local+readout-total).abs().max())<1e-12
        partitions.append(dict(group=group,terms={name:dict(norm_over_total=float(v.norm()/total.norm()),aligned_over_total=float((v*total).sum()/total.square().sum())) for name,v in [('inherited_before_block17',inherited),('local_block17',local),('final_readout',readout)]}))
        for name,j in [('full_head_plus_mlp',1),('compact_head_plus_mlp',2),('mlp_only',3)]:
            pred=z[:,j]-z[:,0];wrong=pred*local<0
            cells.append(dict(group=group,candidate=name,local_relative_error=float((pred-local).norm()/local.norm()),
                same_nonzero_sign=int((pred*local>0).sum()),opposite_nonzero_sign=int(wrong.sum()),
                zero_prediction_count=int((pred==0).sum()),
                maxabs_reference_on_wrong_sign=float(local[wrong].abs().max()) if wrong.any() else 0,
                reference_norm_fraction_on_wrong_sign=float(local[wrong].norm()/local.norm()),
                original_total_relative_error=float((pred-total).norm()/total.norm()),
                original_total_aligned_fraction=float((pred*total).sum()/total.square().sum()),
                meanabs_predicted_effect=float(pred.abs().mean())))
    result=dict(cells=cells,exact_outcome_partition=partitions,scope='Existing diagnostic rows. Different additive backgrounds prevent unique causal coverage interpretation; original total residual is a predictive sufficiency comparison. No fitted gain.')
    (P/'COMPOSED_LAST_BLOCK_PREDICTOR_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    for c in cells:
        if c['candidate']!='mlp_only':print(c)


if __name__=='__main__':main()
