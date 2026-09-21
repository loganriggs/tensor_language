"""Verify the exported shared graph and literal reuse accounting on CPU."""
import json
from pathlib import Path
import torch
from sparse_quartic_bank import features,validate_pairs
P=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(2);rows=[]
    x=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][1]['rows'][:32].double()
    for seed in [1101,1102]:
        old=torch.load(P/f'SPARSE_QUARTIC_BANK_SEED{seed}_V1.pt',weights_only=True)
        new=torch.load(P/f'SPARSE_SUPPORT_EXCHANGE_SEED{seed}_V1.pt',weights_only=True)
        assert all(torch.equal(a,b) for a,b in zip(old['factors'],new['factors']))
        assert torch.equal(old['writer'],new['writer'])
        U,V=[a.double() for a in new['factors']];pairs=new['pairs'];validate_pairs(pairs,144)
        oldset=set(map(tuple,old['pairs'].T.tolist()));newset=set(map(tuple,pairs.T.tolist()))
        assert len(newset)==512 and all((i,i) in newset for i in range(144))
        shared=features(x,U,V,pairs)
        # Separate per-root evaluation deliberately duplicates q computations.
        naive=[]
        for i,j in pairs.T.tolist():
            qi=((x@U[i].T)*(x@V[i].T)).sum(1)
            qj=qi if i==j else ((x@U[j].T)*(x@V[j].T)).sum(1)
            naive.append(qi*qj)
        naive=torch.stack(naive,1);error=float((shared-naive).norm()/naive.norm());assert error<1e-12
        degrees=torch.bincount(pairs.flatten(),minlength=144)
        # Diagonal square consumes its producer once, not twice in this count.
        consumer_counts=degrees-1
        rows.append(dict(seed=seed,changed_pairs=len(newset-oldset),unchanged_pairs=len(newset&oldset),quadratic_producers=144,root_products=512,shared_products=144*4+512,without_cross_root_reuse_products=144*5+368*9,stored_float_coefficients=U.numel()+V.numel()+new['coefficients'].numel()+new['writer'].numel(),integer_indices=pairs.numel(),producer_consumer_min=int(consumer_counts.min()),producer_consumer_max=int(consumer_counts.max()),shared_vs_unshared_replay=error))
    result=dict(rows=rows,scope='Export graph integrity and arithmetic accounting only. Shared1088 versus per-root4032products is an existing representation advantage, not a new saving caused by these constant-cost edits. Linear additions and coefficient multiplications are not included in variable-product count; coefficient storage is separate.')
    (P/'SPARSE_SUPPORT_EXCHANGE_GRAPH_AUDIT_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
