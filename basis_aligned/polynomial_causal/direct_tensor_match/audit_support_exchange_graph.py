"""Verify the exported shared graph and literal reuse accounting on CPU."""
import json
from pathlib import Path
import torch
from sparse_quartic_bank import features,validate_pairs
P=Path(__file__).resolve().parent

def main(mixed=False):
    torch.set_num_threads(2);rows=[]
    x=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][1]['rows'][:32].double()
    for seed in [1101,1102]:
        previous='SPARSE_SUPPORT_EXCHANGE' if mixed else 'SPARSE_QUARTIC_BANK'
        current='GAUSSIAN_SUPPORT_EXCHANGE' if mixed else 'SPARSE_SUPPORT_EXCHANGE'
        old=torch.load(P/f'{previous}_SEED{seed}_V1.pt',weights_only=True)
        new=torch.load(P/f'{current}_SEED{seed}_V1.pt',weights_only=True)
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
        c=new['coefficients'].double();writer=new['writer'].double()
        output_error=float(((shared@c.T)@writer.T-(naive@c.T)@writer.T).norm()/((naive@c.T)@writer.T).norm());assert output_error<1e-12
        m,k,d=U.shape;outputs=c.shape[0];residual=writer.shape[0];roots=pairs.shape[1]
        linear_additions=2*m*k*(d-1)+m*(k-1)+outputs*(roots-1)+residual*(outputs-1)
        coefficient_multiplications=U.numel()+V.numel()+c.numel()+writer.numel()
        degrees=torch.bincount(pairs.flatten(),minlength=144)
        # Diagonal square consumes its producer once, not twice in this count.
        consumer_counts=degrees-1
        rows.append(dict(seed=seed,changed_pairs=len(newset-oldset),unchanged_pairs=len(newset&oldset),quadratic_producers=144,root_products=512,shared_products=144*4+512,without_cross_root_reuse_products=144*5+368*9,stored_float_coefficients=U.numel()+V.numel()+new['coefficients'].numel()+new['writer'].numel(),integer_indices=pairs.numel(),producer_consumer_min=int(consumer_counts.min()),producer_consumer_max=int(consumer_counts.max()),shared_vs_unshared_replay=error,full_output_replay=output_error,linear_additions=linear_additions,coefficient_multiplications=coefficient_multiplications))
    result=dict(rows=rows,scope='Export graph integrity and arithmetic accounting only. Shared1088 versus per-root4032products is an existing representation advantage, not a new saving caused by these constant-cost edits. Linear additions and coefficient multiplications are not included in variable-product count; coefficient storage is separate.')
    result['scope']+=' Added literal dense coefficient multiplication and linear addition counts; variable-variable product count is not total FLOPs. No additional saving from GPU fusion is assumed.'
    output='GAUSSIAN_SUPPORT_EXCHANGE_GRAPH_AUDIT_V1.json' if mixed else 'SPARSE_SUPPORT_EXCHANGE_GRAPH_AUDIT_V1.json'
    (P/output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--mixed',action='store_true');main(parser.parse_args().mixed)
