"""Executed explanation for a weak joint-QK shared-rank gain."""
import json
from pathlib import Path
import torch
from head17_source_interface_v1 import CHECKPOINT
from folded_normalized_router_v1 import fold
from joint_qk_source_gram_v1 import gram


def main():
    torch.set_num_threads(2)
    sd=torch.load(CHECKPOINT,map_location='cpu',weights_only=True,mmap=True)
    def h(layer,key):return sd[f'transformer.h.{layer}.attn.{key}.weight'].reshape(9,128,1152)[2].double()
    mu=float(sd['transformer.h.17.attn.lamb'])
    f=torch.cat(((1-mu)*h(17,'c_v'),mu*h(0,'c_v')),1)
    v=f@f.T;root=torch.linalg.cholesky(v)
    weights=[h(17,key) for key in ['c_q','c_k','c_q2','c_k2']]
    records=[]
    for pos in [1,8]:
        a,b=[torch.cat((s[0],torch.zeros_like(s[0])),1) for s in fold(weights,pos,0)]
        g,_=gram(a,b,f)
        left=torch.linalg.solve_triangular(root,g,upper=False)
        whitened=torch.linalg.solve_triangular(root,left.T,upper=False).T
        ev=torch.linalg.eigvalsh((whitened+whitened.T)/2)
        scale=(g*v).sum()/v.square().sum()
        condition=float(ev[-1]/ev[0])
        records.append(dict(query_position=pos,source_position=0,
            generalized_eigenvalue_ratio=condition,
            best_scaled_value_metric_relative_error=float((g-scale*v).norm()/g.norm()),
            maximum_possible_squared_gain_from_reoptimizing_metric=1-1/condition,
            scope='Uniform norm-equivalence bound for fixed A,B,F and any reader-matrix approximation family '
                  'whose value-metric optimum is attained. Not a bound on changing QK factors or arithmetic topology.'))
    with Path(__file__).with_name('JOINT_QK_METRIC_EQUIVALENCE_V1_RESULT.json').open('x') as out:
        json.dump(records,out,indent=2);out.write('\n')
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
