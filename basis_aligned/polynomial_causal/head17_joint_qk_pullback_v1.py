"""All-query/shared-source numerator coefficient Gram, two fixed RoPE offsets.

Pull the known MLP mixed tensor through the whole one-source QK1*QK2*V
numerator. Denominators and the MLP input/query dependence are not folded.
"""
import json,time
from pathlib import Path
import torch
from head17_output_block_objective_v1 import build
from head17_source_interface_v1 import CHECKPOINT
from folded_normalized_router_v1 import fold
from joint_qk_source_gram_v1 import gram

P=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2);started=time.perf_counter()
    tensor,ids=build();sd=torch.load(CHECKPOINT,map_location='cpu',weights_only=True,mmap=True)
    def head(layer,key):return sd[f'transformer.h.{layer}.attn.{key}.weight'].reshape(9,128,1152)[2].double()
    mu=float(sd['transformer.h.17.attn.lamb'])
    f=torch.cat(((1-mu)*head(17,'c_v'),mu*head(0,'c_v')),1)
    value_root=torch.linalg.cholesky(f@f.T)
    weights=[head(17,key) for key in ['c_q','c_k','c_q2','c_k2']]
    results=[]
    for position in [1,8]:
        signatures=fold(weights,position,0)
        a,b=[torch.cat((sig[0],torch.zeros_like(sig[0])),1) for sig in signatures]
        kernel,terms=gram(a,b,f)
        vals=torch.linalg.eigvalsh(kernel)
        assert float(vals.min())>0
        root=torch.linalg.cholesky(kernel)
        cells=[]
        for name,t in [('individual_tokens',tensor),('pair_contrasts',tensor[::2]-tensor[1::2])]:
            flat=t.flatten(0,1)
            weighted=flat@root;u,s,vh=torch.linalg.svd(weighted,full_matrices=False)
            va,vs,vb=torch.linalg.svd(flat@value_root,full_matrices=False)
            total=weighted.square().sum();ranks=[]
            for rank in [16,32,64,96,120]:
                value_reader=torch.linalg.solve_triangular(value_root.T,vb[:rank].T,upper=True).T
                value_fit=(va[:,:rank]*vs[:rank])@value_reader
                old=float((((value_fit-flat)@root).square().sum()/total).sqrt())
                new=float((s[rank:].square().sum()/total).sqrt())
                assert new<=old+1e-10
                ranks.append(dict(rank=rank,value_only_fit_in_joint_metric=old,joint_qk_fit_error=new,
                                  squared_error_gain=1-(new/old)**2))
            cells.append(dict(target=name,coefficient_energy=float(total),ranks=ranks,
                rank_for_2pct=next(k for k in range(129) if float(s[k:].square().sum()/total)<=.02**2)))
        results.append(dict(query_position=position,source_position=0,
            head_gram_condition=float(vals[-1]/vals[0]),
            query_trace_to_gaussian_frobenius=float(terms['query_trace'].norm()/terms['gaussian'].norm()),
            targets=cells))
    result=dict(schema='head17.joint.qk.pullback.v1',positions=results,wall_seconds=time.perf_counter()-started,
                scope='Exact separately symmetric query2/source3 numerator metric at fixed rounded RoPE. '
                'MLP residual input is an independent port. QK/RMS denominators, multi-source interactions, '
                'retained finite-edit trajectory dependence and native behavior remain outside this object.')
    with (P/'HEAD17_JOINT_QK_PULLBACK_V1_RESULT.json').open('x') as out:
        json.dump(result,out,indent=2);out.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
