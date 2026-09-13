"""Gauge-invariant private span overlap and frozen-global ablation comparison."""
import argparse,json,time
from pathlib import Path
import torch
from audit_fullu_output_functions_v1 import CK
from joint_quadratic_fit_v1 import product_cross

P=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--g',type=int,default=64,choices=[64,128]);args=parser.parse_args()
    torch.set_num_threads(2);started=time.perf_counter()
    sd=torch.load(CK,map_location='cpu',weights_only=True,mmap=True)
    u=sd['lm_head.weight'].double()
    l,r,d=[sd[f'transformer.h.17.mlp.{key}.weight'].double() for key in ['Left','Right','Down']]
    metric=d@product_cross(l,r,l,r)@d.T;root=torch.linalg.cholesky((metric+metric.T)/2)
    target=u@root;norm=target.square().sum()
    program=torch.load(P/f'FULLU_SHARED_LOCAL_FIT_V1_G{args.g}_PROGRAM.pt',map_location='cpu',weights_only=True)
    p=program['global_reader'].double()@root
    q=torch.linalg.qr(p.T,mode='reduced')[0]
    mean=program['mean'].double()@root
    global_part=program['global_codes'].double()@p
    fixed_global_error=float(((global_part+mean-target).square().sum()/norm).sqrt())
    centered=target-mean
    optimal_global=(centered@q)@q.T
    recoded_global_error=float(((optimal_global-centered).square().sum()/norm).sqrt())
    complements=[];global_overlap=[]
    for bank in program['local_readers']:
        raw=bank.double()@root
        private=torch.linalg.qr(raw.T,mode='reduced')[0].T
        projected=(private@q)@q.T
        global_overlap.append(float(projected.square().sum()/private.square().sum()))
        remainder=private-projected
        _,s,vh=torch.linalg.svd(remainder,full_matrices=False)
        assert float(s[-1]/s[0])>1e-8
        complements.append(vh)
    pairs=[]
    for i in range(len(complements)):
        for j in range(i):
            values=torch.linalg.svdvals(complements[i]@complements[j].T)
            pairs.append(dict(first=i,second=j,maximum_principal_cosine=float(values[0]),
                              mean_subspace_overlap=float(values.square().mean()),
                              directions_cosine_above_99pct=int((values>=.99).sum())))
    overlap=torch.tensor([z['mean_subspace_overlap'] for z in pairs],dtype=torch.float64)
    stacked=torch.cat(complements)
    singular=torch.linalg.svdvals(stacked)
    rank=int((singular>singular[0]*1e-8).sum())
    result=dict(schema='shared.local.private.subspaces.v1',global_width=args.g,
        fixed_code_private_deletion_coefficient_error=fixed_global_error,
        optimal_recoding_frozen_global_coefficient_error=recoded_global_error,
        private_bank_fraction_in_global_span=global_overlap,
        median_pairwise_complement_overlap=float(overlap.median()),
        maximum_pairwise_complement_overlap=float(overlap.max()),
        pairs_with_principal_cosine_above_99pct=sum(z['directions_cosine_above_99pct']>0 for z in pairs),
        complementary_union_rank=rank,complementary_union_rows=len(stacked),
        complementary_union_condition=float(singular[0]/singular[-1]),
        strongest_pairs=sorted(pairs,key=lambda z:z['maximum_principal_cosine'],reverse=True)[:8],
        wall_seconds=time.perf_counter()-started,
        scope='Full coefficient geometry of frozen banks. Complement angles remove global overlap and '
        'are invariant to nonsingular within-bank recoding. Counterfactual deletion and optimal recoding '
        'are distinct, neither is native causal removal or cross-behavior reuse.')
    with (P/f'SHARED_LOCAL_SUBSPACE_AUDIT_V1_G{args.g}_RESULT.json').open('x') as out:
        json.dump(result,out,indent=2);out.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
