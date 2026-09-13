"""Input-unfolding lower bound for an LL1 total matrix-rank budget."""
from pathlib import Path
import json
import torch
from head17_output_block_objective_v1 import build
P=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2);T,_=build()
    singular=torch.linalg.svdvals(T.permute(1,0,2).reshape(1152,-1))
    energy=singular.square();total=energy.sum()
    result=dict(input_rank_budget=384,
                any_12_rank32_blocks_error_lower=float((energy[384:].sum()/total).sqrt()),
                minimum_total_block_ranks_for_error={str(t):next((k for k in range(1153)
                    if energy[k:].sum()<=t*t*total),1152) for t in (.02,.05,.1,.2)},
                scope='Input unfolding rank of a sum of LL1 blocks is at most sum of matrix block ranks, '
                'even with nonorthogonal output directions. Spectral tail gives coefficient-Frobenius '
                'lower bound at that rank budget; not a sparse-DAG/arithmetic or native-behavior bound.')
    (P/'HEAD17_OUTPUT_BLOCK_RANK_BOUND_V1.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
