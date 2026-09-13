"""Separate common-token and spelling-contrast coefficient errors without refitting."""
from pathlib import Path
import argparse,json
import torch
from head17_output_block_objective_v1 import build,Objective
P=Path(__file__).resolve().parent


def split(x):
    x=x.reshape(6,2,*x.shape[1:])
    return (x[:,0]+x[:,1])/2**.5,(x[:,0]-x[:,1])/2**.5


def score(predicted,target):
    tm,td=split(target);pm,pd=split(predicted)
    total=float(target.square().sum())
    mean_sq=float((pm-tm).square().sum());diff_sq=float((pd-td).square().sum())
    direct=float((predicted-target).square().sum())
    assert abs(mean_sq+diff_sq-direct)<=1e-10*max(total,1)
    return dict(total_relative_error=(direct/total)**.5,
                mean_target_energy_fraction=float(tm.square().sum())/total,
                difference_target_energy_fraction=float(td.square().sum())/total,
                mean_relative_error=(mean_sq/float(tm.square().sum()))**.5,
                difference_relative_error=(diff_sq/float(td.square().sum()))**.5,
                per_pair_difference_errors=[float((a-b).norm()/b.norm()) for a,b in zip(pd,td)],
                error_partition_relative_check=abs(mean_sq+diff_sq-direct)/total)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fit',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);T,ids=build();obj=Objective(T,32)
    bases={'identity':torch.eye(12,dtype=torch.float64),
           'output_spectral':torch.linalg.eigh(obj.flat@obj.flat.T)[1].flip(1)}
    results={}
    for name,Q in bases.items():
        blocks=obj.evaluate(Q)[3];predicted=(Q@blocks.flatten(1)).reshape_as(T)*obj.scale
        results[name]=score(predicted,T)
    if args.fit:
        receipt=json.loads((P/'HEAD17_OUTPUT_BLOCK_FIT_V1_RESULT.json').read_text())
        program=torch.load(P/'HEAD17_OUTPUT_BLOCK_FIT_V1_PROGRAM.pt',weights_only=True,map_location='cpu')
        assert program['token_ids'].tolist()==ids
        predicted=(program['output_basis']@(program['left']@program['right']).flatten(1)).reshape_as(T)
        results['learned']=score(predicted,T)
        assert abs(results['learned']['total_relative_error']-receipt['best_relative_error'])<1e-8
    result=dict(results=results,token_ids=ids,
                scope='Orthogonal pair sums/differences partition the original twelve-token coefficient '
                'norm exactly. Diagnostic only: no fitting, changed output weights or native softcap claim.')
    name='HEAD17_OUTPUT_MEAN_CONTRAST_V1_FIT_AUDIT.json' if args.fit else 'HEAD17_OUTPUT_MEAN_CONTRAST_V1_BASELINES.json'
    (P/name).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
