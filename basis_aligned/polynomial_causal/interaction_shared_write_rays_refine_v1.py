"""Ten-start screen/top-three300-step refinement of the same K32 ray family."""
import json,time
from pathlib import Path
import torch
from head17_output_block_objective_v1 import build
from interaction_shared_write_rays_v1 import fit
P=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2);start=time.perf_counter();t,_=build();x=t.permute(1,2,0).reshape(-1,12).contiguous()
    screening=[]
    for seed in range(61431,61441):
        _,_,_,history=fit(x,32,seed,20)
        screening.append(dict(seed=seed,loss=history[-1]['loss']))
    selected=sorted(screening,key=lambda v:v['loss'])[:3];refined=[]
    for row in selected:
        # Deterministic replay from the seed includes the screened prefix; no new initializer.
        _,_,_,history=fit(x,32,row['seed'],300)
        result=dict(seed=row['seed'],relative_error=history[-1]['loss']**.5,
                    assignment_converged=history[-1]['changed_assignments']==0,history=history)
        refined.append(result);print(row['seed'],result['relative_error'],result['assignment_converged'],history[-1]['step'],flush=True)
    out=dict(screening=screening,refined=refined,seconds=time.perf_counter()-start,
             scope='Same weight-onlyK32single-output-ray family. Ten20-step starts/topthree300-steprefinements. Unchangedassignments certify a coordinatefixedpoint, notglobaloptimality orstablesemanticidentification. Replaying selectedscreen prefixes included inprice. No behavioraladoption.')
    (P/'INTERACTION_SHARED_WRITE_RAYS_REFINE_V1_RESULT.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':main()
