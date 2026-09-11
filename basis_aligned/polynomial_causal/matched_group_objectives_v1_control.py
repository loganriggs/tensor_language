"""Independent dense and normalized-parameter checks for both native pilot families."""
import json
from pathlib import Path
import torch
import numpy as np
from chunked_bilinear_coefficient_v1 import dense
import symmetric_ll1_objective_v1 as ll1
import shared_reader_group_objective_v1 as shared
from shared_reader_group_fit_v1 import Objective as SharedObjective


def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.manual_seed(1751)
    target=(torch.randn(13,9),torch.randn(13,9),torch.randn(7,13));truth=dense(*target);total=truth.square().sum()
    rows=[]
    for family,module,cls,shapes in [('ll1',ll1,ll1.Objective,[(3,4,9),(3,4),(3,7)]),
                                     ('shared',shared,SharedObjective,[(3,9),(3,2,9),(3,7,2)])]:
        parts=tuple(torch.randn(shape,requires_grad=True) for shape in shapes)
        forms=[dense(*module.cp(*(x[j:j+1] for x in parts))) for j in range(3)]
        loss=((truth-sum(forms)).square().sum()+.01*sum(f.square().sum() for f in forms))/total
        gradients=torch.autograd.grad(loss,parts)
        actual,g,_=module.value_gradient(target,*parts,total,.01,chunk=3)
        errors=[float((x-y).norm()/y.norm()) for x,y in zip(g,gradients)]
        objective=cls(target,tuple(x.detach() for x in parts),total,.01)
        value,gradient=objective.evaluate(objective.initial)
        direction=np.random.default_rng(1751).normal(size=len(gradient));direction/=np.linalg.norm(direction);h=1e-6
        fd=(objective.evaluate(objective.initial+h*direction)[0]-objective.evaluate(objective.initial-h*direction)[0])/(2*h)
        fd_error=abs(fd-gradient@direction)/max(1.,abs(gradient@direction))
        replay=float((dense(*module.cp(*parts))-dense(*module.cp(*objective.physical(objective.initial)))).norm()/truth.norm())
        rows.append(dict(family=family,raw_gradient_errors=errors,loss_error=abs(float(actual)-float(loss.detach())),
                         normalized_fd_error=fd_error,initial_function_replay=replay,initial_loss_replay=abs(value-float(loss.detach()))))
    result=dict(predictions=dict(pred_a_dense=all(max(r['raw_gradient_errors']+[r['loss_error'],r['initial_function_replay'],r['initial_loss_replay']])<=1e-9 for r in rows),
                                 pred_b_normalized_fd=all(r['normalized_fd_error']<=1e-6 for r in rows)),rows=rows)
    Path(__file__).with_name('MATCHED_GROUP_OBJECTIVES_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    assert all(result['predictions'].values())


if __name__=='__main__':main()
