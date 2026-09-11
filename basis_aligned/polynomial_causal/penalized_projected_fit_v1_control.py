"""Planted end-to-end fit: packed gradient, joint stopping and dense replay."""
import json
from pathlib import Path
import numpy as np
import torch
from penalized_projected_sparse_fit_v1 import PenalizedObjective,fit
from folded_sparse_dictionary_v1 import decode
from chunked_bilinear_coefficient_v1 import dense


def main():
    torch.set_grad_enabled(False);torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    out=Path(__file__).with_name('PENALIZED_PROJECTED_FIT_V1_CONTROL.json');assert not out.exists()
    native=(torch.eye(3),torch.eye(3),torch.eye(3));wh=torch.eye(3)
    ids=torch.tensor([[0,1],[1,2],[2,0]]*2);target=dense(*native);total=float(target.square().sum());rows=[]
    for seed in (91,92):
        torch.manual_seed(seed)
        raw=torch.cat((torch.eye(3),torch.ones(1,3)))+.05*torch.randn(4,3)
        values=torch.tensor([[1.,0.]]*6)+.05*torch.randn(6,2)
        objective=PenalizedObjective(native,wh,raw,ids,values,total,.01)
        initial,g=objective.evaluate(objective.initial);direction=-g/max(np.linalg.norm(g),1e-30);h=1e-6
        fd=abs((objective.evaluate(objective.initial+h*direction)[0]-objective.evaluate(objective.initial-h*direction)[0])/(2*h)-g@direction)
        point,writer,report=fit(objective,seconds=60,max_iterations=1000)
        rb,rv=objective.unpack(point);rv=rv/rv.norm(dim=1,keepdim=True)
        readers=decode(rb,ids,rv,torch.ones(len(rv)))[0];a,b=readers.chunk(2)
        component=(torch.einsum('ok,ki,kj->koij',writer,a,b)+torch.einsum('ok,ki,kj->koji',writer,a,b))/2
        actual=float(((dense(a,b,writer)-target).square().sum()+.01*component.square().sum())/total)
        rows.append(dict(seed=seed,fd_error=float(fd),dense_objective_replay=abs(actual-report['final_loss']),optimization=report))
    predictions=dict(pred_a_instrument=all(r['fd_error']<=1e-6 and r['dense_objective_replay']<=1e-8 and
        r['optimization']['maximum_accepted_loss_increase']<=1e-10 for r in rows),
        pred_b_convergence=all(r['optimization']['converged'] for r in rows))
    with out.open('x') as f:json.dump(dict(predictions=predictions,rows=rows,scope='Planted local integration; not native or global recovery.'),f,indent=2);f.write('\n')
    print(json.dumps(dict(predictions=predictions,rows=[dict(seed=r['seed'],fd=r['fd_error'],replay=r['dense_objective_replay'],iterations=r['optimization']['iterations'],converged=r['optimization']['converged']) for r in rows])))
    assert all(predictions.values())


if __name__=='__main__':main()
