"""Preserve the original tiny-fixture gradient miss; audit a direct residual solve."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from folded_sparse_dictionary_v1 import loss_gradient
P=Path(__file__).resolve().parent


def objective(c):
    square=1+c[0]*c[2];mixed=c[1]*c[3]
    return square*square+.5*mixed*mixed,np.array([2*square*c[2],mixed*c[3],2*square*c[0],mixed*c[1]])


def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    parent=json.loads((P/'FOLDED_SPARSE_DICTIONARY_V1_CONTROL.json').read_text())
    initial=np.array(parent['cancellation']['optimized_values']);value,gradient=objective(initial)
    a=torch.tensor([[1.,0.],[1.,1.]]);b=torch.tensor([[0.,1.],[1.,0.]])
    w=torch.tensor([[1.,-1.]])
    old,(_,g),_=loss_gradient((a,b,w),w,torch.eye(2),torch.tensor([[0],[1],[0],[0]]),
        torch.from_numpy(initial).reshape(4,1),torch.ones(4),1.,chunk=2)
    loss_replay=abs(value-float(old));gradient_replay=float(np.max(np.abs(gradient-g.flatten().numpy())))
    fitted=minimize(objective,initial,jac=True,method='L-BFGS-B',
        options=dict(maxiter=1000,ftol=0.,gtol=1e-11,maxls=40))
    final,grad=objective(fitted.x)
    result=dict(predictions=dict(pred_a_replay=loss_replay<=1e-14 and gradient_replay<=1e-10,
        pred_b_recovery=final<=1e-10 and float(np.max(np.abs(grad)))<=1e-8),
        initial_loss=value,initial_gradient_max=float(np.max(np.abs(gradient))),
        loss_replay_absolute=loss_replay,gradient_replay_absolute=gradient_replay,
        final_loss=float(final),final_gradient_max=float(np.max(np.abs(grad))),
        fitted_values=fitted.x.tolist(),iterations=int(fitted.nit),solver_success=bool(fitted.success),
        solver_message=str(fitted.message),original_prediction_c=parent['predictions']['pred_c_functional_recovery'],
        scope='Same known four-code fixture; direct residual formula and tighter function-change stop. Original gradient miss preserved; not a native or global convergence result.')
    with (P/'FOLDED_SPARSE_CANCELLATION_POLISH_V1_AUDIT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2));assert all(result['predictions'].values())


if __name__=='__main__':main()
