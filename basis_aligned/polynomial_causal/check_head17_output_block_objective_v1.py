"""Actual-weight gradient and reconstruction controls for learned output blocks."""
from pathlib import Path
import json,time
import numpy as np
import torch
from head17_output_block_objective_v1 import build,Objective
P=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2);torch.manual_seed(358);tic=time.perf_counter()
    T,ids=build();obj=Objective(T,32);base=torch.eye(12,dtype=torch.float64)
    theta=torch.randn(66,dtype=torch.float64).numpy()*.1
    value,gradient,intrinsic=obj.value_gradient(theta,base)
    rng=np.random.default_rng(358);errors=[]
    for _ in range(3):
        d=rng.normal(size=66);d/=np.linalg.norm(d);eps=1e-5
        plus=obj.value_gradient(theta+eps*d,base)[0]
        minus=obj.value_gradient(theta-eps*d,base)[0]
        fd=(plus-minus)/(2*eps);analytic=float(gradient@d)
        errors.append(abs(fd-analytic)/max(abs(fd),abs(analytic),1e-8))
    Q=obj.rotate(torch.tensor(theta),base);loss,gq,gn,blocks=obj.evaluate(Q)
    reconstruction=(Q@blocks.flatten(1)).reshape_as(T)
    replay=abs(float((reconstruction-obj.tensor).square().sum())-loss)
    assert max(errors)<1e-4 and replay<1e-12
    result=dict(token_ids=ids,tensor_shape=list(T.shape),rank_per_block=32,
                normalized_squared_error=value,relative_error=value**.5,
                finite_difference_relative_errors=errors,reconstruction_loss_error=replay,
                orthogonality_error=float((Q.T@Q-torch.eye(12)).norm()),intrinsic_gradient_norm=intrinsic,
                mixed_representation_scalars=12*32*(1152+128)+144,
                dense_mixed_scalars=T.numel(),seconds=time.perf_counter()-tic,
                scope='Actual twelve-token mixed coefficient tensor; fixed block rank32, orthogonal output '
                'basis only. SVD block fits exact; envelope gradient and skew-chart derivative checked. '
                'No iterative fit/convergence result, native normalization or behavioral validation yet.')
    (P/'HEAD17_OUTPUT_BLOCK_OBJECTIVE_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
