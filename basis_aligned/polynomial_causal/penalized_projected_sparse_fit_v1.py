"""Thin adapter for the existing projected L-BFGS controller; positive penalty."""
import time
import numpy as np
import torch
from projected_sparse_dictionary_fit_v1 import ProjectedObjective, fit
from penalized_projected_sparse_v1 import value_gradient


class PenalizedObjective(ProjectedObjective):
    def __init__(self,native,whitener,raw,indices,values,total,penalty):
        super().__init__(native,whitener,raw,indices,values,torch.ones(len(values),device=values.device),total)
        if penalty<=0:raise ValueError('Positive component penalty required')
        self.penalty=penalty

    def evaluate(self,point):
        tic=time.perf_counter();raw,values=self.unpack(point)
        loss,gradient,writer,details=value_gradient(self.native,self.whitener,raw,self.indices,values,self.total,self.penalty)
        assert details['output_solve']['normal_residual']<=1e-8
        grad=torch.cat((gradient[0].flatten()*self.scales[0],gradient[1].flatten()*self.scales[1]))
        assert bool(torch.isfinite(loss)) and bool(torch.isfinite(grad).all())
        value=float(loss);array=grad.cpu().numpy().copy()
        self.evaluations+=1;self.evaluation_seconds.append(time.perf_counter()-tic)
        self.last=dict(point=np.array(point,copy=True),value=value,gradient=array,writer=writer,details=details)
        return value,array
