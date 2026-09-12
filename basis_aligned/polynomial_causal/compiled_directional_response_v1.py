"""Predict changed raw attention9 input from pristine ports and fixed-writer edit."""
import torch
EPS=torch.finfo(torch.float32).eps

def predict(z,raw9,biasfree_mlp8,amplitude,direction,gain,*,matrix=None,left=None,right=None):
 z=z.double();a=amplitude.double()[...,None];d=direction.double();rho=z.square().mean(-1,keepdim=True)+EPS;changed=z-a*d;rhom=changed.square().mean(-1,keepdim=True)+EPS;mid=z-a*d/2
 mixed=mid@matrix.T if matrix is not None else (mid@right.T)@left.T
 delta=-a*d+(rho/rhom-1)*biasfree_mlp8.double()-a/rhom*mixed
 return raw9.double()+gain*delta
