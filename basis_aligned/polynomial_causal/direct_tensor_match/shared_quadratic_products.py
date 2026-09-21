"""Implicit symmetric CP loss for multiple quadratic outputs sharing products.

Teacher shape [outputs, inputs, inputs], reader [inputs, products],
weights [outputs, products]. Output matrices are sum_r weights[o,r] v_r v_r^T.
Metric geometry and per-output weights must be folded into the teacher first.
"""
import torch

def coefficient_loss(teacher,reader,weights):
    gram=reader.T@reader
    student_norm=((weights.T@weights)*gram.square()).sum()
    cross=(weights*torch.einsum('dr,ode,er->or',reader,teacher,reader)).sum()
    teacher_norm=teacher.square().sum()
    return (teacher_norm+student_norm-2*cross)/teacher_norm

def materialize(reader,weights):
    return torch.einsum('or,dr,er->ode',weights,reader,reader)

def mixed_coefficient_loss(teacher,left,right,weights):
    """Symmetric quadratic coefficient error for products (left*x)(right*x)."""
    gram=.5*((left.T@left)*(right.T@right)+(left.T@right)*(right.T@left))
    student_norm=((weights.T@weights)*gram).sum()
    cross=(weights*torch.einsum('ir,oij,jr->or',left,teacher,right)).sum()
    norm=teacher.square().sum()
    return (norm+student_norm-2*cross)/norm

def materialize_mixed(left,right,weights):
    raw=torch.einsum('or,ir,jr->oij',weights,left,right)
    return .5*(raw+raw.transpose(-1,-2))
