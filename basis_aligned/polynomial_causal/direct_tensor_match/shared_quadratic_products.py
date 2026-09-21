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
