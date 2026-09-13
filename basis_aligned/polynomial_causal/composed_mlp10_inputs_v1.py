"""MLP9 responses through attention10, generating both MLP10 edit inputs."""
import torch
from mlp9_to_mlp10_residual_fold_v1 import execute as residual_fold
from raw_attention_response_v1 import prepare, changed, execute as attention


def execute(z9, biasfree_mlp9, raw10, pristine_z10, first_values, a, b,
            response_program, reentry_scale, attention_matrices,
            attention_mixture, attention_output):
    residuals = residual_fold(z9, biasfree_mlp9, a, b, response_program, reentry_scale)
    projections,rho = prepare(raw10,attention_matrices)
    baseline = attention(projections,rho,first_values,attention_mixture,attention_output)
    partners=[]
    for delta in residuals:
        ports,newrho = changed(raw10,delta,projections,attention_matrices)
        partners.append(attention(ports,newrho,first_values,attention_mixture,attention_output)-baseline)
    c,r = [delta+partner for delta,partner in zip(residuals,partners)]
    joint_rho=(pristine_z10.double()+c+r).square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps
    return dict(child=c,remainder=r,child_residual=residuals[0],remainder_residual=residuals[1],
                child_attention=partners[0],remainder_attention=partners[1],joint_rho=joint_rho)
