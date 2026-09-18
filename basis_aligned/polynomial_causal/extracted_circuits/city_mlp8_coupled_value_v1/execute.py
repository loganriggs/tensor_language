"""Coupled direct-attention plus MLP8 value correction with generated RMS9."""
import torch
import torch.nn.functional as F
from mlp8_value_norm_closed_v1 import prepare as prepare_norm, execute as execute_norm

def prepare(program):
    return prepare_norm(program)

def execute(program,z,delta,token_ids,return_rho=False):
    mlp8,rho=execute_norm(program,z,delta,token_ids,return_rho=True)
    direct=(1-program['mixture'].double())*program['lambdas9'].double()[0]/rho*F.linear(delta.double(),program['value_reader'].double())
    value=direct+mlp8
    return (value,rho) if return_rho else value
