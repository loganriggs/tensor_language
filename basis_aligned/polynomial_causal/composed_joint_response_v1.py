"""Three shared generated branches, retaining inherited mixed response."""
import torch
from response_product_basis_v2 import coefficients
from response_attention_projection_v2 import changed
from raw_attention_response_v1 import execute as attention, EPS


def branch(amplitude, context):
    u = coefficients(amplitude, context['response'])
    residual = torch.einsum('...k,...kd->...d', u, context['basis'])
    ports, rho = changed(amplitude, context['projections'])
    partner = attention(ports, rho, context['first_values'], context['mixture'],
                        context['output']) - context['baseline_attention']
    return context['pristine_z10'] + residual + partner


def block_output(z, left, right, down, bias):
    rho = z.square().mean(-1, keepdim=True) + EPS
    return z + ((z @ left.T) * (z @ right.T)) @ down.T / rho + bias


def evaluate(a, b, context, left, right, down, bias):
    """Return all four post-MLP10 states; downstream suffix is not included."""
    states = {'native': context['pristine_z10'], 'child': branch(a, context),
              'remainder': branch(b, context), 'parent': branch(a+b, context)}
    outputs = {key: block_output(z, left, right, down, bias)
               for key, z in states.items()}
    outputs['mixed'] = (outputs['parent'] - outputs['child']
                        - outputs['remainder'] + outputs['native'])
    return outputs
