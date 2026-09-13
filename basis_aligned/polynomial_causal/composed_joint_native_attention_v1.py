"""Fold the fixed-writer MLP9 response; keep native attention10 execution."""
import torch
import torch.nn.functional as F
from directional_mlp_response_context_v1 import prepare as response_prepare, evaluate


def prepare(z9, biasfree_mlp9, raw10, first_values, program, scale, native_attention):
    return dict(response=response_prepare(z9,biasfree_mlp9,program),raw=raw10,
                first_values=first_values,scale=scale,attention=native_attention)


def branch(amplitude,context):
    raw=(context['raw'].double()+context['scale']*evaluate(amplitude,context['response'])).float()
    first=context['first_values'].expand(raw.shape[0],*context['first_values'].shape[1:])
    return raw+context['attention'](F.rms_norm(raw,(raw.shape[-1],)),first)[0]
