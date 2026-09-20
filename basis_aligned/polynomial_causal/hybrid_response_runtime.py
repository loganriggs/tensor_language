"""Conditional output predictor: nonlinear number plus linear modal responses.

Prepared native contexts, reader fields and full source deltas are required.
This does not produce a replacement residual state or generate its own ports.
"""
import torch
from conditional_attention_mlp_runtime import execute as run_dag


def fold_readers(readers, decoder, encoder):
    result = readers.clone()
    result[0] = readers[0] - (readers[0] @ decoder) @ encoder.T
    return result


def execute(runtime, encoder, case, delta):
    logits, _ = run_dag(runtime, case['attention'], delta @ encoder,
                         case['contexts'], case['final_context'], case['positions'])
    linear = torch.einsum('rbtd,btd->br', case['readers'], delta)
    number = case['baseline_margin'] - (logits[:, 0]-logits[:, 1])*case['direction']
    number = number - linear[:, 0]*case['direction']
    return torch.cat([number[:, None], -linear[:, 1:]], dim=1)
