"""Apply extracted operational features to an explicitly supplied native background.

The caller supplies normalized MLP16 input, actual MLP17 input BEFORE RMSNorm,
and the residual AFTER block17. Final RMSNorm and softcap are applied afterwards
by the caller. This module neither recomputes nor changes attention/cross terms.
"""
import torch
from extract_scalar_modes import evaluate


def remove_modes(final_residual, mlp17_prenorm, mlp16_normalized, artifact, strengths):
    """Subtract modes with continuous strengths (...,4); zero leaves state unchanged.

    All artifact tensors must already share the inputs' device/dtype. Multiple
    edits on a common background can be composed by adding strength vectors.
    Nonlinear downstream output changes must still be evaluated jointly.
    """
    features=evaluate(artifact['program'],mlp16_normalized)
    if strengths.shape[-1] != features.shape[-1]:
        raise ValueError('One strength is required for each extracted feature')
    if final_residual.shape != mlp17_prenorm.shape:
        raise ValueError('Final and pre-normalization residual shapes must match')
    eps=torch.finfo(mlp17_prenorm.dtype).eps
    denominator=mlp17_prenorm.square().mean(-1,keepdim=True)+eps
    edit=(features*strengths)@artifact['residual_writer'].T
    return final_residual-edit/denominator
