"""Spectral outer-quadratic approximation in the exact paired producer metric.
This objective preserves an ordered pairing of the four input slots, before
full quartic symmetrization. No Gaussian-output assumption or language data.
"""
import torch

def factor(matrix, metric, rank, cutoff=1e-12):
    values, vectors = torch.linalg.eigh(metric)
    keep = values > values.max() * cutoff
    frame = vectors[:, keep]
    scale = values[keep].sqrt()
    weighted = (scale[:, None] * (frame.T @ matrix @ frame)) * scale[None, :]
    eig, vec = torch.linalg.eigh((weighted + weighted.T) / 2)
    selected = eig.abs().argsort(descending=True)[:rank]
    readers = (frame / scale[None, :]) @ vec[:, selected]
    retained = eig[selected]
    error = float((eig.square().sum() - retained.square().sum()).clamp_min(0).sqrt() / eig.norm())
    return readers, retained, dict(metric_rank=int(keep.sum()),
        metric_condition=float(values[keep].max()/values[keep].min()),
        paired_coefficient_relative_error=error)
