"""Existing shared head component at a raw query/key/value projection interface."""
import itertools
import torch


def execute(qa, qb, ka, kb, value, rotation, program, reader_coefficients,
            reader_correction=None):
    reads = torch.cat((ka, kb, value), dim=-1) @ reader_coefficients.T
    if reader_correction is not None:
        reads = reads + reader_correction
    z = (reads[:, 0]*reads[:, 1])[:, None]*reads[:, 2:]
    dual = z @ program['dual']
    a = torch.einsum('nk,kl,ril->nri', qa, rotation, program['atom_k1'])
    b = torch.einsum('nk,kl,ril->nri', qb, rotation, program['atom_k2'])
    eps = torch.finfo(torch.float32).eps
    gate = 1/(qa.shape[-1]**2*((qa.square().mean(-1)+eps)
        *(qb.square().mean(-1)+eps)*(ka.square().mean(-1)+eps)
        *(kb.square().mean(-1)+eps)).sqrt())
    result = qa.new_zeros(qa.shape[0], program['atom_write'].shape[-1])
    for i, j, k in itertools.permutations(range(3)):
        result += torch.einsum('nr,nr,nr,ro->no', dual, a[:, :, i],
                               b[:, :, j], program['atom_write'][:, k])
    return gate[:, None]*result
