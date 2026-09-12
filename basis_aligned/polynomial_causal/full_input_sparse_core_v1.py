"""Stream exact orthonormal symmetric-coordinate energies; no basis optimization."""
import torch


@torch.no_grad()
def edge_energies(left, right, writer, frame, chunk=256):
    a, b = left @ frame, right @ frame
    i, j = torch.triu_indices(frame.shape[1], frame.shape[1], device=frame.device)
    energy = torch.empty(len(i), dtype=left.dtype, device=left.device)
    for start in range(0, len(i), chunk):
        ii, jj = i[start:start+chunk], j[start:start+chunk]
        denominator = torch.where(ii == jj, 2., 2.**.5)
        columns = writer @ ((a[:, ii]*b[:, jj] + a[:, jj]*b[:, ii])/denominator)
        energy[start:start+chunk] = columns.square().sum(0)
    return energy, torch.stack((i, j))


def control():
    torch.set_default_dtype(torch.float64)
    gen = torch.Generator().manual_seed(120451)
    l, r = [torch.randn(9, 7, generator=gen) for _ in range(2)]
    w = torch.randn(5, 9, generator=gen)
    q = torch.linalg.qr(torch.randn(7, 7, generator=gen)).Q
    s = torch.einsum('oj,ja,jb->oab', w, l, r)
    s = (s+s.transpose(-1,-2))/2
    transformed = torch.einsum('ai,oab,bj->oij', q, s, q)
    energy, edges = edge_energies(l, r, w, q, 3)
    i, j = edges
    direct = transformed[:, i, j].square().sum(0)*torch.where(i == j, 1., 2.)
    errors = [float((energy-direct).norm()/direct.norm()),
              abs(float(energy.sum()/s.square().sum())-1)]
    return {'errors': errors, 'passed': max(errors) < 1e-12}


if __name__ == '__main__':
    import json
    print(json.dumps(control()))
