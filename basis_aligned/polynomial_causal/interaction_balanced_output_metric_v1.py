"""Weights-only paired metric: exact identity, decoder folding, one update."""
import json, math, time
from pathlib import Path
import torch
from head17_output_block_objective_v1 import build
from interaction_shared_write_subspaces_v1 import assign, update, reconstruction
P = Path(__file__).resolve().parent

def main():
    torch.set_num_threads(2)
    start = time.perf_counter()
    t, ids = build()
    x = t.permute(1, 2, 0).reshape(-1, 12).contiguous()
    program = torch.load(P/'INTERACTION_SHARED_WRITE_POLISH_V1_PROGRAM.pt', weights_only=True)
    assert ids == program['token_ids']
    plus = torch.kron(torch.eye(6, dtype=x.dtype), torch.ones(2, 2, dtype=x.dtype)/2)
    minus = torch.eye(12, dtype=x.dtype)-plus
    ep, em = (x@plus).square().sum(), (x@minus).square().sum()
    gamma = (ep/em).sqrt()
    w, wi = plus+gamma*minus, plus+minus/gamma
    old = reconstruction(program['groups'].long(), program['codes'].double(), program['output_bases'].double())
    error = old-x
    balanced = ((error@w).square().sum()/(x@w).square().sum()).item()
    explicit = .5*((error@plus).square().sum()/ep+(error@minus).square().sum()/em)
    identity = abs(balanced-float(explicit))
    assert identity < 1e-12
    y = x@w
    q = torch.linalg.qr(w@program['output_bases'].double(), mode='reduced')[0]
    labels, codes = assign(y, q)
    initial = float((y-reconstruction(labels,codes,q)).square().sum()/y.square().sum())
    q = update(y, labels, q)
    labels, codes = assign(y, q)
    fit = reconstruction(labels,codes,q)
    final = float((y-fit).square().sum()/y.square().sum())
    assert final <= initial+1e-12
    decoder = wi@q
    decoded = reconstruction(labels,codes,decoder)
    replay = float((decoded-fit@wi).norm()/(fit@wi).norm())
    assert replay < 1e-12
    err = decoded-x
    result = dict(gamma=float(gamma), balanced_identity_error=identity,
        old_balanced_relative_error=balanced**.5, initial_squared_loss=initial,
        one_update_squared_loss=final, decoder_replay_error=replay,
        one_update_total_error=float(err.norm()/x.norm()),
        one_update_sum_error=float((err@plus).norm()/(x@plus).norm()),
        one_update_difference_error=float((err@minus).norm()/(x@minus).norm()),
        nominal_fp32_program_bytes=4*(147456*8+32*12*8)+147456,
        gpca_degree32_ambient12_monomials=math.comb(43,11),
        seconds=time.perf_counter()-start,
        scope='One conditional update, not convergence or native validation. Pair-selected weights-only metric; inverse transform absorbed in existing decoder. No additional runtime adapter.')
    (P/'INTERACTION_BALANCED_OUTPUT_METRIC_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__ == '__main__': main()
