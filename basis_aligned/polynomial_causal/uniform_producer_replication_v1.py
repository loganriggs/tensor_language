"""Frozen1024-context independent fit replication; see preregistration for bars."""
import hashlib
import json
import os
import signal
import time
from datetime import datetime, timezone
import torch
from uniform_producer_context_v1 import UniformProducer
from retained_objective_context_v1 import P, branch_errors
NAMES = ['SPARSE_INTERACTION_EXECUTOR_V1', 'UNIFORM_PRODUCER_GRADIENT_V1', 'INTERACTION_SHARED_WRITE_POLISH_V1']


@torch.no_grad()
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    out = P/'UNIFORM_PRODUCER_REPLICATION_V1_RESULT.json'
    assert not out.exists()
    torch.set_num_threads(2)
    signal.alarm(120)
    start = time.monotonic()
    assert hashlib.sha256((P/'UNIFORM_PRODUCER_GRADIENT_V1_PROGRAM.pt').read_bytes()).hexdigest() == 'bf5d29ca117cd126ab7c61910b2b1bac74d07874e1e94c36029a9b7ebbad58ed'
    model = UniformProducer()
    errors = [model.decode(name)-model.tensor for name in NAMES]
    energies = [[] for _ in NAMES]
    contrasts = [[] for _ in NAMES]
    reference, diagnostics = [], []
    replay = 0.
    for seed in range(170228000, 170228064):
        z, terms, denominator, diagnostic = model.sample(seed, native_site=True)
        diagnostics.append(dict(seed=seed, **diagnostic))
        ref = branch_errors(model.tensor, z, terms, denominator).sum(1)
        reference.append(ref.square().sum(-1))
        for i, error in enumerate(errors):
            branches = branch_errors(error, z, terms, denominator)
            direct = torch.einsum('oih,ni,nh->no', error, z, terms.sum(1))/denominator[:, None]
            gram = torch.einsum('nko,nlo->nkl', branches, branches).sum((1, 2))
            energy = direct.square().sum(-1)
            replay = max(replay, float((branches.sum(1)-direct).norm()/direct.norm().clamp_min(1e-30)),
                         float((gram-energy).norm()/energy.norm().clamp_min(1e-30)))
            energies[i].append(energy)
            contrasts[i].append((direct[:, ::2]-direct[:, 1::2]).square().sum(-1)/2)
        print(json.dumps(diagnostics[-1]), flush=True)
    energies = [torch.cat(v) for v in energies]
    contrasts = [torch.cat(v) for v in contrasts]
    ref = torch.cat(reference)
    rows = []
    for i, name in enumerate(NAMES):
        path = P/(name+'_PROGRAM.pt')
        difference = energies[i]-energies[0]
        cd = contrasts[i]-contrasts[0]
        rows.append(dict(name=name, bytes=path.stat().st_size,
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         mean_energy=float(energies[i].mean()),
                         largest_row_energy_fraction=float(energies[i].max()/energies[i].sum()),
                         top_one_percent_energy_fraction=float(energies[i].topk(11).values.sum()/energies[i].sum()),
                         relative_improvement=float((energies[0]-energies[i]).mean()/energies[0].mean()),
                         own_contribution_relative_error=float((energies[i].sum()/ref.sum()).sqrt()),
                         paired_difference=float(difference.mean()),
                         paired_standard_error=float(difference.std()/32),
                         independent_half_differences=[float(v.mean()) for v in difference.chunk(2)],
                         contrast_mean_energy=float(contrasts[i].mean()),
                         contrast_paired_difference=float(cd.mean()),
                         contrast_paired_standard_error=float(cd.std()/32)))
    result = dict(utc=datetime.now(timezone.utc).isoformat(), contexts=1024,
                  candidates=rows, diagnostics=diagnostics, replay_relative_error=replay,
                  pred_a=replay<=1e-10 and all(d['finite'] and d['producer_replay']<=1e-10 for d in diagnostics),
                  pred_b=all(abs(r['bytes']/rows[0]['bytes']-1)<=.01 for r in rows[1:]),
                  pred_c=rows[1]['relative_improvement']>=.05 and
                         -rows[1]['paired_difference']>3*rows[1]['paired_standard_error'] and
                         max(rows[1]['independent_half_differences'])<0 and
                         rows[1]['contrast_mean_energy']<=1.01*rows[0]['contrast_mean_energy'],
                  seconds=time.monotonic()-start,
                  checkpoint_parameter_count=sum(v.numel() for v in model.sd.values()),
                  scope='Uniform independent checkpoint-token rows; actual hierarchy edits and native suffix. Conditional selected mixed operator, no fit, natural-text validation, compression adoption or full-body replacement.')
    with out.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    signal.alarm(0)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
