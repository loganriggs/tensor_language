"""Sampled-reader initialization: exact Lasso row bound and live-snapshot audit.
No live checkpoint, source, dictionary, code or heldout score is modified.
"""
import hashlib
import io
import json
from pathlib import Path
import time
import torch
from native_reader_msp_generalization_v1 import P,CK


@torch.no_grad()
def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    started=time.perf_counter();penalty=.05;bound=penalty-penalty**2/2
    gen=torch.Generator().manual_seed(956)
    atom=torch.randn(24,12,generator=gen);atom/=atom.norm(dim=1,keepdim=True)
    x=atom[0];z=torch.zeros(24);z[0]=1-penalty
    residual=z@atom-x;objective=float(.5*residual.square().sum()+penalty*z.abs().sum())
    grad=residual@atom.T
    kkt=torch.where(z.abs()>1e-12,(grad+penalty*z.sign()).abs(),(grad.abs()-penalty).clamp_min(0))
    random_codes=torch.randn(100,24,generator=gen)
    candidate=.5*(random_codes@atom-x).square().sum(1)+penalty*random_codes.abs().sum(1)
    control=dict(bound=bound,attained_objective=objective,equality_error=abs(objective-bound),
                 maximum_kkt=float(kkt.max()),sampled_objective_minimum=float(candidate.min()))
    # Writer atomically replaces its cache; read one inode into bytes, freeze those exact bytes.
    source=Path('/dev/shm/bilin18_overcomplete_l1_reader_v1_s0_fit.pt')
    payload=source.read_bytes();sha=hashlib.sha256(payload).hexdigest()
    frozen=Path('/dev/shm/bilin18_l1_reader_anchor_v1_snapshot.pt');assert not frozen.exists()
    frozen.write_bytes(payload)
    state=torch.load(io.BytesIO(payload),weights_only=True,map_location='cpu');del payload
    b=state['dictionary'].double();codes=state['codes'].double();initial=state['initial_dictionary'].double()
    assert b.shape==(2304,1152) and codes.shape==(6144,2304) and state['seed']==0 and state['penalty']==penalty
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    readers=torch.cat([sd[f'transformer.h.17.mlp.{key}.weight'].double() for key in ('Left','Right')])
    readers/=readers.norm(dim=1,keepdim=True)
    order=torch.randperm(4608,generator=torch.Generator().manual_seed(700))
    train=torch.cat((order[:3072],order[:3072]+4608));y=readers[train]
    selected=torch.randperm(6144,generator=torch.Generator().manual_seed(0))[:2304]
    init_error=float((initial-y[selected]).abs().max())
    fit=codes@b;error=(fit-y).square().sum(1);row_obj=.5*error+penalty*codes.abs().sum(1)
    latest=state['history'][-1];verified=latest['fp64_check']
    # A checkpoint may be between five-iteration checks; only replay if it records current diagnostics.
    replay=abs(float(row_obj.mean())-verified['objective']) if verified is not None else None
    energy=codes.square();atom_energy=energy.sum(0)
    effective=atom_energy.square()/energy.square().sum(0).clamp_min(1e-30)
    own=codes[selected,torch.arange(2304)].square()
    alignment=(b*initial).sum(1)/(b.norm(dim=1)*initial.norm(dim=1))
    mask=torch.zeros(6144,dtype=torch.bool);mask[selected]=True
    groups={}
    for name,group in [('initial_atom_readers',mask),('other_training_readers',~mask)]:
        groups[name]=dict(count=int(group.sum()),capture=1-float(error[group].mean()),
            mean_objective=float(row_obj[group].mean()),mean_objective_above_row_bound=float(row_obj[group].mean())-bound,
            median_active_codes=float((codes[group].abs()>1e-8).sum(1).double().median()))
    median=float(effective.median());own_share=float(own.sum()/energy.sum());cosine=float(alignment.median())
    output=dict(control=control,initial_reader_replay_error=init_error,objective_replay_error=replay,
        current_objective=float(row_obj.mean()),snapshot_iteration=latest['iteration'],snapshot_precision=state['precision'],
        last_recorded_check=state['checks'][-1] if state['checks'] else None,
        atom_usage=dict(median_squared_code_effective_readers=median,
            fraction_effective_readers_at_most_three=float((effective<=3).double().mean()),
            own_initial_reader_squared_code_share=own_share,median_initial_atom_cosine=cosine,
            fraction_initial_atom_cosine_over_point9=float((alignment>=.9).double().mean()),
            median_nonzero_reader_count=float((codes.abs()>1e-8).sum(0).double().median())),
        groups=groups,predictions=dict(
            pred_a_instrument=max(control['equality_error'],control['maximum_kkt'],init_error)<=1e-10 and
                (replay is None or replay<=1e-8) and control['sampled_objective_minimum']>=bound,
            pred_b_concentrated_usage=median<=3,pred_c_own_reader_share=own_share>=.5,
            pred_d_initial_alignment=cosine>=.9),
        cache=dict(path=str(frozen),sha256=sha,bytes=frozen.stat().st_size,live_source=str(source)),
        seconds=time.perf_counter()-started,
        scope='Intermediate immutable training-only snapshot, not final fit or heldout validation. Squared code contributions are not additive folded function energy. Universal row bound follows triangle inequality and unit atom norms.')
    with (P/'L1_READER_ANCHOR_V1_AUDIT.json').open('x') as f:json.dump(output,f,indent=2);f.write('\n')
    print(json.dumps(output),flush=True)


if __name__=='__main__':main()
