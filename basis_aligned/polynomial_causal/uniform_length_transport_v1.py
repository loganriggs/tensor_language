"""Preregistered frozen-program length5/32/128 transport,16contexts perlength."""
import json
import signal
import time
from datetime import datetime, timezone
import torch
from uniform_producer_context_v1 import UniformProducer
from retained_objective_context_v1 import P, branch_errors


@torch.no_grad()
def main():
    out=P/'UNIFORM_LENGTH_TRANSPORT_V1_RESULT.json';assert not out.exists()
    signal.alarm(120);torch.set_num_threads(2);start=time.monotonic()
    model=UniformProducer()
    errors=[model.decode(name)-model.tensor for name in
            ('SPARSE_INTERACTION_EXECUTOR_V1','UNIFORM_PRODUCER_GRADIENT_V1')]
    rows=[]
    for length in (5,32,128):
        batches=[];totals=torch.zeros(2,dtype=torch.float64)
        contrasts=torch.zeros(2,dtype=torch.float64);reference=0.
        for seed in range(170229000,170229004):
            z,terms,denom,diagnostics=model.sample(seed,batch=4,native_site=True,sequence_length=length)
            value=[branch_errors(e,z,terms,denom).sum(1) for e in errors]
            energy=torch.stack([v.square().sum() for v in value])
            totals+=energy
            contrasts+=torch.stack([(v[:,::2]-v[:,1::2]).square().sum()/2 for v in value])
            reference+=float(branch_errors(model.tensor,z,terms,denom).sum(1).square().sum())
            batches.append(dict(seed=seed,energies=energy.tolist(),diagnostics=diagnostics))
        rows.append(dict(length=length,batches=batches,energies=totals.tolist(),
                         fit_over_baseline=float(totals[1]/totals[0]),
                         contrast_fit_over_baseline=float(contrasts[1]/contrasts[0]),
                         own_mixed_relative_errors=(totals/reference).sqrt().tolist()))
        print(json.dumps(rows[-1]),flush=True)
    result=dict(utc=datetime.now(timezone.utc).isoformat(),rows=rows,
                pred_a=all(b['diagnostics']['finite'] and b['diagnostics']['producer_replay']<=1e-10 for r in rows for b in r['batches']),
                pred_b=all(r['fit_over_baseline']<=.95 for r in rows),
                seconds=time.monotonic()-start,scope='Small descriptive length-transport screen, uniform tokens, no fit or native-text/OOD transfer.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    signal.alarm(0);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
