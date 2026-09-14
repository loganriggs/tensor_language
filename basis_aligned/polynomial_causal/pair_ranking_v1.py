"""Fixed8192-context paired candidate ranking; preregistered on AGENT_BOARD.
A Gram/direct<=1e-10. B shared file costs within1% of sparse.
C both shared candidates worse than sparse by>3pairedSE. No adaptive extension.
"""
import os,json,time,signal,hashlib
from datetime import datetime,timezone
from pathlib import Path
import torch
from retained_objective_context_v1 import Contexts,branch_errors,P

NAMES = ['SPARSE_INTERACTION_EXECUTOR_V1','INTERACTION_SHARED_WRITE_POLISH_V1','INTERACTION_BALANCED_SUBSPACES_V1']


@torch.no_grad()
def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(2)
    signal.alarm(120)
    started=time.perf_counter()
    model=Contexts()
    errors=[model.decode(name)-model.tensor for name in NAMES]
    values=[[] for _ in NAMES]; contrasts=[[] for _ in NAMES]
    replay=0.
    for seed in range(170216000,170216128):
        z,terms,denominator=model.sample(seed)
        for index,error in enumerate(errors):
            branch=branch_errors(error,z,terms,denominator)
            direct=torch.einsum('oih,ni,nh->no',error,z,terms.sum(1))/denominator[:,None]
            replay=max(replay,float((branch.sum(1)-direct).norm()/direct.norm()))
            gram=torch.einsum('nko,nlo->nkl',branch,branch)
            energy=direct.square().sum(-1)
            replay=max(replay,float((gram.sum((1,2))-energy).norm()/energy.norm()))
            values[index].append(energy)
            contrasts[index].append((direct[:,::2]-direct[:,1::2]).square().sum(-1)/2)
    values=[torch.cat(v) for v in values]; contrasts=[torch.cat(v) for v in contrasts]
    rows=[]
    for index,name in enumerate(NAMES):
        path=P/(name+'_PROGRAM.pt');delta=values[index]-values[0]
        se=float(delta.std()/len(delta)**.5)
        cd=contrasts[index]-contrasts[0]
        rows.append(dict(name=name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            coefficient_error=float(errors[index].norm()/model.tensor.norm()),mean_energy=float(values[index].mean()),
            paired_difference=float(delta.mean()),paired_standard_error=se,
            independent_half_differences=[float(v.mean()) for v in delta.chunk(2)],
            contrast_mean_energy=float(contrasts[index].mean()),contrast_paired_difference=float(cd.mean()),
            contrast_paired_standard_error=float(cd.std()/len(cd)**.5)))
    result=dict(utc=datetime.now(timezone.utc).isoformat(),contexts=8192,seeds=[170216000,170216127],candidates=rows,
        replay_relative_error=replay,pred_a=replay<=1e-10,
        pred_b=all(abs(r['bytes']/rows[0]['bytes']-1)<=.01 for r in rows[1:]),
        pred_c=all(r['paired_difference']>3*r['paired_standard_error'] for r in rows[1:]),
        seconds=time.perf_counter()-started,
        scope='Frozen candidates, identical synthetic normalized producer contexts, conditional final RMS. Paired token differences before softcap; no text, fitting, native behavior or adoption claim.')
    signal.alarm(0)
    return result


if __name__=='__main__':
    target=P/'PAIR_RANKING_V1_RESULT.json';assert not target.exists()
    result=run()
    with target.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
