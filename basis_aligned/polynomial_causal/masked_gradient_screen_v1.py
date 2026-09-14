"""Full-support gradient discriminator, frozen on AGENT_BOARD 2026-09-14 09:56.
Two2048-context gradients, exact first-panel line solve,8192fresh validation.
No adaptive extension or native-text objective. CPU cap120seconds.
"""
import os,json,time,signal,hashlib
from datetime import datetime,timezone
import numpy as np
import torch
from retained_objective_context_v1 import Contexts,branch_errors,P
from sparse_interaction_executor_v1 import Executor


@torch.no_grad()
def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(2);signal.alarm(120);started=time.perf_counter()
    model=Contexts();p=torch.load(P/'SPARSE_INTERACTION_EXECUTOR_V1_PROGRAM.pt',weights_only=True)
    output,head=p['output'].double(),p['head'].double()
    mask=torch.from_numpy(np.unpackbits(p['mask'].numpy(),bitorder='little',count=model.tensor.numel()).copy()).bool()
    original=model.decode('SPARSE_INTERACTION_EXECUTOR_V1');error=original-model.tensor
    gradients=[]
    for first_seed in (170220000,170221000):
        gradient=torch.zeros_like(model.tensor)
        for seed in range(first_seed,first_seed+32):
            z,terms,denominator=model.sample(seed)
            a=terms.sum(1)
            e=branch_errors(error,z,terms,denominator).sum(1)
            gradient+=2*torch.einsum('no,ni,nh->oih',e@output,z,(a@head)/denominator[:,None])/2048
        gradients.append(gradient.flatten()[mask])
    cosine=float(torch.nn.functional.cosine_similarity(*gradients,dim=0))
    values=-gradients[0]/gradients[0].norm()*(.1*model.tensor.norm())
    flat=torch.zeros(model.tensor.numel(),dtype=torch.float64);flat[mask]=values
    direction=torch.einsum('op,pih,ah->oia',output,flat.reshape(p['shape']),head)
    ed=dd=0.
    for seed in range(170220000,170220032):
        z,terms,denominator=model.sample(seed)
        e=branch_errors(error,z,terms,denominator).sum(1)
        d=branch_errors(direction,z,terms,denominator).sum(1)
        ed+=float((e*d).sum())/2048;dd+=float(d.square().sum())/2048
    gradient_replay=abs(float(gradients[0]@values)-2*ed)/max(abs(2*ed),1e-30)
    alpha=-ed/dd
    changed=dict(p,values=(p['values'].double()+alpha*values).float())
    path=P/'MASKED_GRADIENT_SCREEN_V1_PROGRAM.pt';assert not path.exists();torch.save(changed,path)
    fit=model.decode('MASKED_GRADIENT_SCREEN_V1');ferror=fit-model.tensor
    executor=Executor(changed);baseline=[];candidate=[];bc=[];cc=[];replay=0.
    for seed in range(170222000,170222128):
        z,terms,denominator=model.sample(seed)
        e=branch_errors(error,z,terms,denominator).sum(1)
        f=branch_errors(ferror,z,terms,denominator).sum(1)
        if seed==170222000:
            exact=torch.einsum('oih,ni,nh->no',fit,z,terms.sum(1))
            actual=executor(z.float(),terms.sum(1).float()).double()
            replay=float((exact-actual).norm()/exact.norm())
        baseline.append(e.square().sum(-1));candidate.append(f.square().sum(-1))
        bc.append((e[:,::2]-e[:,1::2]).square().sum(-1)/2)
        cc.append((f[:,::2]-f[:,1::2]).square().sum(-1)/2)
    baseline,candidate,bc,cc=map(torch.cat,(baseline,candidate,bc,cc))
    gain=baseline-candidate;se=float(gain.std()/len(gain)**.5)
    improvement=float(gain.mean()/baseline.mean());contrast_ratio=float(cc.mean()/bc.mean())
    price_ratio=path.stat().st_size/(P/'SPARSE_INTERACTION_EXECUTOR_V1_PROGRAM.pt').stat().st_size
    invariant=all(torch.equal(changed[k],p[k]) for k in ('mask','output','head'))
    result=dict(utc=datetime.now(timezone.utc).isoformat(),train_contexts_per_gradient=2048,validation_contexts=8192,
        train_seed_ranges=[[170220000,170220031],[170221000,170221031]],validation_seed_range=[170222000,170222127],
        gradient_cosine=cosine,gradient_norms=[float(g.norm()) for g in gradients],
        gradient_cross_inner_product=float(gradients[0]@gradients[1]),gradient_directional_replay=gradient_replay,
        alpha=alpha,relative_energy_improvement=improvement,paired_gain=float(gain.mean()),paired_standard_error=se,
        original_energy=float(baseline.mean()),candidate_energy=float(candidate.mean()),contrast_energy_ratio=contrast_ratio,
        runtime_replay=replay,price_ratio=price_ratio,artifact_bytes=path.stat().st_size,
        artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),mask_and_adapters_unchanged=invariant,
        pred_a=max(replay,gradient_replay)<=1e-6 and invariant and abs(price_ratio-1)<=.01,
        pred_b=improvement>=.05 and float(gain.mean())>3*se,
        pred_c=cosine>=.5 and contrast_ratio<=1.01,seconds=time.perf_counter()-started,
        scope='One fixed-support gradient step with exact sampled line solve; synthetic normalized producer measure, no native intervention law, no full convergence or adoption.')
    signal.alarm(0);return result


if __name__=='__main__':
    target=P/'MASKED_GRADIENT_SCREEN_V1_RESULT.json';assert not target.exists()
    result=run()
    with target.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
