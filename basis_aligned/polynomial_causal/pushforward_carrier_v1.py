"""Actual-weight pushforward synthetic mean carrier; no native-data-fitted background.

Board preregistration10:45UTC: replay<=1e-6, price<=1%extra,
>=5%normalizedrawerror improvement on both historical and fresh native panels.
"""
import hashlib
import json
import signal
import time
from datetime import datetime, timezone
import torch
from retained_objective_context_v1 import Contexts,P
from sparse_interaction_executor_v1 import Executor
from coupled_writer_tail_control_v1 import native_blocks
import torch.nn.functional as F


@torch.no_grad()
def main():
    out=P/'PUSHFORWARD_CARRIER_V1_RESULT.json'
    artifact=P/'PUSHFORWARD_CARRIER_V1_PROGRAM.pt'
    assert not out.exists() and not artifact.exists()
    torch.set_num_threads(2);signal.alarm(120);start=time.monotonic();model=Contexts()
    blocks=native_blocks(model.sd,0,18)
    samples=[]
    for seed in range(170232000,170232016):
        generator=torch.Generator().manual_seed(seed)
        ids=torch.randint(50304,(16,16),generator=generator)
        initial=F.rms_norm(model.sd['transformer.wte.weight'][ids].float(),(1152,))
        x=initial;inherited=None
        for layer,block in blocks.items():
            raw=block.lambdas[0]*x+block.lambdas[1]*initial
            attention,inherited=block.attn(F.rms_norm(raw,(1152,)),inherited)
            x=raw+attention
            if layer<17:x=x+block.mlp(F.rms_norm(x,(1152,)))
        samples.append(x[:,-1].double())
    samples=torch.cat(samples)
    b=samples.mean(0)
    recurrence=float((samples[:128].mean(0)-samples[128:].mean(0)).norm()/b.norm())
    approximate=model.decode('SPARSE_INTERACTION_EXECUTOR_V1')
    correction=torch.einsum('oih,i->oh',model.tensor-approximate,b)
    package=torch.load(P/'SPARSE_INTERACTION_EXECUTOR_V1_PROGRAM.pt',weights_only=True)
    changed=dict(package,correction=correction.float())
    torch.save(changed,artifact)  # Freeze the weight-derived artifact before native evaluation.
    executor=Executor(package)
    a=torch.eye(128,dtype=torch.float32)
    actual=executor(b.float().expand(128,-1),a)+a@changed['correction'].T
    expected=torch.einsum('oih,i,nh->no',model.tensor,b,a.double())
    replay=float((actual.double()-expected).norm()/expected.norm().clamp_min(1e-30))
    torch.save(dict(carrier=b,seed_range=[170232000,170232015],half_mean_disagreement=recurrence),P/'PUSHFORWARD_CARRIER_V1_PROVENANCE.pt')
    panels=[]
    for prefix,sl in [('COMPOSED_LAST_BLOCK_STATES_V1',slice(None)),('MINIMAX_FRESH_CACHE_V1',slice(24,None))]:
        path=P/(prefix+'_ARTIFACT.pt')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==json.loads((P/(prefix+'_RESULT.json')).read_text())['artifact_sha']
        cache=torch.load(path,weights_only=True)
        z=cache['linear_parts'][sl,3].double();state=cache['linear_parts'][sl,2].double()
        heads=torch.linalg.lstsq(model.writer,(state-z).T).solution.T
        eps=torch.finfo(torch.float32).eps
        denom=(state.square().mean(-1)+eps)*(cache['states'][sl,2].square().mean(-1)+eps).sqrt()
        baseline=torch.einsum('oih,ni,nh->no',approximate-model.tensor,z,heads)/denom[:,None]
        repaired=baseline+heads@changed['correction'].double().T/denom[:,None]
        panels.append(dict(prefix=prefix,baseline_energy=float(baseline.square().sum()),
                           corrected_energy=float(repaired.square().sum()),
                           relative_improvement=float(1-repaired.square().sum()/baseline.square().sum()),
                           background_rms=float(z.square().mean().sqrt()),
                           centered_background_rms=float((z-b).square().mean().sqrt())))
    price=artifact.stat().st_size/(P/'SPARSE_INTERACTION_EXECUTOR_V1_PROGRAM.pt').stat().st_size
    result=dict(utc=datetime.now(timezone.utc).isoformat(),bias_rms=float(b.square().mean().sqrt()),
                half_mean_disagreement=recurrence,bias_site_execution_replay=replay,price_ratio=price,
                artifact_bytes=artifact.stat().st_size,artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                correction_values=1536,extra_macs_per_row=1536,panels=panels,
                pred_a=recurrence<=.05 and replay<=1e-6,pred_b=price<=1.01,
                pred_c=all(p['relative_improvement']>=.05 for p in panels),seconds=time.monotonic()-start,
                scope='Weights-generated uniform-token pushforward mean, not a native mean claim; no fit to native data. Sparse body plus small dense correction; native conditional raw scores only, no adoption.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    signal.alarm(0);print(json.dumps(result))


if __name__=='__main__':main()
