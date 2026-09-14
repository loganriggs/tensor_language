"""Weight-derived spherical MLP-mean carrier; no fitted background or native-isotropy claim.

Board preregistration10:44UTC: replay<=1e-6, price<=1%extra,
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


@torch.no_grad()
def main():
    out=P/'SPHERICAL_CARRIER_CORRECTION_V1_RESULT.json'
    artifact=P/'SPHERICAL_CARRIER_CORRECTION_V1_PROGRAM.pt'
    assert not out.exists() and not artifact.exists()
    torch.set_num_threads(2);signal.alarm(120);start=time.monotonic();model=Contexts()
    means=[]
    for layer in range(17):
        left,right,down=[model.sd[f'transformer.h.{layer}.mlp.{name}.weight'].double() for name in ('Left','Right','Down')]
        means.append(down@(left*right).sum(1)+model.sd[f'transformer.h.{layer}.mlp.Down_bias'].double())
    b=torch.zeros(1152,dtype=torch.float64)
    for layer in range(17):
        b=model.sd[f'transformer.h.{layer}.lambdas'][0].double()*b+means[layer]
    b=model.sd['transformer.h.17.lambdas'][0].double()*b
    expanded=torch.zeros_like(b)
    for layer in range(17):
        scale=torch.ones((),dtype=torch.float64)
        for later in range(layer+1,18):scale*=model.sd[f'transformer.h.{later}.lambdas'][0].double()
        expanded+=scale*means[layer]
    recurrence=float((b-expanded).norm()/expanded.norm().clamp_min(1e-30))
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
                recurrence_replay=recurrence,bias_site_execution_replay=replay,price_ratio=price,
                artifact_bytes=artifact.stat().st_size,artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                correction_values=1536,extra_macs_per_row=1536,panels=panels,
                pred_a=max(recurrence,replay)<=1e-6,pred_b=price<=1.01,
                pred_c=all(p['relative_improvement']>=.05 for p in panels),seconds=time.monotonic()-start,
                scope='Weight-derived isotropic spherical MLP-mean carrier, not actual native mean; no fit to native data. Sparse body plus small dense correction; native conditional raw scores only, no adoption.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    signal.alarm(0);print(json.dumps(result))


if __name__=='__main__':main()
