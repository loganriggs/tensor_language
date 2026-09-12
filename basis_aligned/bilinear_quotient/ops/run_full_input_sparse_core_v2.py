#!/usr/bin/env python3
# BQGATE:0bodyforwards,0textsequences;2full-input spectral cores;900sec alarm.
"""pred_a identities<=1e-8; pred_b producer4096capture>=1.25*identity;
pred_c producer4096capture>=.5. Fixed frames, no global sparsity/circuit claim.
"""
import os, sys, json, signal, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
P=ROOT/'basis_aligned/polynomial_causal'
sys.path.insert(0,str(P))
import torch
from sparse_path_stability_atlas_v1 import digest
from quadratic_producer_projection_v1 import atom_gram
from fullu_input_blocks_v1 import sandwich
from full_input_sparse_core_v1 import edge_energies, control
STEM='FULL_INPUT_SPARSE_CORE_V2'

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
    assert all(digest(k)==v for k,v in binding.items())
    checked=control();assert checked['passed']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(body_forwards=0,text_sequences=0,control=checked)));return
    out=P/(STEM+'_RESULT.json');assert not out.exists()
    signal.alarm(900);tic=time.perf_counter();torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False
    sd=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True)
    u=sd['lm_head.weight'].double().cuda();u-=u.mean(0)
    metric=u.T@u;root=torch.linalg.cholesky(metric).T;del u
    l0,r0,d0,l1,r1,d1=[sd[f'transformer.h.{layer}.mlp.{name}.weight'].double().cuda() for layer in (16,17) for name in ('Left','Right','Down')]
    scale=float(sd['transformer.h.17.lambdas'][0])
    h=scale**2*d0@atom_gram(l0,r0)@d0.T
    he,hv=torch.linalg.eigh(h);assert float(he.min())>0
    hs=(hv*he.sqrt())@hv.T;eye=torch.eye(1152,device='cuda')
    g=d1.T@metric@d1;writer=root@d1
    prior=json.loads((P/'FULLU_PAIRED_PRODUCER_V1_RESULT.json').read_text())['reports']
    reports=[];errors=list(checked['errors'])
    for name,transform in [('centered_identity',eye),('centered_producer',hs)]:
        l,r=l1@transform,r1@transform
        k=sandwich(l,r,g,eye);k=(k+k.T)/2
        ev,q=torch.linalg.eigh(k);q=q.flip(1)
        total=float(torch.trace(k));energy,edges=edge_energies(l,r,writer,q)
        errors.extend([abs(float(energy.sum())/total-1),float((q.T@q-eye).abs().max()),
                       abs(total/next(v['total_paired_coefficient_energy'] for v in prior if v['name']==name)-1)])
        order=energy.argsort(descending=True);cumulative=energy[order].cumsum(0)/total
        restricted=(edges[0]<128)&(edges[1]<128)
        re=energy[restricted].sort(descending=True).values
        report=dict(name=name,capture={str(n):float(cumulative[n-1]) for n in (256,1024,4096)},
                    edges90=int(torch.searchsorted(cumulative,.9))+1,
                    restricted128_total=float(re.sum()/total),restricted128_top256=float(re[:256].sum()/total),
                    total=total,top4096_edges=edges[:,order[:4096]].cpu().tolist(),
                    active_readers4096=int(torch.unique(edges[:,order[:4096]]).numel()))
        reports.append(report);print(json.dumps({k:v for k,v in report.items() if k!='top4096_edges'}),flush=True)
    result={'pred_a':max(errors)<=1e-8,'pred_b':reports[1]['capture']['4096']>=1.25*reports[0]['capture']['4096'],
                'pred_c':reports[1]['capture']['4096']>=.5}
    result.update(reports=reports,identity_errors=errors,
                execution_seconds=time.perf_counter()-tic,source_shas=binding,
                scope='Fixed complete spectral frames; paired coefficients; no behavioral validation or global optimality.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('reports','source_shas')}),flush=True)

if __name__=='__main__':main()
