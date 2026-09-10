#!/usr/bin/env python3
# BQGATE: five frozen compiled correspondence checks;28forwards448seq, no fitting.
"""pred_a native/capture<=1e-3abs/1e-5rel; pred_b scalar scalederror<=1;
pred_c donor/mean fullvocab<=1e-3abs/1e-5rel; pred_d margin<=.002;
pred_e28forwards448seq/14blocks26ports. Conditional native background fully charged.
"""
import json,os,sys,time
from pathlib import Path
import torch
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(POLY),str(ROOT)]
import circuit_unit_greedy as g
import circuit_fast_screen_producer as P
import circuit_fast_screen_candidate_correlative_pair as A
import circuit_fast_screen_candidate_correlative_disjoint_either_not as D
import correlative_native_folded_executor_v1 as C
from induction_context_transport_v2 import digest
from circuit_fast_screen_managed_runner import atomic_create_json
BINDING=POLY/'CORRELATIVE_FOLDED_PROGRAM_V1_BINDING.json';OUT=POLY/'CORRELATIVE_FOLDED_PROGRAM_V1_RESULT.json'


def bridge(a,b):
    a=a.double();b=b.double()
    return {'max_abs':float((a-b).abs().max()),'relative_l2':float((a-b).norm()/b.norm().clamp_min(1e-30))}


def main():
    binding=json.loads(BINDING.read_text());assert all(digest(p)==h for p,h in binding.items())
    held=lambda rows:rows[2::4]+rows[3::4]
    panels={'A1':held(g.rows_of(A,'A1')),'A2':held(g.rows_of(A,'A2')),
            'P':held(g.rows_of(A,'P')),'C':held(g.rows_of(D,'A1'))}
    assert all(len(rows)==16 for rows in panels.values())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'model_loaded':False,'gpu_accessed':False,'forwards':28,'sequences':448,'rows':{k:len(v) for k,v in panels.items()}}));return
    assert not OUT.exists();start=time.perf_counter();backend=P.Bilin18TorchBackend.load('cuda')
    artifact=torch.load(POLY/'CORRELATIVE_REPLAYABLE_INTERFACE_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')
    blocks=torch.load(POLY/'CORRELATIVE_WEIGHT_PORTS_V1_MAPS.pt',weights_only=True,map_location='cpu')
    q={key:value.cuda() for key,value in artifact['q'].items()};units=artifact['units']
    counts=[0,0]
    def count(_m,args):
        counts[0]+=1;counts[1]+=args[0].shape[0]
        if counts[0]>28 or counts[1]>448:raise RuntimeError('Frozen native execution cap')
    counter=backend.model.transformer.h[0].attn.register_forward_pre_hook(count)
    reports={};finite=True;scalar_collections=0
    with torch.inference_mode():
        try:
            for name,rows in panels.items():
                base=g.batch_of(rows,'base');donor=g.batch_of(rows,'donor')
                da,dz,dc=C.forward(backend,donor,blocks)
                ba,bz,bc=C.forward(backend,base,blocks)
                na,nz=g.forward_units(backend,base,return_logits=True)
                ra,rz=g.forward_units(backend,base,units=units,donor_cache=dc['cache'],q=q,return_logits=True)
                ca,cz,cc=C.forward(backend,base,blocks,dc['scalars'])
                mean_cache={(rid,u):artifact['mu'][u] for rid in base.row_ids for u in units}
                ma,mz=g.forward_units(backend,base,units=units,donor_cache=mean_cache,q=q,return_logits=True)
                mean_scalars={}
                for layer,block in blocks.items():
                    concat=torch.cat([artifact['mu'][u].float() for u in block['units']])
                    mean_scalars[layer]=(concat@artifact['q'][(layer,'heads')][:,0]).double().cuda().repeat(16)
                fa,fz,fc=C.forward(backend,base,blocks,mean_scalars)
                bridges={'native':bridge(bz,nz),'donor':bridge(cz,rz),'mean':bridge(fz,mz)}
                scalar_errors={context:{str(layer):value for layer,value in cap['errors'].items()}
                               for context,cap in [('donor',dc),('base',bc),('donor_edit',cc),('mean_edit',fc)]}
                scalar_collections+=sum(len(v) for v in scalar_errors.values())
                margins={'donor':float(((ca[:,0]-ca[:,1])-(ra[:,0]-ra[:,1])).abs().max()),
                         'mean':float(((fa[:,0]-fa[:,1])-(ma[:,0]-ma[:,1])).abs().max())}
                tensors={'native':nz,'donor_native':dz,'donor_original':rz,'donor_folded':cz,'mean_original':mz,'mean_folded':fz}
                finite=finite and all(bool(torch.isfinite(z).all()) for z in tensors.values())
                targets=torch.tensor(base.answer_ids,device='cuda')
                ce={key:torch.nn.functional.cross_entropy(z.float(),targets,reduction='none').cpu().tolist()
                    for key,z in tensors.items() if key!='donor_native'}
                margin={key:(af[:,0]-af[:,1]).cpu().tolist() for key,af in [('native',na),('donor_original',ra),('donor_folded',ca),('mean_original',ma),('mean_folded',fa)]}
                reports[name]={'rows':rows,'bridges':bridges,'scalar_errors':scalar_errors,
                               'margin_difference_maxabs':margins,'base_answer_ce':ce,'base_oriented_margin':margin,
                               'donor_oriented_native_margin':(da[:,0]-da[:,1]).cpu().tolist()}
        finally:counter.remove()
    valid=lambda b:b['max_abs']<=1e-3 and b['relative_l2']<=1e-5
    predictions={'pred_a_native_bridge':all(valid(r['bridges']['native']) for r in reports.values()),
                 'pred_b_scalar_producer':all(e['scaled_error']<=1 for r in reports.values() for c in r['scalar_errors'].values() for e in c.values()),
                 'pred_c_intervention_logits':finite and all(valid(r['bridges'][arm]) for r in reports.values() for arm in ('donor','mean')),
                 'pred_d_intervention_margins':all(v<=.002 for r in reports.values() for v in r['margin_difference_maxabs'].values()),
                 'pred_e_counts':counts==[28,448] and scalar_collections==224 and len(blocks)==14 and sum(len(b['ports']) for b in blocks.values())==26}
    result={'schema':'correlative.folded_program.v1','predictions':predictions,'reports':reports,
            'price':{'body_forwards':counts[0],'sequences':counts[1],'core_scalars':76032,'native_parameters':sum(p.numel() for p in backend.model.parameters()),'native_weight_saving':0},
            'scalar_collections':scalar_collections,'binding_sha256':digest(BINDING),'runner_sha256':digest(RUNNER),
            'wall_seconds':time.perf_counter()-start,'scope':'Conditional route-read-write program with native normalized routing and all upstream/background retained; no standalone extraction or semantic promotion'}
    atomic_create_json(OUT,result)
    print(json.dumps({'predictions':predictions,'price':result['price'],'wall_seconds':result['wall_seconds'],
                      'bridges':{k:v['bridges'] for k,v in reports.items()},
                      'scalar_worst_scaled_error':max(e['scaled_error'] for r in reports.values() for c in r['scalar_errors'].values() for e in c.values())},indent=2))


if __name__=='__main__':main()
