"""Native behavior under ordinary versus mutually constrained MLP1 reader edits.

A: hashes/count18, zero/joint full-logit replay abs1e-3 AND relative1e-5,
local FP64 algebra1e-9, deployed FP32 reader abs1e-3/relative1e-5, hooks.
B: each dual own signed effect >=90% ordinary, ordinary native projection>=.01.
C: cross/own margin RMS<=.1, P KL<=ordinary+1e-6 and zero top1 flips.
18 forwards/576 sequences; zero fit/update; no native weight or background savings.
Null prevents promotion; fixed MLP1/readers/data, no rank/site/dose rescue.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_own_effect_retained pred_c_behavior_selective
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import run_bilin18_mlp1_joint_reader_weight_v1 as N
from circuit_fast_screen_managed_runner import atomic_create_json

RUNNER=Path(__file__).resolve();POLY=N.POLY;sys.path.insert(0,str(POLY))
import dual_reader_output_intervention as edit
import dual_reader_edit_reference as dual
V=N.pooled.v1;atlas=N.pooled.atlasrun;das=V.das
PRIOR=POLY/'BILIN18_MLP1_DUAL_READER_NATIVE_V1_PREREGISTRATION.md'
ROWS=POLY/'BILIN18_MLP1_DUAL_READER_NATIVE_V1_ROWS.json'
ROW_AUDIT=POLY/'BILIN18_MLP1_DUAL_READER_NATIVE_V1_ROW_AUDIT.json'
OUT=POLY/'BILIN18_MLP1_DUAL_READER_NATIVE_V1_RESULT.json'
FILES={'prior':PRIOR,'rows':ROWS,'row_audit':ROW_AUDIT,'readers':N.READERS,
       'feasibility':POLY/'BILIN18_MLP1_DUAL_READER_EDIT_V1_RESULT.json',
       'primitive':Path(edit.__file__),'dual_reference':Path(dual.__file__),
       'parent_runner':Path(N.__file__),'parent_sources':N.BINDING,
       'capture':Path(V.__file__),'atlas_capture':Path(atlas.__file__),'head':Path(das.__file__)}
EXPECTED={'prior': 'c6e05c42943e11f4e4b4ce1a0f4c9152fa7654411e7c8e819d6de0e16960862a', 'rows': '69fea2a04edd4e6d7108814745e872f8d7a8893647c588c76dfcd151f998c4b6', 'row_audit': 'e24d13589f35ffa0e9605ddb64583a011f96e8adab68d61b1086ed210711e2e6', 'readers': '2d26a6cb4487597000e65cceb9cf62145fdd68ce856e064c2169b912b9d603fc', 'feasibility': '0b53f05a510d446f4bc69475b49510a1e29e58aefced0caff5f000cf88e9e02c', 'primitive': '3eda16a07c33c826dd36b3229425840b1c42f6140ee26560e44a4d2cd6888eb6', 'dual_reference': 'ac91e0dcebbb5177200623ff41ff818b1b4072fdf38d6908009d14c5c7ea9333', 'parent_runner': 'cd9b4efd8ffe9fa8d4b76d2c1113e5de496c695dc20bb9313ac58dc3680abdf2', 'parent_sources': 'd5ee3731c07fe85525c4b69eaa8a7d88cf1d29b8e889e79be30602af96609460', 'capture': '8843cbb584d87823aedc346361b3f62679e06b8f31901b057b80b4d6b403cad3', 'atlas_capture': '6e28d38ec1446eafb3518c1bfe603a5e3469ceadb6f80266e2c695a274692366', 'head': '49d67620b09c80edd1c999476ea9cfddb375f41016443f58cb6cc96111809d3f'}
ARMS=('zero','ordinary_A','ordinary_B','dual_A','dual_B','dual_AB','joint_orthogonal')


def value_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def agree(x,y,atol=1e-3,rtol=1e-5):
    error=x.double()-y.double();absolute=float(error.abs().max());den=float(y.double().norm())
    relative=float(error.norm())/max(den,1e-30)
    return {'max_abs':absolute,'relative_frobenius':relative,'passed':bool(x.isfinite().all() and y.isfinite().all()) and absolute<=atol and relative<=rtol}


def flatten_valid(x,stops):return __import__('torch').cat([x[i,:n] for i,n in enumerate(stops)])


def main():
    observed={k:N.sha(p) for k,p in FILES.items()};assert observed==EXPECTED
    authorities=json.loads(N.BINDING.read_text());assert {p:N.sha(p) for p in authorities}==authorities
    manifest=json.loads(ROWS.read_text());ra=json.loads(ROW_AUDIT.read_text())
    assert {p:N.sha(p) for p in manifest['source_sha256']}==manifest['source_sha256']
    rows={k:manifest[k] for k in ('targets','controls')}
    assert {k:value_sha(v) for k,v in rows.items()}==ra['row_sha256']==manifest['row_sha256']
    assert len(rows['targets'])==48 and len(rows['controls'])==16
    for r in rows['targets']+rows['controls']:
        assert len(r['base_ids'])==len(r['donor_ids'])
        assert r['base_semantic_position']==r['donor_semantic_position']==len(r['base_ids'])-1
    feasibility=json.loads(FILES['feasibility'].read_text());assert all(feasibility['predictions'].values())
    dry={'dryrun':True,'model_loaded':False,'gpu_accessed':False,'model_forwards':18,'sequence_evaluations':576,
         'arms':ARMS,'row_sha256':ra['row_sha256'],'model_updates':0,'fit_updates':0}
    if os.environ.get('BQLIB_DRYRUN')=='1' or os.environ.get('BQLIB_NO_MODEL')=='1':print(json.dumps(dry));return
    assert not OUT.exists();signal.alarm(600);tic=time.perf_counter()
    backend=N.producer.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    module=backend.model.transformer.h[1].mlp;controls=edit.controls();assert controls['passed']
    saved=json.loads(N.READERS.read_text());c=torch.cat([torch.tensor(saved['physical_readers'][task],dtype=torch.float64,device='cuda').T for task in N.TASKS])
    d=torch.linalg.solve(c@c.T,c).T
    ordinary=torch.cat([torch.linalg.solve(ct@ct.T,ct).T for ct in (c[:4],c[4:])],dim=1)
    q=torch.linalg.qr(c.T,mode='reduced').Q
    specs={'zero':(c,d,[]),'ordinary_A':(c,ordinary,list(range(4))),
           'ordinary_B':(c,ordinary,list(range(4,8))),'dual_A':(c,d,list(range(4))),
           'dual_B':(c,d,list(range(4,8))),'dual_AB':(c,d,list(range(8))),
           'joint_orthogonal':(q.T,q,list(range(8)))}
    count=[0];sequence_count=[0]
    def counted(_m,args,_out):count[0]+=1;sequence_count[0]+=int(args[0].shape[0])
    counter=module.register_forward_hook(counted);replay=[];local=[];algebra=[];finite=True;reports={};restored=True
    with torch.inference_mode():
        try:
            for population,rs in rows.items():
                batch,bo,bc,do,dc=V.cap(backend,rs)
                stops=[int(p)+1 for p in batch.semantic_positions];donor=dc['mlp'][1].to('cuda')
                logits={}
                for name,out in [('base',bo),('donor',do)]:
                    state=atlas.states(torch,backend,out,rs);logits[name]=das.head_logits(backend,state).double()
                for name in ARMS:
                    cc,dd,selected=specs[name];captures=[]
                    with edit.at_module(module,donor,cc,dd,selected,stops,captures):out=backend.native(batch,capture=True)
                    assert len(captures)==1
                    before,after=captures[0]
                    if name in ('dual_A','dual_B','dual_AB'):
                        held=[i for i in range(8) if i not in selected]
                        bf=flatten_valid(before,stops).double();af=flatten_valid(after,stops).double();target=flatten_valid(donor,stops).double()
                        check={'population':population,'arm':name,'selected':agree(af@c[selected].T,target@c[selected].T)}
                        if held:
                            error=float(((af-bf)@c[held].T).abs().max());check['other_reader_max_abs']=error;check['other_reader_passed']=error<=1e-3
                        else:check['other_reader_passed']=True
                        local.append(check)
                        ideal=edit.apply(before.double(),donor.double(),cc,dd,selected,stops)
                        algebra.append(agree(flatten_valid(ideal,stops)@c[selected].T,target@c[selected].T,1e-9,1e-9))
                    if name=='dual_AB':
                        first=edit.apply(before.double(),donor.double(),c,d,list(range(4)),stops)
                        ab=edit.apply(first,donor.double(),c,d,list(range(4,8)),stops)
                        first_b=edit.apply(before.double(),donor.double(),c,d,list(range(4,8)),stops)
                        ba=edit.apply(first_b,donor.double(),c,d,list(range(4)),stops)
                        together=edit.apply(before.double(),donor.double(),c,d,list(range(8)),stops)
                        algebra.extend([agree(ab,together,1e-9,1e-9),agree(ba,together,1e-9,1e-9)])
                    restored=restored and len(module._forward_hooks)==1
                    state=atlas.states(torch,backend,out,rs);logits[name]=das.head_logits(backend,state).double()
                replay.extend([agree(logits['zero'],logits['base']),agree(logits['dual_AB'],logits['joint_orthogonal'])])
                finite=finite and all(bool(z.isfinite().all()) for z in logits.values())
                indices=torch.arange(len(rs),device='cuda');answers=torch.tensor([r['donor_answer_id'] for r in rs],device='cuda');foils=torch.tensor([r['donor_foil_id'] for r in rs],device='cuda')
                margins={name:z[indices,answers]-z[indices,foils] for name,z in logits.items()}
                full=margins['donor']-margins['base'];changes={name:margins[name]-margins['base'] for name in ARMS}
                base_lp=logits['base'].log_softmax(-1);kl={name:(base_lp.exp()*(base_lp-logits[name].log_softmax(-1))).sum(-1) for name in ARMS}
                flips={name:logits[name].argmax(-1)!=logits['base'].argmax(-1) for name in ARMS}
                masks={'ALL':torch.ones(len(rs),dtype=torch.bool,device='cuda')}
                if population=='targets':masks.update({'temporal':torch.tensor([i<24 for i in range(48)],device='cuda'),'iswas':torch.tensor([i>=24 for i in range(48)],device='cuda')})
                panels={}
                for group,mask in masks.items():
                    reference=full[mask];den=float(reference.square().sum());record={}
                    for name in ARMS:
                        delta=changes[name][mask]
                        record[name]={'signed_native_projection':float(delta@reference)/den if den>0 else None,
                          'margin_change_rms':float(delta.square().mean().sqrt()),'mean_kl':float(kl[name][mask].mean()),
                          'max_kl':float(kl[name][mask].max()),'top1_flips':int(flips[name][mask].sum())}
                    mixed=logits['dual_AB'][mask]-logits['dual_A'][mask]-logits['dual_B'][mask]+logits['base'][mask]
                    panels[group]={'native_margin_change_rms':float(reference.square().mean().sqrt()),'arms':record,
                                   'joint_logit_interaction_rms':float(mixed.square().mean().sqrt()),'rows':int(mask.sum())}
                reports[population]={'panels':panels,'rows':[{'row_id':r['row_id'],'construction_id':r.get('construction_id'),'native_margin_change':float(full[i]),
                    'native_base_top1':int(logits['base'][i].argmax()),'native_donor_top1':int(logits['donor'][i].argmax()),
                    'margin_changes':{name:float(changes[name][i]) for name in ARMS},'kl':{name:float(kl[name][i]) for name in ARMS},
                    'top1_flips':{name:bool(flips[name][i]) for name in ARMS}} for i,r in enumerate(rs)]}
                print(json.dumps({'population':population,'panels':panels,'forwards':count[0]}),flush=True)
        finally:counter.remove()
    a=finite and restored and len(module._forward_hooks)==0 and count[0]==18 and sequence_count[0]==576 and all(z['passed'] for z in replay+algebra) and all(z['selected']['passed'] and z['other_reader_passed'] for z in local)
    b=True;cross=True;decisions={}
    for task,own,other in [('temporal','A','B'),('iswas','B','A')]:
        arms=reports['targets']['panels'][task]['arms'];ordinary_score=arms['ordinary_'+own]['signed_native_projection'];dual_score=arms['dual_'+own]['signed_native_projection']
        own_rms=arms['dual_'+own]['margin_change_rms'];cross_rms=arms['dual_'+other]['margin_change_rms']
        retained=ordinary_score is not None and dual_score is not None and ordinary_score>=.01 and dual_score>=.9*ordinary_score
        selective=own_rms>0 and cross_rms<=.1*own_rms
        b=b and retained;cross=cross and selective
        decisions[task]={'ordinary_signed_effect':ordinary_score,'dual_signed_effect':dual_score,'own_retained':retained,
                         'cross_to_own_rms':cross_rms/own_rms if own_rms>0 else None,'cross_selective':selective}
    ctl=reports['controls']['panels']['ALL']['arms']
    control_ok=all(ctl['dual_'+task]['mean_kl']<=ctl['ordinary_'+task]['mean_kl']+1e-6 and ctl['dual_'+task]['top1_flips']==0 for task in ('A','B'))
    result={'terminal':'invalid' if not a else 'dual_reader_selective_screen_pass' if b and cross and control_ok else 'local_reader_separation_not_behavior_selectivity',
        'predictions':{'pred_a_instrument':a,'pred_b_own_effect_retained':b,'pred_c_behavior_selective':cross and control_ok},
        'decisions':decisions,'control_selective':control_ok,'replay_audits':replay,'local_reader_audits':local,'algebra_audits':algebra,'controls':controls,
        'reports':reports,'authority_sha256':observed,'runner_sha256':N.sha(RUNNER),
        'price':{'model_forwards':count[0],'sequence_evaluations':sequence_count[0],'model_updates':0,'fit_updates':0,
                 'reader_scalars':c.numel(),'writer_scalars':d.numel(),'native_parameters':sum(p.numel() for p in backend.model.parameters()),'adoption':False},
        'scope':'Previously opened fit-disjoint target text, temporal P controls only; local readout constraints are not semantic circuit identification.',
        'wall_seconds':time.perf_counter()-tic}
    atomic_create_json(OUT,result)
    print(json.dumps({k:result[k] for k in ('terminal','predictions','decisions','control_selective','price','wall_seconds')},indent=2))

if __name__=='__main__':main()
