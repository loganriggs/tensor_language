#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_factored_replay pred_b_twenty_percent_saving pred_c_frozen_validation
"""Calibrated mean-write diagnostic for failed isotropic radial replacement.

Compare joint full-write atom pruning, parent-reader weighted atom pruning and
matched-budget random supports. Each retained product is evaluated once and
feeds all parent reader features. Preserve actual attention17 and normalization.
Budgets0/256/1024/2304/4608. Same opened native intervention panels as v629.
4608 calibration product means; no model updates; supports remain weight-defined.16forwards/256replays. Not optimal sparse DAG.
"""
import json,os,sys,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3];BQ=ROOT/'basis_aligned/bilinear_quotient'
OUT=BQ/'circuits/followups/recent_radial_feature_v635_result.json'
PREDICTIONS={'pred_a_factored_replay':'<=3e-5','pred_b_twenty_percent_saving':'<=.8 dense values',
             'pred_c_frozen_validation':'<=.10 scalar and changes'}


def main():
    plan=dict(panels=['fineweb_n96_skip1200.pt','fineweb_n192_skip11000.pt'],ranks=[0,256,1024,2304,4608],
        families=['global_centered','parent_centered','random_products'],scales=[1.,0.,.5,1.5],
        documents=8,tokens=64,forwards_max=16,model_backwards=0,model_updates=0,fit_parameters=4608,
        execution_policy='managed_queue_only')
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(plan));return
    import torch
    import disk_guard
    import circuit_fast_screen_producer as producer
    sys.path.insert(0,str(ROOT/'basis_aligned/polynomial_causal'))
    from recent_folded_component import extract,execute
    from joint_atom_pruning import atom_gram,greedy_removal_order,removal_energy
    tic=time.perf_counter();torch.set_grad_enabled(False);torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    p=torch.load(OUT.with_name('output_component_rank_v621_program.pt'),map_location='cuda',weights_only=True)
    model=producer.Bilin18TorchBackend.load('cuda').model;w=extract(model,p);b16,b17=model.transformer.h[16:18]
    reader=model.lm_head.weight.float().T@p['vocabulary_writer'];bias=b17.mlp.Down_bias.float()@reader
    batches=[];forwards=0
    for panel in plan['panels']:
        ids=torch.load(BQ/'.rowcache'/panel,map_location='cpu',weights_only=True)[:8,:65].cuda()
        for start in range(0,8,4):
            ports=None;targets={};conditional={}
            for scale in plan['scales']:
                cache={}
                def m16(_m,args,out):cache['write16']=out.float();return out*scale
                def e16(_m,args,out):cache['after16']=out[0].float()
                def s17(_m,args):cache.update(v1=args[1].float(),x0=args[2].float())
                def m17(_m,args,out):cache.update(n=args[0].float(),out=out.float())
                hooks=[b16.mlp.register_forward_hook(m16),b16.register_forward_hook(e16),
                    b17.register_forward_pre_hook(s17),b17.mlp.register_forward_hook(m17)]
                try:model(ids[start:start+4,:-1],ids[start:start+4,1:].contiguous());forwards+=1
                finally:
                    for hook in hooks:hook.remove()
                if ports is None:ports=(cache['after16']-cache['write16'],cache['x0'],cache['v1'])
                targets[scale]=cache['out']@reader-bias
                conditional[scale]=(cache['n']@p['readers']).square()@p['coefficients']
            batches.append(dict(panel=panel,ports=ports,targets=targets,conditional=conditional))
    L,R,D=w['left16'],w['right16'],w['down16'];h,d=L.shape
    global_gram=atom_gram(D,L,R)
    # Weight each upstream quadratic reader by sqrt(abs(parent coefficient)).
    # This is a declared quadratic dictionary metric, not quartic or behavioral error.
    C=(w['readers']*w['coefficients'].abs().sqrt()[None,:]).T@D
    parent_gram=atom_gram(C,L,R)
    trace=(L*R).sum(-1)
    import torch.nn.functional as F
    calibration_n=torch.cat([F.rms_norm(b['ports'][0],(d,)).reshape(-1,d) for b in batches if b['panel']==plan['panels'][0]])
    empirical=((calibration_n@L.T)*(calibration_n@R.T)).mean(0)
    mean_audit=dict(isotropic_norm=float((D@trace).norm()),empirical_norm=float((D@empirical).norm()),
        relative_mean_error=float((D@(trace-empirical)).norm()/(D@empirical).norm()),
        calibration_tokens=len(calibration_n))
    global_centered=global_gram-(D.T@D)*torch.outer(trace,trace)/d
    parent_centered=parent_gram-(C.T@C)*torch.outer(trace,trace)/d
    orders=dict(global_centered=greedy_removal_order(global_centered),
        parent_centered=greedy_removal_order(parent_centered),
        random_products=torch.randperm(h,generator=torch.Generator().manual_seed(632)).to(L.device))
    supports={}
    def build(family,rank):
        if family=='dense':return w
        kept=orders[family][-rank:] if rank else orders[family][:0]
        supports[family+':'+str(rank)]=kept.cpu().tolist()
        mask=torch.ones(h,dtype=torch.bool,device=L.device);mask[kept]=False
        radial=(D[:,mask]@empirical[mask])/d
        return dict(w,left16=L[kept],right16=R[kept],down16=D[:,kept],radial16=radial)
    def size(q):return sum(v.numel() for v in q.values() if torch.is_tensor(v))
    configs=[('dense',h)]+[(family,rank) for family in plan['families'] for rank in plan['ranks']]
    rows=[];dense_replay=[];local_calls=0;full_rank_replay=[]
    for family,rank in configs:
        q=build(family,rank);stats={panel:{} for panel in plan['panels']}
        for batch in batches:
            base=None
            for scale in plan['scales']:
                alpha=execute(q,*batch['ports'],scale)['alpha'];local_calls+=1
                truth=batch['targets'][scale]
                if scale==1.:base=alpha
                a=stats[batch['panel']].setdefault(scale,dict(error=0.,target=0.,change_error=0.,change=0.))
                a['error']+=float((alpha-truth).double().square().sum());a['target']+=float(truth.double().square().sum())
                delta=truth-batch['targets'][1.]
                a['change_error']+=float((alpha-base-delta).double().square().sum());a['change']+=float(delta.double().square().sum())
                if rank==h:full_rank_replay.append(float((alpha-batch['conditional'][scale]).norm()/batch['conditional'][scale].norm()))
                if family=='dense':dense_replay.append(float((alpha-batch['conditional'][scale]).norm()/batch['conditional'][scale].norm()))
        panels={panel:{str(scale):dict(relative_error=(a['error']/a['target'])**.5,
            relative_change_error=(a['change_error']/a['change'])**.5 if a['change'] else None,
            squared_error=a['error'],target_squared_norm=a['target'],change_squared_error=a['change_error'],change_squared_norm=a['change'])
            for scale,a in values.items()} for panel,values in stats.items()}
        maximum=lambda panel:max(max(a['relative_error'],a['relative_change_error'] or 0.) for a in panels[panel].values())
        row=dict(family=family,rank=rank,stored_values=size(q),fraction_of_dense=size(q)/size(w),panels=panels,
                 calibration_max_error=maximum(plan['panels'][0]),validation_max_error=maximum(plan['panels'][1]))
        if family!='dense':
            kept=orders[family][-rank:] if rank else orders[family][:0]
            row['isotropic_support_global_error']=float((removal_energy(global_centered,kept).clamp_min(0)/global_gram.sum()).sqrt())
            row['isotropic_support_parent_error']=float((removal_energy(parent_centered,kept).clamp_min(0)/parent_gram.sum()).sqrt())
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='panels'}),flush=True)
    selected=min((r for r in rows if r['calibration_max_error']<=.1),key=lambda r:r['stored_values'])
    result=dict(plan=plan,rows=rows,selected={k:v for k,v in selected.items() if k!='panels'},
        forwards=forwards,local_replays=local_calls,maximum_dense_replay=max(dense_replay),maximum_full_rank_replay=max(full_rank_replay),
        supports=supports,mean_audit=mean_audit,
        predictions={'pred_a_factored_replay':max(full_rank_replay)<=3e-5,
            'pred_b_twenty_percent_saving':selected['fraction_of_dense']<=.8,
            'pred_c_frozen_validation':selected['fraction_of_dense']<=.8 and selected['validation_max_error']<=.1},
        scope='Calibration-fitted mean replacement diagnostic; supports fixed by weight metric; not weights-only discovery; opened panels',
        seconds=time.perf_counter()-tic,finished_utc=datetime.now(timezone.utc).isoformat())
    assert forwards<=plan['forwards_max']
    disk_guard.guard_write(1000000,label='v635 JSON');OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['selected'],indent=2));print(result['predictions'])


if __name__=='__main__':main()
