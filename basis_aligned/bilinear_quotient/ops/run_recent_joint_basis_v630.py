#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_factored_replay pred_b_twenty_percent_saving pred_c_frozen_validation
"""Gauge audit and joint-tensor input-mode baseline for the recent path.

Same native targets, scalar/intervention metrics and opened panels as v629.
Compare raw-factor SVD, row-balanced SVD, and symmetric joint-tensor mode Gram.
Ranks256/512/768/1152; include full-rank recovery in every family. No fitting.
Select using calibration only. Dense fallback cannot pass compressed validation.
16 native forwards,208 local replays. No package export for a failed baseline.
"""
import json,os,sys,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3];BQ=ROOT/'basis_aligned/bilinear_quotient'
OUT=BQ/'circuits/followups/recent_joint_basis_v630_result.json'
PREDICTIONS={'pred_a_factored_replay':'<=3e-5','pred_b_twenty_percent_saving':'<=.8 dense values',
             'pred_c_frozen_validation':'<=.10 scalar and changes'}


def main():
    plan=dict(panels=['fineweb_n96_skip1200.pt','fineweb_n192_skip11000.pt'],ranks=[256,512,768,1152],
        families=['raw_input','balanced_input','joint_tensor_input'],scales=[1.,0.,.5,1.5],
        documents=8,tokens=64,forwards_max=16,model_backwards=0,model_updates=0,fit_parameters=0,
        execution_policy='managed_queue_only')
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(plan));return
    import torch
    import disk_guard
    import circuit_fast_screen_producer as producer
    sys.path.insert(0,str(ROOT/'basis_aligned/polynomial_causal'))
    from recent_folded_component import extract,execute
    from joint_folded_tucker import mode_grams
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
    _,_,raw_v=torch.linalg.svd(torch.cat([L,R]),full_matrices=False)
    balance=(R.norm(dim=1)/L.norm(dim=1).clamp_min(1e-30)).sqrt()
    _,_,balanced_v=torch.linalg.svd(torch.cat([balance[:,None]*L,R/balance[:,None]]),full_matrices=False)
    _,input_gram=mode_grams(D,L,R)
    eig,joint_basis=torch.linalg.eigh(input_gram)
    bases=dict(raw_input=raw_v.T,balanced_input=balanced_v.T,joint_tensor_input=joint_basis.flip(1))
    # Tensor Gram invariance under a deliberately broad, exact channel gauge.
    gauge=torch.logspace(-1,1,h,device=L.device)
    _,gauged_gram=mode_grams(D,L*gauge[:,None],R/gauge[:,None])
    gram_gauge_error=float((input_gram-gauged_gram).norm()/input_gram.norm())
    assert gram_gauge_error<1e-4
    def build(family,rank):
        if family=='dense':return w
        P=bases[family][:,:rank]
        return dict(w,input_basis16=P,left16=L@P,right16=R@P)
    def size(q):return sum(v.numel() for v in q.values() if torch.is_tensor(v))
    configs=[('dense',d)]+[(family,rank) for family in plan['families'] for rank in plan['ranks']]
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
                if rank==d:full_rank_replay.append(float((alpha-batch['conditional'][scale]).norm()/batch['conditional'][scale].norm()))
                if family=='dense':dense_replay.append(float((alpha-batch['conditional'][scale]).norm()/batch['conditional'][scale].norm()))
        panels={panel:{str(scale):dict(relative_error=(a['error']/a['target'])**.5,
            relative_change_error=(a['change_error']/a['change'])**.5 if a['change'] else None,
            squared_error=a['error'],target_squared_norm=a['target'],change_squared_error=a['change_error'],change_squared_norm=a['change'])
            for scale,a in values.items()} for panel,values in stats.items()}
        maximum=lambda panel:max(max(a['relative_error'],a['relative_change_error'] or 0.) for a in panels[panel].values())
        row=dict(family=family,rank=rank,stored_values=size(q),fraction_of_dense=size(q)/size(w),panels=panels,
                 calibration_max_error=maximum(plan['panels'][0]),validation_max_error=maximum(plan['panels'][1]))
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='panels'}),flush=True)
    selected=min((r for r in rows if r['calibration_max_error']<=.1),key=lambda r:r['stored_values'])
    result=dict(plan=plan,rows=rows,selected={k:v for k,v in selected.items() if k!='panels'},
        forwards=forwards,local_replays=local_calls,maximum_dense_replay=max(dense_replay),maximum_full_rank_replay=max(full_rank_replay),
        gram_gauge_error=gram_gauge_error,balance_range=[float(balance.min()),float(balance.max())],
        predictions={'pred_a_factored_replay':max(full_rank_replay)<=3e-5 and gram_gauge_error<1e-4,
            'pred_b_twenty_percent_saving':selected['fraction_of_dense']<=.8,
            'pred_c_frozen_validation':selected['fraction_of_dense']<=.8 and selected['validation_max_error']<=.1},
        scope='Joint-tensor input-mode projection baseline, not optimized Tucker core or sparse DAG; opened panels; upstream ports required',
        seconds=time.perf_counter()-tic,finished_utc=datetime.now(timezone.utc).isoformat())
    assert forwards<=plan['forwards_max']
    disk_guard.guard_write(1000000,label='v630 JSON');OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['selected'],indent=2));print(result['predictions'])


if __name__=='__main__':main()
