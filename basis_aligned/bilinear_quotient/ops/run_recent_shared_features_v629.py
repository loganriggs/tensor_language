#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_factored_replay pred_b_twenty_percent_saving pred_c_frozen_validation
"""Shared-feature versus matched-cost independent SVD in complete recent path.

Weights-only decompositions: joint stacked Left16/Right16 input frame, optional
Down16 output factorization, independent input frames with matched input-factor
storage. Ranks128/256/512/768. Closed-form SVD, no truncated iterative fit.
Select smallest full program with<=.10 scalar AND intervention-change error
on calibration panel. Freeze that choice for reporting validation. Rank and
family selection use calibration, no coefficients fit on rows. All candidate
curves reported. Native attention/norm/bias operations remain in every arm.
Gates: dense extracted reference vs256-square conditional target<=3e-5;
selected program stores<=.8 dense values; selected validation errors<=.10.
PRICE16 full native forwards,208 local replays,0backwards/modelupdates.
"""
import json,os,sys,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3];BQ=ROOT/'basis_aligned/bilinear_quotient'
OUT=BQ/'circuits/followups/recent_shared_features_v629_result.json'
PREDICTIONS={'pred_a_factored_replay':'<=3e-5','pred_b_twenty_percent_saving':'<=.8 dense values',
             'pred_c_frozen_validation':'<=.10 scalar and changes'}


def main():
    plan=dict(panels=['fineweb_n96_skip1200.pt','fineweb_n192_skip11000.pt'],ranks=[128,256,512,768],
        families=['shared_input','shared_input_output','independent_matched_input_output'],scales=[1.,0.,.5,1.5],
        documents=8,tokens=64,forwards_max=16,model_backwards=0,model_updates=0,fit_parameters=0,
        execution_policy='managed_queue_only')
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(plan));return
    import torch
    import disk_guard
    import circuit_fast_screen_producer as producer
    sys.path.insert(0,str(ROOT/'basis_aligned/polynomial_causal'))
    from recent_folded_component import extract,execute
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
    _,_,shared_v=torch.linalg.svd(torch.cat([L,R]),full_matrices=False)
    _,_,lv=torch.linalg.svd(L,full_matrices=False);_,_,rv=torch.linalg.svd(R,full_matrices=False)
    du,_,_=torch.linalg.svd(D,full_matrices=False)
    def build(family,rank):
        if family=='dense':return w
        q=dict(w)
        if family.startswith('shared'):
            P=shared_v[:rank].T;q.update(input_basis16=P,left16=L@P,right16=R@P)
        else:
            k=((2*h+d)*rank)//(2*(h+d))
            P,Q=lv[:k].T,rv[:k].T;q.update(left_basis16=P,right_basis16=Q,left16=L@P,right16=R@Q)
        if family.endswith('output'):
            Q=du[:,:rank];q.update(output_basis16=Q,down16=Q.T@D)
        return q
    def size(q):return sum(v.numel() for v in q.values() if torch.is_tensor(v))
    configs=[('dense',d)]+[(family,rank) for family in plan['families'] for rank in plan['ranks']]
    rows=[];dense_replay=[];local_calls=0
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
                if family=='dense':dense_replay.append(float((alpha-batch['conditional'][scale]).norm()/batch['conditional'][scale].norm()))
        panels={panel:{str(scale):dict(relative_error=(a['error']/a['target'])**.5,
            relative_change_error=(a['change_error']/a['change'])**.5 if a['change'] else None)
            for scale,a in values.items()} for panel,values in stats.items()}
        maximum=lambda panel:max(max(a['relative_error'],a['relative_change_error'] or 0.) for a in panels[panel].values())
        row=dict(family=family,rank=rank,stored_values=size(q),fraction_of_dense=size(q)/size(w),panels=panels,
                 calibration_max_error=maximum(plan['panels'][0]),validation_max_error=maximum(plan['panels'][1]))
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='panels'}),flush=True)
    selected=min((r for r in rows if r['calibration_max_error']<=.1),key=lambda r:r['stored_values'])
    directory=OUT.with_name('recent_shared_features_v629_package');directory.mkdir(exist_ok=True)
    chosen=build(selected['family'],selected['rank']);shards={'mlp16':{},'attention17':{},'component':{}}
    for key,value in chosen.items():
        category='mlp16' if key.endswith('16') else 'component' if key in ['readers','coefficients','residual_writer','vocabulary_writer'] else 'attention17'
        shards[category][key]=value.detach().cpu().contiguous().clone() if torch.is_tensor(value) else value
    manifest={}
    for name,shard in shards.items():
        path=directory/(name+'.pt');disk_guard.guard_torch_save(shard,str(path))
        manifest[path.name]=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    runtime=directory/'execute.py';runtime.write_bytes((ROOT/'basis_aligned/polynomial_causal/recent_folded_component.py').read_bytes())
    manifest[runtime.name]=dict(bytes=runtime.stat().st_size,sha256=hashlib.sha256(runtime.read_bytes()).hexdigest())
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    result=dict(plan=plan,rows=rows,selected={k:v for k,v in selected.items() if k!='panels'},
        forwards=forwards,local_replays=local_calls,maximum_dense_replay=max(dense_replay),package_files=manifest,
        predictions={'pred_a_factored_replay':max(dense_replay)<=3e-5,
            'pred_b_twenty_percent_saving':selected['fraction_of_dense']<=.8,
            'pred_c_frozen_validation':selected['validation_max_error']<=.1},
        scope='shared-linear-factor spectral baseline, not joint tensor optimum or sparse circuit; upstream ports still required',
        seconds=time.perf_counter()-tic,finished_utc=datetime.now(timezone.utc).isoformat())
    assert forwards<=plan['forwards_max']
    disk_guard.guard_write(1000000,label='v629 JSON');OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['selected'],indent=2));print(result['predictions'])


if __name__=='__main__':main()
