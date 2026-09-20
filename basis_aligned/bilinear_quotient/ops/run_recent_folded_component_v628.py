#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_recent_replay pred_b_component_replay pred_c_intervention_prediction
"""Extract/replay MLP16 -> attention17 -> frozen quadratic MLP17 component.

Input ports h16 (pre-MLP16), normalized embedding x0, first-layer values v1.
Independent Torch executor contains only local weights and the frozen component;
keeps MLP16 bias, residual lambdas, QK norms, RoPE, product attention, value mix,
MLP17 normalization and every background/self/cross term. Scale full MLP16
write by0,.5,1,1.5 in both native model and executor. No fitting.
Gates: recent h17 replay<=3e-5; folded scalar equals conditional256-square
component<=3e-5; predicted scalar *changes* vs native quadratic changes<=.10
for every nonbaseline scale/panel. PRICE16 batched native forwards and16 local
path replays;0backwards/updates/fits. Upstream ports remain native dependencies.
"""
import json,os,sys,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3];BQ=ROOT/'basis_aligned/bilinear_quotient'
OUT=BQ/'circuits/followups/recent_folded_component_v628_result.json'
PREDICTIONS={'pred_a_recent_replay':'<=3e-5','pred_b_component_replay':'<=3e-5',
             'pred_c_intervention_prediction':'<=.10 all changes'}


def main():
    plan=dict(panels=['fineweb_n96_skip1200.pt','fineweb_n192_skip11000.pt'],scales=[1.,0.,.5,1.5],
        documents=8,tokens=64,forwards_max=16,local_path_replays=16,
        model_backwards=0,model_updates=0,fit_parameters=0,execution_policy='managed_queue_only')
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
    model=producer.Bilin18TorchBackend.load('cuda').model;w=extract(model,p)
    b16,b17=model.transformer.h[16:18]
    reader=model.lm_head.weight.float().T@p['vocabulary_writer'];bias=b17.mlp.Down_bias.float()@reader
    state_errors=[];scalar_errors=[];rows=[];forwards=0
    for panel in plan['panels']:
        ids=torch.load(BQ/'.rowcache'/panel,map_location='cpu',weights_only=True)[:8,:65].cuda()
        sums={scale:dict(error=0.,effect=0.,target=0.,absolute=0.,cross=0.) for scale in plan['scales']}
        for start in range(0,8,4):
            ports=None;base=None
            for scale in plan['scales']:
                cache={}
                def write_hook(_m,args,out):cache['write16']=out.float();return out*scale
                def end16(_m,args,out):cache['after16']=out[0].float()
                def start17(_m,args):cache.update(v1=args[1].float(),x0=args[2].float())
                def mlp17(_m,args,out):cache.update(n17=args[0].float(),write17=out.float())
                def end17(_m,args,out):cache['end17']=out[0].float()
                hooks=[b16.mlp.register_forward_hook(write_hook),b16.register_forward_hook(end16),
                    b17.register_forward_pre_hook(start17),b17.mlp.register_forward_hook(mlp17),b17.register_forward_hook(end17)]
                try:model(ids[start:start+4,:-1],ids[start:start+4,1:].contiguous());forwards+=1
                finally:
                    for hook in hooks:hook.remove()
                if ports is None:
                    ports=(cache['after16']-cache['write16'],cache['x0'],cache['v1'])
                predicted=execute(w,*ports,scale)
                native_h=cache['end17']-cache['write17']
                state_errors.append(float((predicted['h17']-native_h).norm()/native_h.norm()))
                conditional=(cache['n17']@p['readers']).square()@p['coefficients']
                scalar_errors.append(float((predicted['alpha']-conditional).norm()/conditional.norm()))
                truth=cache['write17']@reader-bias
                if base is None:base=(predicted['alpha'],truth)
                delta_pred=predicted['alpha']-base[0];delta_true=truth-base[1]
                stats=sums[scale]
                stats['error']+=float((delta_pred-delta_true).double().square().sum())
                stats['effect']+=float(delta_true.double().square().sum())
                stats['target']+=float(truth.double().square().sum())
                stats['absolute']+=float((predicted['alpha']-truth).double().square().sum())
                stats['cross']+=float(predicted['terms'][...,1].double().square().sum())
        for scale,s in sums.items():
            rows.append(dict(panel=panel,scale=scale,relative_prediction_error=(s['absolute']/s['target'])**.5,
                relative_change_error=(s['error']/s['effect'])**.5 if s['effect'] else None,
                cross_term_l2_over_target=(s['cross']/s['target'])**.5))
    # Save a genuinely local package, split so every file stays below100MB.
    directory=OUT.with_name('recent_folded_component_v628_package');directory.mkdir(exist_ok=True)
    shards={'mlp16':{},'attention17':{},'component':{}}
    for key,value in w.items():
        category='mlp16' if key.endswith('16') else 'component' if key in ['readers','coefficients','residual_writer','vocabulary_writer'] else 'attention17'
        shards[category][key]=value.cpu() if torch.is_tensor(value) else value
    manifest={}
    for name,shard in shards.items():
        path=directory/(name+'.pt');disk_guard.guard_torch_save(shard,str(path))
        manifest[path.name]=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    runtime=directory/'execute.py'
    runtime.write_bytes((ROOT/'basis_aligned/polynomial_causal/recent_folded_component.py').read_bytes())
    manifest[runtime.name]=dict(bytes=runtime.stat().st_size,sha256=hashlib.sha256(runtime.read_bytes()).hexdigest())
    result=dict(plan=plan,rows=rows,forwards=forwards,maximum_state_replay=max(state_errors),
        maximum_component_replay=max(scalar_errors),package_files=manifest,
        package_tensor_values=sum(v.numel() for v in w.values() if torch.is_tensor(v)),
        predictions={'pred_a_recent_replay':max(state_errors)<=3e-5,'pred_b_component_replay':max(scalar_errors)<=3e-5,
            'pred_c_intervention_prediction':all(r['relative_change_error']<=.1 for r in rows if r['relative_change_error'] is not None)},
        scope='local extracted recent path; requires upstream h16/x0/v1 ports and is not yet a simpler full circuit',
        seconds=time.perf_counter()-tic,finished_utc=datetime.now(timezone.utc).isoformat())
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    disk_guard.guard_write(100000,label='v628 JSON');OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
