#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replay pred_b_mlp16_closure pred_c_attn5_closure
"""Do frozen source deletions predict actual upstream ablations?

MLP16 and attention5 selected from v625's calibration census. Remove their
whole native write, retain all other module logic, recapture MLP17 input.
Compare with subtracting only the carried baseline write from h17, using
actual recomputed RMSNorm and exact native MLP17 for both predictions.
Thus a failure measures downstream source drift, not the256-feature error.
Gates: reconstruction of baseline h17 normed input<=3e-5; conditional deletion
predicts actual alpha change to<=.10 relative L2 for each source on both panels.
PRICE12 full native forwards +8 isolated MLP17 evaluations;0fits/backwards.
"""
import json,os,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3];BQ=ROOT/'basis_aligned/bilinear_quotient'
OUT=BQ/'circuits/followups/output_source_causal_closure_v627_result.json'
PREDICTIONS={'pred_a_baseline_replay':'<=3e-5','pred_b_mlp16_closure':'<=.10 both',
             'pred_c_attn5_closure':'<=.10 both'}


def main():
    plan=dict(panels=['fineweb_n96_skip1200.pt','fineweb_n192_skip11000.pt'],sources=['mlp16','attn5'],
        documents=8,tokens=64,forwards_max=12,isolated_mlp_evaluations=8,model_backwards=0,model_updates=0,
        fit_parameters=0,execution_policy='managed_queue_only')
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(plan));return
    import torch
    import torch.nn.functional as F
    import disk_guard
    import circuit_fast_screen_producer as producer
    tic=time.perf_counter();torch.set_grad_enabled(False);torch.set_num_threads(8)
    p=torch.load(OUT.with_name('output_component_rank_v621_program.pt'),map_location='cuda',weights_only=True)
    census=json.loads(OUT.with_name('output_component_sources_v625_result.json').read_text())
    model=producer.Bilin18TorchBackend.load('cuda').model;b17=model.transformer.h[17]
    reader=model.lm_head.weight.float().T@p['vocabulary_writer'];bias=b17.mlp.Down_bias.float()@reader
    modules={'mlp16':model.transformer.h[16].mlp,'attn5':model.transformer.h[5].attn}
    def run(tokens,target,remove=None):
        cache={};hooks=[]
        for name,module in modules.items():
            def source(_m,args,out,name=name):
                value=out[0] if name.startswith('attn') else out
                cache[name]=value.float()
                if name==remove:
                    return (torch.zeros_like(value),out[1]) if isinstance(out,tuple) else torch.zeros_like(value)
            hooks.append(module.register_forward_hook(source))
        def target_hook(_m,args,out):cache.update(n=args[0].float(),mlp17=out.float())
        def final_hook(_m,args,out):cache['final']=out[0].float()
        hooks.extend([b17.mlp.register_forward_hook(target_hook),b17.register_forward_hook(final_hook)])
        try:cache['loss']=float(model(tokens,target))
        finally:
            for hook in hooks:hook.remove()
        cache['h']=cache['final']-cache['mlp17']
        cache['alpha']=cache['mlp17']@reader-bias
        return cache
    rows=[];checks=[];forwards=0
    for panel in plan['panels']:
        ids=torch.load(BQ/'.rowcache'/panel,map_location='cpu',weights_only=True)[:8,:65].cuda()
        values={name:[] for name in modules}
        for start in range(0,8,4):
            tokens,target=ids[start:start+4,:-1],ids[start:start+4,1:].contiguous()
            base=run(tokens,target);forwards+=1
            checks.append(float((F.rms_norm(base['h'],(1152,))-base['n']).norm()/base['n'].norm()))
            for name in modules:
                frozen_h=base['h']-census['carriage_weights'][name]*base[name]
                frozen_alpha=b17.mlp(F.rms_norm(frozen_h,(1152,))).float()@reader-bias
                actual=run(tokens,target,name);forwards+=1
                observed=actual['alpha']-base['alpha'];predicted=frozen_alpha-base['alpha']
                values[name].append(dict(error=float((observed-predicted).double().square().sum()),
                    effect=float(observed.double().square().sum()),baseline=float(base['alpha'].double().square().sum()),
                    drift=float((actual['h']-frozen_h).double().square().sum()),
                    state=float(actual['h'].double().square().sum()),delta_ce=actual['loss']-base['loss']))
        for name,measurements in values.items():
            total=lambda key:sum(r[key] for r in measurements)
            rows.append(dict(panel=panel,source=name,relative_effect_prediction_error=(total('error')/max(total('effect'),1e-30))**.5,
                effect_over_baseline=(total('effect')/total('baseline'))**.5,
                downstream_state_drift_relative=(total('drift')/total('state'))**.5,
                delta_ce=total('delta_ce')/len(measurements)))
    assert forwards<=plan['forwards_max']
    result=dict(plan=plan,rows=rows,forwards=forwards,maximum_baseline_replay=max(checks),
        predictions={'pred_a_baseline_replay':max(checks)<=3e-5,
            'pred_b_mlp16_closure':all(r['relative_effect_prediction_error']<=.1 for r in rows if r['source']=='mlp16'),
            'pred_c_attn5_closure':all(r['relative_effect_prediction_error']<=.1 for r in rows if r['source']=='attn5')},
        scope='actual upstream whole-write ablations compared to frozen-write boundary prediction; no semantic selectivity claim',
        seconds=time.perf_counter()-tic,finished_utc=datetime.now(timezone.utc).isoformat())
    disk_guard.guard_write(100000,label='v627 JSON');OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
