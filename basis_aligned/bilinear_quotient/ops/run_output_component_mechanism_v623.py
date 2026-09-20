#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_suffix_replay pred_b_isotropic_prediction pred_c_direction_dominates
"""Bias-corrected quadratic mechanism screen; preserves failed v622 receipt.

Fix: native MLP Down_bias is a constant, not part of C,A,B. Subtract its
projected scalar before comparing/removing the quadratic component; retain
the original bias in the model. v622 wrongly included it in alpha.

Exact scalar alpha=n^T Q n from normalized MLP17 input. The weights-only
isotropic competitor is trace(Q)*||n||²/d; no constant fitted to data.
Compare exact/256-square/isotropic/traceless removals with native suffix.
Split exact removal into fixed-final-norm direct effect and renormalization;
within fixed norm, compare common/centered vocabulary directions through tanh.
Frozen gates: manual native suffix and actual edited-model CE replay<=2e-5;
isotropic competitor predicts alpha with<=.15 error on both panels; magnitude
of fixed-norm CE effect >= magnitude of residual normalization effect on both.
Last two are hypotheses, not required to pass. PRICE6 model forwards,0fits,
0backwards/updates. Remaining suffix evaluations are explicitly counted.
"""
import json,os,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3];BQ=ROOT/'basis_aligned/bilinear_quotient'
OUT=BQ/'circuits/followups/output_component_mechanism_v623_result.json'
PREDICTIONS={'pred_a_native_suffix_replay':'<=2e-5','pred_b_isotropic_prediction':'<=.15 both',
             'pred_c_direction_dominates':'abs(direct)>=abs(norm) both'}


def main():
    plan=dict(panels=['fineweb_n96_skip1200.pt','fineweb_n192_skip11000.pt'],documents=8,tokens=64,
        forwards_max=6,model_backwards=0,model_updates=0,fit_parameters=0,execution_policy='managed_queue_only')
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(plan));return
    import torch
    import torch.nn.functional as F
    import disk_guard
    import circuit_fast_screen_producer as producer
    tic=time.perf_counter();torch.set_grad_enabled(False);torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    p=torch.load(OUT.with_name('output_component_rank_v621_program.pt'),map_location='cuda',weights_only=True)
    model=producer.Bilin18TorchBackend.load('cuda').model;block=model.transformer.h[17]
    U=model.lm_head.weight.float();v=p['residual_writer'];writer=U@v
    reader=U.T@p['vocabulary_writer'];D=block.mlp.Down.weight.float()
    L,R=block.mlp.Left.weight.float(),block.mlp.Right.weight.float()
    bias_alpha=block.mlp.Down_bias.float()@reader
    c=D.T@reader;trace=(c*(L*R).sum(-1)).sum();d=L.shape[1]
    rows=[];forwards=suffix_calls=0;checks=[]
    for panel in plan['panels']:
        ids=torch.load(BQ/'.rowcache'/panel,map_location='cpu',weights_only=True)[:8,:65].cuda()
        measurements=[];alphas=[];isotropics=[];rank_predictions=[]
        for start in range(0,8,4):
            tokens,target=ids[start:start+4,:-1],ids[start:start+4,1:].contiguous()
            cache={}
            def capture_mlp(_m,args,out):cache.update(n=args[0].float(),write=out.float())
            def capture_end(_m,args,out):cache['y']=out[0].float()
            hooks=[block.mlp.register_forward_hook(capture_mlp),block.register_forward_hook(capture_end)]
            try:native=float(model(tokens,target));forwards+=1
            finally:
                for hook in hooks:hook.remove()
            n,y=cache['n'],cache['y'];alpha=cache['write']@reader-bias_alpha
            predicted=((n@L.T)*(n@R.T))@c
            checks.append(float((predicted-alpha).norm()/alpha.norm()))
            iso=trace*n.square().mean(-1)
            approximate=(n@p['readers']).square()@p['coefficients']
            alphas.append(alpha.flatten().double());isotropics.append(iso.flatten().double())
            rank_predictions.append(approximate.flatten().double())
            def ce(raw):
                nonlocal suffix_calls
                suffix_calls+=1
                return float(F.cross_entropy((30*torch.tanh(raw/30)).reshape(-1,U.shape[0]),target.flatten()))
            scale=(y.square().mean(-1,keepdim=True)+torch.finfo(y.dtype).eps).sqrt()
            raw=(y/scale)@U.T
            values={'baseline':ce(raw)};checks.append(abs(values['baseline']-native)/max(abs(native),1e-9))
            for name,a in [('exact',alpha),('rank256',approximate),('isotropic',iso),('traceless',alpha-iso)]:
                edited=y-a[...,None]*v
                values[name]=ce(F.rms_norm(edited,(d,))@U.T)
            delta=alpha[...,None]/scale*writer
            values['fixed_norm']=ce(raw-delta)
            values['common_fixed_norm']=ce(raw-delta.mean(-1,keepdim=True))
            values['centered_fixed_norm']=ce(raw-(delta-delta.mean(-1,keepdim=True)))
            if start==0:
                def edit(_m,args,out):return out-((out.float()@reader)-bias_alpha)[...,None]*v
                hook=block.mlp.register_forward_hook(edit)
                try:actual_edit=float(model(tokens,target));forwards+=1
                finally:hook.remove()
                checks.append(abs(actual_edit-values['exact'])/max(abs(actual_edit),1e-9))
            measurements.append(values)
        mean={k:sum(m[k] for m in measurements)/len(measurements) for k in measurements[0]}
        a,i,pred=map(torch.cat,(alphas,isotropics,rank_predictions))
        delta_ce={k:x-mean['baseline'] for k,x in mean.items() if k!='baseline'}
        rows.append(dict(panel=panel,rows_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
            alpha_mean=float(a.mean()),alpha_std=float(a.std()),isotropic_value_mean=float(i.mean()),
            isotropic_prediction_error=float((a-i).norm()/a.norm()),
            rank256_prediction_error=float((a-pred).norm()/a.norm()),losses=mean,delta_ce=delta_ce,
            final_norm_increment=mean['exact']-mean['fixed_norm'],
            common_centered_interaction=mean['fixed_norm']-mean['common_fixed_norm']-mean['centered_fixed_norm']+mean['baseline']))
    assert forwards<=plan['forwards_max']
    result=dict(plan=plan,rows=rows,trace_Q=float(trace),excluded_bias_alpha=float(bias_alpha),maximum_replay_relative_error=max(checks),
        forwards=forwards,manual_suffix_evaluations=suffix_calls,
        predictions={'pred_a_native_suffix_replay':max(checks)<=2e-5,
            'pred_b_isotropic_prediction':all(r['isotropic_prediction_error']<=.15 for r in rows),
            'pred_c_direction_dominates':all(abs(r['delta_ce']['fixed_norm'])>=abs(r['final_norm_increment']) for r in rows)},
        scope='path-dependent intervention decomposition, no semantic circuit or domain OOD claim; common and centered effects need not add through softcap',
        seconds=time.perf_counter()-tic,finished_utc=datetime.now(timezone.utc).isoformat())
    disk_guard.guard_write(100000,label='v623 JSON');OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
