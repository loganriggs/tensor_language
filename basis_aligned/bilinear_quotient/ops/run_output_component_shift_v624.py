#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_quadratic_replay pred_b_all_family_prediction pred_c_removal_prediction
"""Frozen256-square component on three predefined synthetic input families.

Eight code, arithmetic, repetition prompts each. No fitting, ranking or width
selection. These shift from the prose calibration panels; no assertion that
the model's pretraining omitted these domains. Native bias stays outside the
quadratic target. Compare frozen scalar prediction and exact-vs-approximate
component removal through actual final norm/softcap.
Gates: quadratic replay<=2e-5; relative component error<=.10 in ALL families;
approx/exact removal mean CE differ<=.02 in ALL families. PRICE24 native
forwards,72 suffix evaluations,0fits/backwards/updates.
"""
import json,os,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3];BQ=ROOT/'basis_aligned/bilinear_quotient'
OUT=BQ/'circuits/followups/output_component_shift_v624_result.json'
PREDICTIONS={'pred_a_quadratic_replay':'<=2e-5','pred_b_all_family_prediction':'<=.10 all',
             'pred_c_removal_prediction':'<=.02 CE all'}


def prompts():
    return {
        'code':[f'def add_{i}(x):\n    result = x + {i}\n    return result\n\nprint(add_{i}({i+3}))\n' for i in range(1,9)],
        'arithmetic':[f'Compute each sum. {i} + {i+2} = {2*i+2}. {i+4} + {i+1} = {2*i+5}. The next calculation is {i+3} + {i+7} =' for i in range(1,9)],
        'repetition':[' '.join([word]*24)+'.' for word in ['hello','world','red','green','zero','seven','again','never']]}


def main():
    texts=prompts()
    plan=dict(families=texts,forwards_max=24,model_backwards=0,model_updates=0,fit_parameters=0,
        execution_policy='managed_queue_only',component='frozen v621 width256')
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(plan));return
    import torch,tiktoken
    import torch.nn.functional as F
    import disk_guard
    import circuit_fast_screen_producer as producer
    tic=time.perf_counter();torch.set_grad_enabled(False);torch.set_num_threads(8)
    ppath=OUT.with_name('output_component_rank_v621_program.pt')
    p=torch.load(ppath,map_location='cuda',weights_only=True)
    model=producer.Bilin18TorchBackend.load('cuda').model;block=model.transformer.h[17]
    U=model.lm_head.weight.float();reader=U.T@p['vocabulary_writer'];v=p['residual_writer']
    c=block.mlp.Down.weight.float().T@reader;bias=block.mlp.Down_bias.float()@reader
    L,R=block.mlp.Left.weight.float(),block.mlp.Right.weight.float()
    enc=tiktoken.get_encoding('gpt2');rows=[];checks=[];forwards=0
    for family,strings in texts.items():
        num=den=0.;records=[]
        for text in strings:
            ids=torch.tensor([enc.encode(text)],device='cuda');tokens=ids[:,:-1];target=ids[:,1:].contiguous()
            cache={}
            def capture_mlp(_m,args,out):cache.update(n=args[0].float(),out=out.float())
            def capture_y(_m,args,out):cache['y']=out[0].float()
            hooks=[block.mlp.register_forward_hook(capture_mlp),block.register_forward_hook(capture_y)]
            try:native=float(model(tokens,target));forwards+=1
            finally:
                for hook in hooks:hook.remove()
            n,y=cache['n'],cache['y'];truth=cache['out']@reader-bias
            direct=((n@L.T)*(n@R.T))@c
            checks.append(float((direct-truth).norm()/truth.norm()))
            approx=(n@p['readers']).square()@p['coefficients']
            num+=float((truth-approx).double().square().sum());den+=float(truth.double().square().sum())
            losses={}
            for arm,scalar in [('baseline',torch.zeros_like(truth)),('exact',truth),('approximate',approx)]:
                raw=F.rms_norm(y-scalar[...,None]*v,(1152,))@U.T
                logits=30*torch.tanh(raw/30)
                losses[arm]=float(F.cross_entropy(logits.reshape(-1,U.shape[0]),target.flatten()))
            checks.append(abs(losses['baseline']-native)/max(abs(native),1e-9))
            records.append(dict(tokens=tokens.numel(),relative_prediction_error=float((truth-approx).norm()/truth.norm()),losses=losses))
        total_tokens=sum(r['tokens'] for r in records)
        means={arm:sum(r['tokens']*r['losses'][arm] for r in records)/total_tokens for arm in records[0]['losses']}
        rows.append(dict(family=family,tokens=total_tokens,relative_prediction_error=(num/den)**.5,
            token_weighted_losses=means,removal_ce_difference=means['approximate']-means['exact'],per_prompt=records))
    assert forwards<=plan['forwards_max']
    result=dict(plan=plan,rows=rows,forwards=forwards,manual_suffix_evaluations=forwards*3,
        maximum_replay_relative_error=max(checks),program_sha256=hashlib.sha256(ppath.read_bytes()).hexdigest(),
        predictions={'pred_a_quadratic_replay':max(checks)<=2e-5,
            'pred_b_all_family_prediction':all(r['relative_prediction_error']<=.1 for r in rows),
            'pred_c_removal_prediction':all(abs(r['removal_ce_difference'])<=.02 for r in rows)},
        scope='synthetic shifts relative to prose calibration; conditional component and model loss only, not semantic selectivity or reuse',
        seconds=time.perf_counter()-tic,finished_utc=datetime.now(timezone.utc).isoformat())
    disk_guard.guard_write(1000000,label='v624 JSON');OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({**result,'rows':[{k:v for k,v in r.items() if k!='per_prompt'} for r in rows]},indent=2))


if __name__=='__main__':main()
