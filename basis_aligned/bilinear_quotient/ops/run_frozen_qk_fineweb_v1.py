#!/usr/bin/env python3
# BQGATE: 24forwards192seq128tokens;32FineWeb rows,6arms,no fitting.
import os
os.environ['OMP_NUM_THREADS']='2'
import sys,json,time,hashlib,signal
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];P=ROOT/'basis_aligned/polynomial_causal';BQ=RUNNER.parents[1]
sys.path[:0]=[str(RUNNER.parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from compiled_qk_vector_edit_v1 import compile_vector,predict_vector
from shared_position_qk_edit_v1 import prepare_from_head_vectors
DATA=BQ/'.rowcache/fineweb_n192_skip7000.pt'
SOURCE,QUERY=32,127


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    binding=json.loads((P/'FROZEN_QK_FINEWEB_V1_BINDING.json').read_text());assert all(digest(p)==h for p,h in binding.items())
    assert json.loads((P/'COMPILED_QK_VECTOR_EDIT_V1_CONTROL.json').read_text())['instrument_passed']
    all_rows=torch.load(DATA,weights_only=True,map_location='cpu');assert all_rows.shape==(192,513)
    rows=all_rows[:32,:129].contiguous();row_hashes=[hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in rows]
    assert len(set(row_hashes))==32 and rows.dtype==torch.long
    prior=json.loads((P/'POSITION_SHARED_QK_SOURCE_V1_RESULT.json').read_text());frame_cache=prior['cache']
    assert digest(frame_cache['path'])==frame_cache['sha256']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,model_loaded=False,forwards=24,sequences=192,distinct_rows=32,
                             input_tokens=128,source_position=SOURCE,query_position=QUERY,corpus='FineWeb',fitting=False)));return
    out=P/'FROZEN_QK_FINEWEB_V1_RESULT.json';assert not out.exists();signal.alarm(900)
    start=time.perf_counter();torch.set_num_threads(2);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();torch.set_num_threads(2)
    attention=model.transformer.h[17].attn
    assert attention.squared_attn and attention.n_head==9 and attention.head_dim==128
    learned=torch.load(frame_cache['path'],weights_only=True,map_location='cpu')['frames'].cuda()
    weights={key:getattr(attention,key).weight.detach().double().reshape(9,128,1152) for key in ['c_q','c_k','c_q2','c_k2','c_v']}
    torch.manual_seed(1321);random=[]
    for h in range(9):
        key_basis=torch.linalg.qr(torch.cat((weights['c_k'][h],weights['c_k2'][h]),dim=0).T).Q
        gaussian=torch.randn(256,17,dtype=torch.float64,device='cpu').cuda()
        random.append(key_basis@torch.linalg.qr(gaussian).Q)
    spaces={'learned':learned,'random':torch.stack(random)}
    errors=[];scope_errors=[];counts=[0,0];state={'arm':'baseline'};cache={};attention_outputs={}
    for basis in spaces.values():errors.append(float((basis.transpose(1,2)@basis-torch.eye(17,device='cuda')).abs().max()))
    static={}
    mix=float(attention.lamb)
    for variant,bases in spaces.items():
        static[variant]=[compile_vector([weights[k][h] for k in ['c_q','c_k','c_q2','c_k2']],bases[h],weights['c_v'][h],mix) for h in range(9)]
    def count(_module,args):
        counts[0]+=1;counts[1]+=len(args[0]);assert counts[0]<=24
    def capture_input(_module,args):
        if state['arm']=='baseline':
            cache['source']=args[0][:,SOURCE].detach().clone()
            cache['base_value']=args[1][:,SOURCE].detach().clone()
        else:
            scope_errors.append(float((args[0][:,SOURCE]-cache['source']).abs().max()))
            scope_errors.append(float((args[1][:,SOURCE]-cache['base_value']).abs().max()))
    def linear_hook(name):
        def hook(module,args,output):
            position=QUERY if name in ['c_q','c_q2'] else SOURCE
            if state['arm']=='baseline':cache[name]=output[:,position].reshape(-1,9,128).detach().clone()
            elif name in ['c_q','c_q2']:
                scope_errors.append(float((output[:,position].reshape(-1,9,128)-cache[name]).abs().max()))
            if state['arm'].startswith('physical_') and name in ['c_k','c_k2','c_v']:
                variant=state['arm'].split('_',1)[1];basis=spaces[variant]
                source=args[0][:,SOURCE].double()
                z=torch.einsum('bd,hdr->bhr',source,basis)
                edited=(source[:,None,:]-torch.einsum('bhr,hdr->bhd',z,basis)).to(output.dtype)
                native_weights=module.weight.reshape(9,128,1152)
                changed=torch.einsum('bhd,hkd->bhk',edited,native_weights)
                result=output.clone();result[:,SOURCE]=changed.flatten(1)
                scope_errors.extend([float((result[:,:SOURCE]-output[:,:SOURCE]).abs().max()),
                                     float((result[:,SOURCE+1:]-output[:,SOURCE+1:]).abs().max())])
                return result
        return hook
    original_squared=attention.squared_attention
    def squared(q,k,v,q2,k2):
        if state['arm']=='baseline':cache['mixed_value']=v[:,SOURCE].detach().clone()
        return original_squared(q,k,v,q2,k2)
    def output_hook(_module,args,output):
        y,v1=output
        if state['arm'].startswith('predicted_'):
            variant=state['arm'].split('_',1)[1]
            scope_errors.append(float((y[:,QUERY]-attention_outputs['baseline']).abs().max()))
            y=y.clone();y[:,QUERY]+=state['predicted_delta'][variant].to(y.dtype)
        attention_outputs[state['arm']]=y[:,QUERY].detach().clone()
        return y,v1
    handles=[model.transformer.h[0].attn.register_forward_pre_hook(count),attention.register_forward_pre_hook(capture_input),attention.register_forward_hook(output_hook)]
    handles.extend(getattr(attention,key).register_forward_hook(linear_hook(key)) for key in weights)
    attention.squared_attention=squared
    def forward(tokens,arm):
        state['arm']=arm
        x=F.rms_norm(model.transformer.wte(tokens),(1152,));x0=x;v1=None
        for block in model.transformer.h:x,v1=block(x,v1,x0)
        return 30*torch.tanh(model.lm_head(F.rms_norm(x[:,QUERY],(1152,)))/30)
    records=[];batch_errors=[];stored_logits={};stored_ports=[];stored_attention=[];zero_error=0.
    try:
        for offset in range(0,32,8):
            tokens=rows[offset:offset+8,:128].cuda();target=rows[offset:offset+8,128].cuda()
            logits={'baseline':forward(tokens,'baseline')};saved_ports={k:v.cpu() for k,v in cache.items()}
            state['predicted_delta']={};projected={}
            for variant,bases in spaces.items():
                changes=[]
                projected[variant]=torch.einsum('bd,hdr->bhr',cache['source'].double(),bases).square().sum(-1)/cache['source'].double().square().sum(-1)[:,None]
                for h in range(9):
                    ports=prepare_from_head_vectors(static[variant][h],cache['source'].double(),
                        [cache[k][:,h].double() for k in ['c_q','c_q2']],
                        [cache[k][:,h].double() for k in ['c_k','c_k2']],cache['mixed_value'][:,h].double(),QUERY,SOURCE)
                    edited=predict_vector(ports,static[variant][h],torch.eye(17,dtype=torch.float64,device='cuda'))
                    original=predict_vector(ports,static[variant][h],torch.zeros(17,17,dtype=torch.float64,device='cuda'))
                    changes.append(edited['contribution']-original['contribution'])
                state['predicted_delta'][variant]=torch.stack(changes,1).flatten(1)@attention.c_proj.weight.double().T
            # Predictions are now fixed, before either physical edited forward.
            for arm in ['zero','physical_learned','predicted_learned','physical_random','predicted_random']:
                logits[arm]=forward(tokens,arm)
            zero_error=max(zero_error,float((logits['zero']-logits['baseline']).abs().max()))
            base_log=logits['baseline'].double().log_softmax(-1);base_prob=base_log.exp()
            base_ce=-base_log.gather(1,target[:,None]).squeeze(1)
            for variant in spaces:
                physical=logits['physical_'+variant].double();predicted=logits['predicted_'+variant].double()
                actual_delta=attention_outputs['physical_'+variant].double()-attention_outputs['baseline'].double()
                residual_error=float((state['predicted_delta'][variant]-actual_delta).norm()/actual_delta.norm().clamp_min(1e-30))
                actual_logits=physical-logits['baseline'].double();predicted_logits=predicted-logits['baseline'].double()
                logit_error=float((predicted_logits-actual_logits).norm()/actual_logits.norm().clamp_min(1e-30))
                physical_log=physical.log_softmax(-1);predicted_log=predicted.log_softmax(-1)
                physical_ce=-physical_log.gather(1,target[:,None]).squeeze(1);predicted_ce=-predicted_log.gather(1,target[:,None]).squeeze(1)
                kl=(base_prob*(base_log-physical_log)).sum(-1)
                batch_errors.append(dict(offset=offset,variant=variant,attention_change_relative_error=residual_error,
                                         logit_change_relative_error=logit_error))
                for i in range(8):records.append(dict(row=offset+i,variant=variant,target=int(target[i]),baseline_ce=float(base_ce[i]),
                    physical_ce=float(physical_ce[i]),predicted_ce=float(predicted_ce[i]),ce_change=float(physical_ce[i]-base_ce[i]),
                    ce_prediction_absolute_error=float(abs(predicted_ce[i]-physical_ce[i])),baseline_to_edited_kl=float(kl[i]),
                    mean_projected_source_energy_fraction=float(projected[variant][i].mean())))
            for key,value in logits.items():stored_logits.setdefault(key,[]).append(value.cpu())
            stored_ports.append(saved_ports);stored_attention.append({k:v.cpu() for k,v in attention_outputs.items()})
            print(json.dumps(dict(rows_completed=offset+8,batch_errors=batch_errors[-2:])),flush=True)
    finally:
        attention.squared_attention=original_squared
        for handle in handles:handle.remove()
    summaries={}
    for variant in spaces:
        group=[r for r in records if r['variant']==variant]
        summaries[variant]={key:sum(r[key] for r in group)/32 for key in ['baseline_ce','physical_ce','ce_change','ce_prediction_absolute_error','baseline_to_edited_kl','mean_projected_source_energy_fraction']}
    valid=counts==[24,192] and max(scope_errors+[zero_error]+errors)<=1e-6 and all(r['attention_change_relative_error']<=1e-3 for r in batch_errors)
    finite=all(torch.isfinite(torch.tensor(list(r.values()))).all() for r in summaries.values());valid=valid and bool(finite)
    target_cache=Path('/dev/shm/bilin18_frozen_qk_fineweb_v1.pt');assert not target_cache.exists()
    torch.save(dict(logits={k:torch.cat(v) for k,v in stored_logits.items()},ports=stored_ports,attention=stored_attention,
                    frames={k:v.cpu() for k,v in spaces.items()},rows=rows),target_cache)
    result=dict(predictions={'pred_a_instrument':valid,'pred_b_endpoint_prediction':valid and all(r['logit_change_relative_error']<=.01 for r in batch_errors),
                            'pred_c_ce_prediction':valid and all(r['ce_prediction_absolute_error']<=1e-3 for r in summaries.values())},
                summaries=summaries,batch_errors=batch_errors,rows=records,zero_hook_maximum_error=zero_error,maximum_hook_scope_error=max(scope_errors),
                data=dict(path=str(DATA),sha256=digest(DATA),row_sha256=row_hashes,source='HuggingFaceFW/fineweb',historically_opened=True,fresh_or_document_holdout=False),
                cache=dict(path=str(target_cache),sha256=digest(target_cache),bytes=target_cache.stat().st_size,ephemeral=True),
                price=dict(body_forwards=counts[0],sequences=counts[1],distinct_input_positions=4096,processed_positions=24576,native_model_retained=True),
                wall_seconds=time.perf_counter()-start,
                scope='Frozen weight-derived source read-edge intervention validated on32cached FineWeb rows. Conditional attention prediction plus native downstream replay; no fitting, semantic selectivity, OOD claim or whole-model saving.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','data']}),flush=True)


if __name__=='__main__':main()
