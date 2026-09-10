#!/usr/bin/env python3
# BQGATE: MLP8 directcarry vs A9-clamped bypass;192forwards3072seq+128decoderbatches.
"""A instrument; B task fidelity; C whole-vocabulary fidelity. .10allworlds;0fits."""
import hashlib,json,os,signal,sys,time,math
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path.insert(0,str(POLY));sys.path.insert(0,str(ROOT))
import numpy as np
import circuit_fast_screen_producer as P
import mixed_state_projector_v1 as Q
import module_output_delta_v1 as D
import attention_output_delta_v1 as A
import source_margin_gradient as G
import attention_write_factorial_executor_v1 as S
from circuit_fast_screen_managed_runner import atomic_create_json
BINDING=POLY/'MLP8_BYPASS_CARRY_V1_BINDING.json';OUT=POLY/'MLP8_BYPASS_CARRY_V1_RESULT.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    binding=json.loads(BINDING.read_text());assert all(sha(p)==v for p,v in binding.items())
    bank=json.loads((POLY/'THIRD_NOUN_CAUSAL_ROLE_ALIGNMENT_V1.json').read_text())
    parent=json.loads((POLY/'MLP8_ATTENTION9_FACTORIAL_V1_RESULT.json').read_text());assert parent['predictions']['pred_a_instrument']
    lookup={r['world_id']:r for r in parent['reports']}
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'gpu_accessed':False,'model_loaded':False,'forwards':192,'sequences':3072,'decoder_batches':128,'decoder_states':2048,'fits':0}));return
    assert not OUT.exists();signal.alarm(600)
    backend=P.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    model=backend.model;mlp=model.transformer.h[8].mlp;attn=model.transformer.h[9].attn
    q=torch.tensor(Q.projector(bank['corners'],2,4),device='cuda',dtype=torch.float64)
    lambdas=[float(b.lambdas[0]) for b in model.transformer.h[9:]];gamma=math.prod(lambdas)
    counts=[0,0];decoder_batches=0;reports=[];tic=time.perf_counter()
    def count(_m,args,_out):counts[0]+=1;counts[1]+=len(args[0])
    counter=model.transformer.h[0].attn.register_forward_hook(count)
    def ratio(x,y):return float(x.norm()/y.norm()) if y.norm()>0 else None
    def bridge(x,y):return {'max_abs':float((x-y).abs().max()),'relative':float((x-y).norm()/y.norm().clamp_min(1e-30))}
    try:
        with torch.inference_mode():
            for world in bank['worlds']:
                foil=1 if world['foil']=='himself' else 2;ids=bank['reader_ids']
                def batch(start):
                    rows=world['rows'][start:start+16]
                    return P.ModelBatch(tuple(r['row_id'] for r in rows),'base',tuple(tuple(r['ids']) for r in rows),
                        (ids[0],)*16,(ids[foil],)*16,(world['length']-1,)*16)
                def run(b,final=False):
                    record={};handles=[]
                    try:
                        handles.append(mlp.register_forward_hook(lambda _m,_a,out:record.update(mlp=out.detach().clone())))
                        handles.append(attn.register_forward_hook(lambda _m,_a,out:record.update(write=out[0].detach().clone(),first=out[1].detach().clone())))
                        with G.capture(model) as g:out=backend.native(b,capture=final)
                        record['logits']=G.endpoint_logits(g['raw_logits'],b).double()
                        if final:record['final']=torch.stack([out.captured[r,'resid:18'] for r in b.row_ids])
                    finally:
                        for h in handles:h.remove()
                    return record
                def join(chunks):return {k:torch.cat([r[k] for r in chunks]) for k in chunks[0]}
                native=join([run(batch(start),True) for start in (0,16)])
                delta=torch.einsum('ij,jtd->itd',q,native['mlp'].double());audit=[];parts=[]
                for start in (0,16):
                    with D.subtract(mlp,native['mlp'][start:start+16],delta[start:start+16],audit):parts.append(run(batch(start)))
                source=join(parts);assert torch.equal(native['first'],source['first']);parts=[]
                for start in (0,16):
                    expected=source['write'][start:start+16];target=native['write'][start:start+16]
                    with D.subtract(mlp,native['mlp'][start:start+16],delta[start:start+16],audit):
                        with A.subtract(attn,expected,expected.double()-target.double(),[]):parts.append(run(batch(start)))
                bypass=join(parts);final=native['final']
                states={'native':final,'carry':(final.double()-gamma*delta[:,-1]).to(final)};decoded={}
                for arm,state in states.items():
                    parts=[]
                    for chunk in state.split(16):
                        normalized=torch.nn.functional.rms_norm(chunk,(chunk.size(-1),))
                        parts.append((30*torch.tanh(model.lm_head(normalized)/30)).double());decoder_batches+=1
                    decoded[arm]=torch.cat(parts)
                z0=native['logits'];target=q@(z0-bypass['logits']);prediction=q@(z0-decoded['carry']);error=prediction-target
                old=np.array(lookup[world['world_id']]['cube_logits'])
                bridges={'native':S.bridge(z0[:,ids].cpu().numpy(),old[0,0]),
                    'source':S.bridge(source['logits'][:,ids].cpu().numpy(),old[1,1]),
                    'bypass':S.bridge(bypass['logits'][:,ids].cpu().numpy(),old[1,0]),
                    'native_decoder_full':bridge(decoded['native'],z0)}
                def objects(v):
                    readers=v[:,ids];return {'margin':readers[:,0]-readers[:,foil],
                        'readers':readers-readers.mean(-1,keepdim=True),
                        'vocabulary':v-v.mean(-1,keepdim=True)}
                e,t,n=objects(error),objects(target),objects(q@z0)
                errors={k:ratio(e[k],t[k]) for k in e};materiality={k:ratio(t[k],n[k]) for k in ('margin','readers')}
                live=all(v is not None and v>=.10 for v in materiality.values())
                reports.append({'world_id':world['world_id'],'layout':world['layout'],'foil':world['foil'],
                    'native_logits':z0[:,ids].cpu().tolist(),'bypass_logits':bypass['logits'][:,ids].cpu().tolist(),
                    'carry_logits':decoded['carry'][:,ids].cpu().tolist(),'bridges':bridges,
                    'fidelity_errors':errors,'bypass_materiality':materiality,
                    'task_passed':live and all(S.passed(errors[k],.10) for k in ('margin','readers')),
                    'vocabulary_passed':S.passed(errors['vocabulary'],.10),
                    'instrument_passed':all(b['max_abs']<=1e-3 and b['relative']<=1e-5 for b in bridges.values()) and bool(torch.isfinite(prediction).all()) and bool(torch.isfinite(target).all())})
    finally:counter.remove()
    restored=not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
    preds={'pred_a_instrument':restored and counts==[192,3072] and decoder_batches==128 and all(r['instrument_passed'] for r in reports),
        'pred_b_task':all(r['task_passed'] for r in reports),'pred_c_vocabulary':all(r['vocabulary_passed'] for r in reports)}
    result={'terminal':'bypass_carry_complete' if preds['pred_a_instrument'] else 'invalid','predictions':preds,'reports':reports,
        'runner_sha256':sha(RUNNER),'binding_sha256':sha(BINDING),'corners':bank['corners'],'wall_seconds':time.perf_counter()-tic,
        'residual_lambdas_9_to_17':lambdas,'carry_scale':gamma,
        'price':{'forwards':counts[0],'sequences':counts[1],'decoder_batches':decoder_batches,'decoder_states':2048,'fits':0,'native_parameters':sum(p.numel() for p in model.parameters()),'actual_weight_saving':0},
        'scope':'Frozen-write direct residual transport with native final reader, compared to fully live A9-clamped bypass. No independent source generation or new OOD evidence.'}
    atomic_create_json(OUT,result)
    print(json.dumps({'predictions':preds,'wall_seconds':result['wall_seconds'],'carry_scale':gamma,'worlds':[{k:r[k] for k in ('world_id','task_passed','vocabulary_passed','fidelity_errors')} for r in reports]}));assert preds['pred_a_instrument']

if __name__=='__main__':main()
