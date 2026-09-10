#!/usr/bin/env python3
# BQGATE: final-reader interaction;512forwards8192seq+192decoderbatches6144states,0fits.
"""A instrument; B final-reader sufficiency; C internal sufficiency, .10 allreadouts/worlds."""
import hashlib,json,os,signal,sys,time
from pathlib import Path
from contextlib import ExitStack
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path.insert(0,str(POLY));sys.path.insert(0,str(ROOT))
import numpy as np
import circuit_fast_screen_producer as P
import mixed_state_projector_v1 as Q
import module_output_delta_v1 as D
import readout_interaction_decomposition_v1 as R
import intervening_write_clamp as W
import source_margin_gradient as G
import attention_write_factorial_executor_v1 as S
from circuit_fast_screen_managed_runner import atomic_create_json
BINDING=POLY/'MLP8_COUPLING_READOUT_V1_BINDING.json';OUT=POLY/'MLP8_COUPLING_READOUT_V1_RESULT.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    binding=json.loads(BINDING.read_text());assert all(sha(p)==v for p,v in binding.items())
    bank=json.loads((POLY/'THIRD_NOUN_CAUSAL_ROLE_ALIGNMENT_V1.json').read_text())
    parent=json.loads((POLY/'MLP8_COUPLING_DIRECTIONS_V1_RESULT.json').read_text());assert parent['predictions']['pred_a_instrument']
    lookup={r['world_id']:r for r in parent['reports']}
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'gpu_accessed':False,'model_loaded':False,'forwards':512,'sequences':8192,'decoder_batches':192,'decoder_states':6144,'fits':0}));return
    assert not OUT.exists();signal.alarm(600)
    backend=P.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    model=backend.model;mlp=model.transformer.h[8].mlp;attn=model.transformer.h[9].attn
    q_np=Q.projector(bank['corners'],2,4);q=torch.tensor(q_np,device='cuda',dtype=torch.float64)
    counts=[0,0];reports=[];tic=time.perf_counter()
    def count(_m,args,_out):counts[0]+=1;counts[1]+=len(args[0])
    counter=model.transformer.h[0].attn.register_forward_hook(count)
    try:
        with torch.inference_mode():
            for world in bank['worlds']:
                foil=1 if world['foil']=='himself' else 2;ids=bank['reader_ids'];length=world['length']
                def batch(start):
                    rows=world['rows'][start:start+16]
                    return P.ModelBatch(tuple(r['row_id'] for r in rows),'base',tuple(tuple(r['ids']) for r in rows),(ids[0],)*16,(ids[foil],)*16,(length-1,)*16)
                def run(b):
                    record={};handles=[]
                    try:
                        handles.append(mlp.register_forward_hook(lambda _m,_a,out:record.update(mlp=out.detach().clone())))
                        handles.append(attn.register_forward_hook(lambda _m,_a,out:record.update(first=out[1].detach().clone())))
                        with W.capture(model,layers=range(9,18)) as writes:
                            with G.capture(model) as g:
                                trace=backend.native(b,capture=True)
                        record['final']=torch.stack([trace.captured[(row_id,'resid:18')] for row_id in b.row_ids]).to('cuda')
                        record['writes']=writes;record['logits']=G.endpoint_logits(g['raw_logits'],b).double()
                    finally:
                        for h in handles:h.remove()
                    return record
                def join(chunks):
                    result={k:torch.cat([r[k] for r in chunks]) for k in chunks[0] if k!='writes'}
                    result['writes']={k:torch.cat([r['writes'][k] for r in chunks]) for k in chunks[0]['writes']}
                    return result
                native=join([run(batch(start)) for start in (0,16)])
                delta=torch.einsum('ij,jtd->itd',q,native['mlp'].double());audit=[];first_ok=True;frozen_ok=True
                def arm(a_bank,m_bank,source_edit=True):
                    nonlocal first_ok,frozen_ok
                    chunks=[]
                    for start in (0,16):
                        values={k:v[start:start+16] for k,v in native['writes'].items()}
                        frozen={('attention',9):values['attention',9]}
                        source_delta=delta[start:start+16] if source_edit else torch.zeros_like(delta[start:start+16])
                        with ExitStack() as stack:
                            stack.enter_context(D.subtract(mlp,native['mlp'][start:start+16],source_delta,audit))
                            stack.enter_context(W.clamp(model,values,(length,)*16,('attention',),layers=(9,)))
                            for incoming,kind,layers in [(a_bank,'attention',range(10,18)),(m_bank,'mlp',range(9,18))]:
                                if incoming is not None:
                                    selected={k:v[start:start+16] for k,v in incoming.items()}
                                    stack.enter_context(W.clamp(model,selected,(length,)*16,(kind,),layers=layers))
                                    frozen.update({(kind,i):selected[kind,i] for i in layers})
                            record=run(batch(start))
                        frozen_ok &= all(torch.equal(record['writes'][k],v) for k,v in frozen.items())
                        first_ok &= torch.equal(record['first'],native['first'][start:start+16]);chunks.append(record)
                    return join(chunks)
                bank_a=arm(None,native['writes']);bank_m=arm(native['writes'],None)
                identity=arm(native['writes'],native['writes'],False)
                cube=[];state_cube=[]
                for am in (0,1):
                    row=[];state_row=[]
                    for ma in (0,1):
                        record=arm(None if ma else bank_a['writes'],None if am else bank_m['writes'])
                        row.append(record['logits'].cpu().numpy());state_row.append(record['final'])
                    cube.append(row);state_cube.append(torch.stack(state_row))
                cube=np.array(cube);z0=native['logits'].cpu().numpy();old=lookup[world['world_id']];old_cube=np.array(old['cube_logits'])
                states=torch.stack(state_cube)
                def decode(state):
                    normalized=torch.nn.functional.rms_norm(state.float(),(state.shape[-1],))
                    return (30*torch.tanh(model.lm_head(normalized)/30)).double().cpu().numpy()
                additive=(states[1,0].double()+states[0,1].double()-states[0,0].double()).float()
                za=decode(additive);parts=R.decompose(cube,za)
                bridges={'native':S.bridge(z0[:,ids],old['native_logits']),
                    'native_identity_full':S.bridge(identity['logits'].cpu().numpy(),z0),
                    'native_decoder':S.bridge(decode(native['final']),z0)}
                for am in (0,1):
                    for ma in (0,1):
                        bridges[f'cube_{am}{ma}']=S.bridge(cube[am,ma][:,ids],old_cube[am,ma])
                        bridges[f'decoder_{am}{ma}']=S.bridge(decode(states[am,ma]),cube[am,ma])
                total=cube[0,0]-cube[1,1];am_effect=cube[0,0]-cube[1,0];ma_effect=cube[0,0]-cube[0,1];interaction=total-am_effect-ma_effect
                def norms(x,y):
                    xr=x[:,ids];yr=y[:,ids]
                    return {'margin':S.ratio(xr[:,0]-xr[:,foil],yr[:,0]-yr[:,foil]),
                        'readers':S.ratio(xr-xr.mean(-1,keepdims=True),yr-yr.mean(-1,keepdims=True)),
                        'vocabulary':S.ratio(x-x.mean(-1,keepdims=True),y-y.mean(-1,keepdims=True))}
                errors={name:{'mixed':norms(q_np@part,q_np@parts['interaction']),
                    'full':norms(part,parts['interaction'])}
                    for name,part in [('reader_sufficiency',parts['internal']),('internal_sufficiency',parts['readout'])]}
                closure=float(np.max(np.abs(parts['interaction']-parts['readout']-parts['internal'])))
                reports.append({'world_id':world['world_id'],'layout':world['layout'],'foil':world['foil'],
                    'native_logits':z0[:,ids].tolist(),'cube_logits':cube[...,ids].tolist(),
                    'additive_endpoint_logits':za[:,ids].tolist(),'bridges':bridges,'errors':errors,
                    'raw_interaction_norm':float(torch.linalg.vector_norm(states[1,1].double()-additive.double())),
                    'closure_max_abs':closure,
                    'reader_passed':all(S.passed(v,.10) for obj in errors['reader_sufficiency'].values() for v in obj.values()),
                    'internal_passed':all(S.passed(v,.10) for obj in errors['internal_sufficiency'].values() for v in obj.values()),
                    'instrument_passed':first_ok and frozen_ok and np.isfinite(cube).all() and np.isfinite(za).all()
                        and closure<=1e-10 and all(b['max_abs']<=1e-3 and b['relative']<=1e-5 for b in bridges.values())})
    finally:counter.remove()
    restored=not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
    preds={'pred_a_instrument':restored and counts==[512,8192] and all(r['instrument_passed'] for r in reports),
        'pred_b_reader':all(r['reader_passed'] for r in reports),
        'pred_c_internal':all(r['internal_passed'] for r in reports)}
    result={'terminal':'coupling_readout_complete' if preds['pred_a_instrument'] else 'invalid','predictions':preds,'reports':reports,
        'runner_sha256':sha(RUNNER),'binding_sha256':sha(BINDING),'corners':bank['corners'],'wall_seconds':time.perf_counter()-tic,
        'price':{'forwards':counts[0],'sequences':counts[1],'decoder_batches':192,'decoder_states':6144,'fits':0,'native_parameters':sum(p.numel() for p in model.parameters()),'actual_weight_saving':0},
        'scope':'Baseline-anchored decomposition of directional-switch interaction into final-reader and internal-state terms. Synthetic endpoint edit; native context and all weights retained; not an independently extracted circuit.'}
    atomic_create_json(OUT,result)
    print(json.dumps({'predictions':preds,'wall_seconds':result['wall_seconds'],'worlds':[{k:r[k] for k in ('world_id','reader_passed','internal_passed','errors')} for r in reports]}));assert preds['pred_a_instrument']

if __name__=='__main__':main()
