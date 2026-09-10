#!/usr/bin/env python3
# BQGATE: cross-family direction cut cube;512forwards8192seq,0fits.
"""A instrument; B A-to-M; C M-to-A; D interaction, .10 allreadouts/worlds."""
import hashlib,json,os,signal,sys,time
from pathlib import Path
from contextlib import ExitStack
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path.insert(0,str(POLY));sys.path.insert(0,str(ROOT))
import numpy as np
import circuit_fast_screen_producer as P
import mixed_state_projector_v1 as Q
import module_output_delta_v1 as D
import intervening_write_clamp as W
import source_margin_gradient as G
import attention_write_factorial_executor_v1 as S
from circuit_fast_screen_managed_runner import atomic_create_json
BINDING=POLY/'MLP8_COUPLING_DIRECTIONS_V1_BINDING.json';OUT=POLY/'MLP8_COUPLING_DIRECTIONS_V1_RESULT.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    binding=json.loads(BINDING.read_text());assert all(sha(p)==v for p,v in binding.items())
    bank=json.loads((POLY/'THIRD_NOUN_CAUSAL_ROLE_ALIGNMENT_V1.json').read_text())
    parent=json.loads((POLY/'MLP8_LATE_GROUPS_V2_RESULT.json').read_text());assert parent['predictions']['pred_a_instrument']
    lookup={r['world_id']:r for r in parent['reports']}
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'gpu_accessed':False,'model_loaded':False,'forwards':512,'sequences':8192,'fits':0}));return
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
                            with G.capture(model) as g:backend.native(b,capture=False)
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
                cube=[]
                for am in (0,1):
                    row=[]
                    for ma in (0,1):
                        record=arm(None if ma else bank_a['writes'],None if am else bank_m['writes'])
                        row.append(record['logits'].cpu().numpy())
                    cube.append(row)
                cube=np.array(cube);z0=native['logits'].cpu().numpy();old=lookup[world['world_id']];old_cube=np.array(old['cube_logits'])
                bridges={'native':S.bridge(z0[:,ids],old['native_logits']),
                    'native_identity_full':S.bridge(identity['logits'].cpu().numpy(),z0),
                    'bank_a':S.bridge(bank_a['logits'][:,ids].cpu().numpy(),old_cube[1,0]),
                    'bank_m':S.bridge(bank_m['logits'][:,ids].cpu().numpy(),old_cube[0,1]),
                    'both_live':S.bridge(cube[1,1][:,ids],old_cube[1,1])}
                total=cube[0,0]-cube[1,1];am_effect=cube[0,0]-cube[1,0];ma_effect=cube[0,0]-cube[0,1];interaction=total-am_effect-ma_effect
                def norms(x,y):
                    xr=x[:,ids];yr=y[:,ids]
                    return {'margin':S.ratio(xr[:,0]-xr[:,foil],yr[:,0]-yr[:,foil]),
                        'readers':S.ratio(xr-xr.mean(-1,keepdims=True),yr-yr.mean(-1,keepdims=True)),
                        'vocabulary':S.ratio(x-x.mean(-1,keepdims=True),y-y.mean(-1,keepdims=True))}
                tq=q_np@total;materiality=norms(tq,q_np@(z0-cube[1,1]))
                errors={k:norms(q_np@v-tq,tq) for k,v in [('a_to_m',am_effect),('m_to_a',ma_effect)]}
                interactions={'mixed':norms(q_np@interaction,tq),'full':norms(interaction,total)}
                live=materiality['vocabulary'] is not None and materiality['vocabulary']>=.05
                reports.append({'world_id':world['world_id'],'layout':world['layout'],'foil':world['foil'],
                    'native_logits':z0[:,ids].tolist(),'cube_logits':cube[...,ids].tolist(),'bridges':bridges,
                    'group_axes':['a_to_m_enabled','m_to_a_enabled'],'response_baseline':'both independent response banks installed',
                    'dominance_errors':errors,'coupling_materiality':materiality,'interaction_errors':interactions,
                    'a_to_m_passed':live and all(S.passed(v,.10) for v in errors['a_to_m'].values()),
                    'm_to_a_passed':live and all(S.passed(v,.10) for v in errors['m_to_a'].values()),
                    'task_a_to_m_passed':live and all(S.passed(errors['a_to_m'][k],.10) for k in ('margin','readers')),
                    'task_m_to_a_passed':live and all(S.passed(errors['m_to_a'][k],.10) for k in ('margin','readers')),
                    'interaction_passed':all(S.passed(v,.10) for obj in interactions.values() for v in obj.values()),
                    'instrument_passed':first_ok and frozen_ok and np.isfinite(cube).all() and all(b['max_abs']<=1e-3 and b['relative']<=1e-5 for b in bridges.values())})
    finally:counter.remove()
    restored=not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
    preds={'pred_a_instrument':restored and counts==[512,8192] and all(r['instrument_passed'] for r in reports),
        'pred_b_a_to_m':all(r['a_to_m_passed'] for r in reports),'pred_c_m_to_a':all(r['m_to_a_passed'] for r in reports),
        'pred_d_interaction':all(r['interaction_passed'] for r in reports)}
    result={'terminal':'coupling_directions_complete' if preds['pred_a_instrument'] else 'invalid','predictions':preds,'reports':reports,
        'runner_sha256':sha(RUNNER),'binding_sha256':sha(BINDING),'corners':bank['corners'],'wall_seconds':time.perf_counter()-tic,
        'price':{'forwards':counts[0],'sequences':counts[1],'fits':0,'native_parameters':sum(p.numel() for p in model.parameters()),'actual_weight_saving':0},
        'scope':'Counterfactual write-bank directional cuts in the coupled late response. Native context and all weights retained; not a nonlinear path-count theorem or independent circuit.'}
    atomic_create_json(OUT,result)
    print(json.dumps({'predictions':preds,'wall_seconds':result['wall_seconds'],'worlds':[{k:r[k] for k in ('world_id','a_to_m_passed','m_to_a_passed','interaction_passed','dominance_errors')} for r in reports]}));assert preds['pred_a_instrument']

if __name__=='__main__':main()
