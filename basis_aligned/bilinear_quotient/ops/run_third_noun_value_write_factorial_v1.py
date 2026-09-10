#!/usr/bin/env python3
# BQGATE: live mixed/complement attention-write factorial; fixed 160 forwards, no fits.
"""A instrument; B mixed fidelity .10; C spill .25; D gender .25; E composition .10.

All 16 frozen worlds. Full-write, mixed-write, complement-write removals at L9
with native suffix. Nulls preserve failed branches; no rank, head or row tuning.
Price 160 forwards / 2560 sequences, all 545902902 native parameters retained.
"""
import hashlib,json,os,signal,sys,time
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path.insert(0,str(POLY));sys.path.insert(0,str(ROOT))
import numpy as np
import circuit_fast_screen_producer as P
import live_value_factorial_executor_v1 as E
import mixed_state_projector_v1 as Q
import source_margin_gradient as G
import attention_output_delta_v1 as A
from factorial_effect_metrics_v1 import measure
from circuit_fast_screen_managed_runner import atomic_create_json
BINDING=POLY/'THIRD_NOUN_VALUE_WRITE_FACTORIAL_V1_BINDING.json'
OUT=POLY/'THIRD_NOUN_VALUE_WRITE_FACTORIAL_V1_RESULT.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def bridge(x,y):
    x=np.asarray(x);y=np.asarray(y)
    return {'max_abs':float(np.max(abs(x-y))), 'relative':float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))}


def main():
    binding=json.loads(BINDING.read_text());assert all(sha(p)==v for p,v in binding.items())
    source=json.loads((POLY/'THIRD_NOUN_VALUE_TRANSFER_V1_ROWS.json').read_text())
    parent=json.loads((POLY/'THIRD_NOUN_VALUE_TRANSFER_V1_RESULT.json').read_text())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'gpu_accessed':False,'model_loaded':False,'forwards':160,'sequences':2560,'fits':0}));return
    assert not OUT.exists();signal.alarm(600)
    backend=P.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    counts=[0,0];reports=[];tic=time.perf_counter()
    def count(_m,args,_out):counts[0]+=1;counts[1]+=len(args[0])
    count_handle=backend.model.transformer.h[0].attn.register_forward_hook(count)
    q_np=Q.projector(source['corners'],2,4);q=torch.tensor(q_np,device='cuda',dtype=torch.float64)
    attn=backend.model.transformer.h[9].attn
    ratio=lambda x,y:float(np.linalg.norm(x)/np.linalg.norm(y)) if np.linalg.norm(y)>0 else None
    passed=lambda x,bar:x is not None and x<=bar
    try:
        with torch.inference_mode():
            for world,old in zip(source['worlds'],parent['reports'],strict=True):
                assert world['world_id']==old['world_id']
                writes=[]
                hook=attn.register_forward_hook(lambda _m,_a,out:writes.append(out[0].detach().clone()))
                try:base=E.run_world(backend,world,q,source['reader_ids'])
                finally:hook.remove()
                assert len(writes)==4
                native=torch.cat(writes[:2]);delta=native.double()-torch.cat(writes[2:]).double()
                mixed=torch.einsum('ij,jtd->itd',q,delta);spill=delta-mixed
                denom=delta.norm().clamp_min(1e-30)
                closure=float((delta-mixed-spill).norm()/denom)
                pure=float((torch.einsum('ij,jtd->itd',q,mixed)-mixed).norm()/denom)
                orthogonal=float(torch.einsum('ij,jtd->itd',q,spill).norm()/denom)
                early=float(delta[:,:7].norm()/denom)
                arms={};purity=[]
                foil=1 if world['foil']=='himself' else 2
                for arm,d in [('full',delta),('mixed',mixed),('spill',spill)]:
                    chunks=[];actual=[]
                    for start in (0,16):
                        rows=world['rows'][start:start+16]
                        batch=P.ModelBatch(tuple(r['row_id'] for r in rows),'base',tuple(tuple(r['ids']) for r in rows),
                            (source['reader_ids'][0],)*16,(source['reader_ids'][foil],)*16,(9,)*16)
                        with A.subtract(attn,native[start:start+16],d[start:start+16],actual):
                            with G.capture(backend.model) as captured:backend.native(batch,capture=False)
                        chunks.append(G.endpoint_logits(captured['raw_logits'],batch)[:,source['reader_ids']].double())
                    arms[arm]=torch.cat(chunks).cpu().numpy()
                    if arm=='mixed':
                        installed=torch.cat(actual)
                        purity.append(float((installed-torch.einsum('ij,jtd->itd',q,installed)).norm()/installed.norm().clamp_min(1e-30)))
                z0=np.array(base['native_logits']);zv=np.array(base['edited_logits'])
                bridges={'native_parent':bridge(z0,old['native_logits']),'edited_parent':bridge(zv,old['edited_logits']),
                         'full_replay':bridge(arms['full'],zv)}
                effect={k:z0-z for k,z in arms.items()}
                margins={k:z[:,0]-z[:,foil] for k,z in effect.items()}
                natural=z0[:,0]-z0[:,foil]
                metrics={}
                for k in ('full','mixed','spill'):
                    metrics[k]=measure(source['corners'],natural,natural-margins[k]) if np.linalg.norm(q_np@margins[k])>0 and np.linalg.norm(q_np@natural)>0 else None
                fidelity=ratio(q_np@(margins['full']-margins['mixed']),q_np@margins['full'])
                e=effect['mixed'];gender=ratio(q_np@(e[:,1]-e[:,2]),q_np@(e[:,0]-(e[:,1]+e[:,2])/2))
                interaction=effect['full']-effect['mixed']-effect['spill']
                cm=ratio(margins['full']-margins['mixed']-margins['spill'],margins['full'])
                cv=ratio(interaction-interaction.mean(1,keepdims=True),effect['full']-effect['full'].mean(1,keepdims=True))
                instrument=base['instrument']['passed'] and max(closure,pure,orthogonal)<=1e-10 and early<=1e-6 and max(purity)<=1e-4 and all(
                    b['max_abs']<=1e-3 and b['relative']<=1e-5 for b in bridges.values()) and all(np.isfinite(z).all() for z in arms.values())
                reports.append({'world_id':world['world_id'],'foil':world['foil'],'instrument_passed':bool(instrument),
                    'bridges':bridges,'write_closure':closure,'write_purity':pure,'write_spill_orthogonality':orthogonal,
                    'installed_mixed_write_impurity':max(purity),'early_delta_relative':early,
                    'local_spill_to_mixed_write_ratio':float(spill.norm()/mixed.norm().clamp_min(1e-30)),
                    'mixed_fidelity_error':fidelity,'mixed_gender_ratio':gender,'composition_margin_error':cm,'composition_reader_error':cv,
                    'metrics':metrics,'native_logits':z0.tolist(),'arm_logits':{k:z.tolist() for k,z in arms.items()},
                    'b':passed(fidelity,.10),'c':metrics['mixed'] is not None and metrics['mixed']['nonmixed_to_mixed_margin_ratio']<=.25,
                    'd':passed(gender,.25),'e':passed(cm,.10) and passed(cv,.10)})
    finally:count_handle.remove()
    restored=not any(m._forward_hooks or m._forward_pre_hooks for m in backend.model.modules()) and all('squared_attention' not in b.attn.__dict__ for b in backend.model.transformer.h)
    preds={'pred_a_instrument':restored and counts==[160,2560] and all(r['instrument_passed'] for r in reports),
        'pred_b_mixed_fidelity':all(r['b'] for r in reports),'pred_c_mixed_selectivity':all(r['c'] for r in reports),
        'pred_d_mixed_gender':all(r['d'] for r in reports),'pred_e_composition':all(r['e'] for r in reports)}
    result={'terminal':'write_factorial_complete' if preds['pred_a_instrument'] else 'invalid','predictions':preds,'reports':reports,
        'runner_sha256':sha(RUNNER),'binding_sha256':sha(BINDING),'corners':source['corners'],'wall_seconds':time.perf_counter()-tic,
        'price':{'forwards':counts[0],'sequences':counts[1],'fits':0,'native_parameters':sum(p.numel() for p in backend.model.parameters()),'actual_weight_saving':0},
        'scope':'New intervention on opened cases. Native-derived all-position attention-write factors with live suffix, not independent producer extraction or new OOD evidence.'}
    atomic_create_json(OUT,result)
    print(json.dumps({'predictions':preds,'wall_seconds':result['wall_seconds'],'worlds':[{k:r[k] for k in ('world_id','b','c','d','e','mixed_fidelity_error','mixed_gender_ratio','composition_margin_error','composition_reader_error')}|
        {'mixed_spill':None if r['metrics']['mixed'] is None else r['metrics']['mixed']['nonmixed_to_mixed_margin_ratio']} for r in reports]}));assert preds['pred_a_instrument']


if __name__=='__main__':main()
