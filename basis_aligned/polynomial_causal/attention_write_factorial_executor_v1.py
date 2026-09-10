"""Reusable all-position mixed/complement write experiment on a signed cube."""
import numpy as np
import torch
import circuit_fast_screen_producer as P
import live_value_factorial_executor_v2 as E
import attention_output_delta_v1 as A
import source_margin_gradient as G
from factorial_effect_metrics_v1 import measure


def ratio(x,y):
    return float(np.linalg.norm(x)/np.linalg.norm(y)) if np.linalg.norm(y)>0 else None


def passed(value,bar):
    return value is not None and value<=bar


def bridge(x,y):
    x=np.asarray(x);y=np.asarray(y)
    return {'max_abs':float(np.max(abs(x-y))),
            'relative':float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))}


def run_world(backend,world,q,readers,*,full_replay=False):
    attn=backend.model.transformer.h[9].attn;writes=[]
    hook=attn.register_forward_hook(lambda _m,_a,out:writes.append(out[0].detach().clone()))
    try:base=E.run_world(backend,world,q,readers)
    finally:hook.remove()
    assert len(writes)==4
    native=torch.cat(writes[:2]);delta=native.double()-torch.cat(writes[2:]).double()
    mixed=torch.einsum('ij,jtd->itd',q,delta);spill=delta-mixed
    denom=delta.norm().clamp_min(1e-30)
    errors={'closure':float((delta-mixed-spill).norm()/denom),
            'purity':float((torch.einsum('ij,jtd->itd',q,mixed)-mixed).norm()/denom),
            'orthogonality':float(torch.einsum('ij,jtd->itd',q,spill).norm()/denom)}
    early=float(delta[:,:world['interaction_start']].norm()/denom)
    arms={'full':base['edited_logits']};purity=0.
    foil=1 if world['foil']=='himself' else 2
    choices=[('mixed',mixed),('spill',spill)]
    if full_replay:choices.insert(0,('full',delta))
    with torch.inference_mode():
        for arm,d in choices:
            chunks=[];actual=[]
            for start in (0,16):
                rows=world['rows'][start:start+16]
                batch=P.ModelBatch(tuple(r['row_id'] for r in rows),'base',tuple(tuple(r['ids']) for r in rows),
                    (readers[0],)*16,(readers[foil],)*16,(world['length']-1,)*16)
                with A.subtract(attn,native[start:start+16],d[start:start+16],actual):
                    with G.capture(backend.model) as captured:backend.native(batch,capture=False)
                chunks.append(G.endpoint_logits(captured['raw_logits'],batch)[:,readers].double())
            arms[arm]=torch.cat(chunks).cpu().tolist()
            if arm=='mixed':
                installed=torch.cat(actual)
                purity=float((installed-torch.einsum('ij,jtd->itd',q,installed)).norm()/installed.norm().clamp_min(1e-30))
    replay=bridge(arms['full'],base['edited_logits']) if full_replay else None
    instrument=base['instrument']['passed'] and max(errors.values())<=1e-10 and early<=1e-6 and purity<=1e-4
    if replay is not None:instrument &= replay['max_abs']<=1e-3 and replay['relative']<=1e-5
    instrument &= all(np.isfinite(z).all() for z in arms.values())
    return {'world_id':world['world_id'],'layout':world['layout'],'foil':world['foil'],
        'native_logits':base['native_logits'],'arm_logits':arms,'instrument_passed':bool(instrument),
        'inherited_instrument':base['instrument'],'algebra_errors':errors,
        'installed_mixed_impurity':purity,'early_delta_relative':early,'full_replay':replay,
        'local_spill_to_mixed_write_ratio':float(spill.norm()/mixed.norm().clamp_min(1e-30))}


def score(corners,q,report):
    z0=np.asarray(report['native_logits']);foil=1 if report['foil']=='himself' else 2
    effects={k:z0-np.asarray(z) for k,z in report['arm_logits'].items()}
    margins={k:z[:,0]-z[:,foil] for k,z in effects.items()}
    natural=z0[:,0]-z0[:,foil];rms=float(np.sqrt(np.mean((q@natural)**2)))
    metrics={k:measure(corners,natural,natural-v) if np.linalg.norm(q@v)>0 and rms>0 else None for k,v in margins.items()}
    fidelity=ratio(q@(margins['full']-margins['mixed']),q@margins['full'])
    e=effects['mixed'];gender=ratio(q@(e[:,1]-e[:,2]),q@(e[:,0]-(e[:,1]+e[:,2])/2))
    interaction=effects['full']-effects['mixed']-effects['spill']
    cm=ratio(margins['full']-margins['mixed']-margins['spill'],margins['full'])
    cv=ratio(interaction-interaction.mean(1,keepdims=True),effects['full']-effects['full'].mean(1,keepdims=True))
    m=metrics['mixed']
    return {'native_mixed_rms':rms,'metrics':metrics,'mixed_fidelity_error':fidelity,
        'mixed_gender_ratio':gender,'composition_margin_error':cm,'composition_reader_error':cv,
        'transfer_passed':rms>=.05 and m is not None and m['live_natural_mixed_projection']>=.05 and m['nonmixed_to_mixed_margin_ratio']<=.25,
        'fidelity_passed':passed(fidelity,.10),'gender_passed':passed(gender,.25),
        'composition_passed':passed(cm,.10) and passed(cv,.10)}
