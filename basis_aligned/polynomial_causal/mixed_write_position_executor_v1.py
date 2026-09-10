"""Position partition of a fixed mixed attention write; no role transport."""
import numpy as np
import torch
import circuit_fast_screen_producer as P
import live_value_factorial_executor_v2 as E
import attention_output_delta_v1 as A
import source_margin_gradient as G
from attention_write_factorial_executor_v1 import bridge


def run_world(backend,world,q,readers):
    attn=backend.model.transformer.h[9].attn;writes=[]
    handle=attn.register_forward_hook(lambda _m,_a,out:writes.append(out[0].detach().clone()))
    try:base=E.run_world(backend,world,q,readers)
    finally:handle.remove()
    assert len(writes)==4
    native=torch.cat(writes[:2]);delta=native.double()-torch.cat(writes[2:]).double()
    mixed=torch.einsum('ij,jtd->itd',q,delta)
    common=torch.zeros_like(mixed);common[:,world['common_positions']]=mixed[:,world['common_positions']]
    noun=mixed-common;den=mixed.norm().clamp_min(1e-30)
    closure=float((mixed-common-noun).norm()/den)
    pure=max(float((x-torch.einsum('ij,jtd->itd',q,x)).norm()/den) for x in (common,noun))
    early=float(mixed[:,:world['interaction_start']].norm()/den)
    arms={};impurity=0.;foil=1 if world['foil']=='himself' else 2
    with torch.inference_mode():
        for arm,d in [('full',mixed),('mixed',common),('spill',noun)]:
            # Generic scorer calls the two parts mixed/spill. They are C/N here;
            # both are pure mixed over sentence factors, separated by position.
            chunks=[];actual=[]
            for start in (0,16):
                rows=world['rows'][start:start+16]
                batch=P.ModelBatch(tuple(r['row_id'] for r in rows),'base',tuple(tuple(r['ids']) for r in rows),
                    (readers[0],)*16,(readers[foil],)*16,(world['length']-1,)*16)
                with A.subtract(attn,native[start:start+16],d[start:start+16],actual):
                    with G.capture(backend.model) as captured:backend.native(batch,capture=False)
                chunks.append(G.endpoint_logits(captured['raw_logits'],batch)[:,readers].double())
            arms[arm]=torch.cat(chunks).cpu().tolist()
            if arm=='full':
                installed=torch.cat(actual)
                impurity=float((installed-torch.einsum('ij,jtd->itd',q,installed)).norm()/installed.norm().clamp_min(1e-30))
    valid=base['instrument']['passed'] and max(closure,pure)<=1e-10 and early<=1e-6 and impurity<=1e-4
    valid &= all(np.isfinite(z).all() for z in arms.values())
    return {'world_id':world['world_id'],'layout':world['layout'],'foil':world['foil'],
        'native_logits':base['native_logits'],'arm_logits':arms,
        'arm_meanings':{'full':'complete mixed write','mixed':'common mature positions C','spill':'remaining noun position N'},
        'instrument_passed':bool(valid),'inherited_instrument':base['instrument'],
        'position_partition_closure':closure,'position_partition_purity':pure,
        'early_mixed_relative':early,'installed_full_impurity':impurity,
        'common_positions':world['common_positions'],
        'common_write_norm_fraction':float(common.norm()/den),'noun_write_norm_fraction':float(noun.norm()/den)}
