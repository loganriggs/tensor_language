"""Shared source-stage execution with optional attention context (completed v1 preserved)."""
from contextlib import ExitStack
import torch
import circuit_fast_screen_producer as P
import module_output_delta_v1 as D
import prefix_memory_port_v1 as M
import source_margin_gradient as G


def run(backend,world,ids,delta=None,native=None,memory_pair=None,attention_context=None):
    model=backend.model;mlp=model.transformer.h[8].mlp;attn=model.transformer.h[9].attn
    foil=1 if world['foil']=='himself' else 2;chunks=[];memory_chunks=[];audit=[];source_audit=[]
    for start in (0,16):
        rows=world['rows'][start:start+16];length=world['length']
        b=P.ModelBatch(tuple(r['row_id'] for r in rows),'base',tuple(tuple(r['ids']) for r in rows),
            (ids[0],)*16,(ids[foil],)*16,(length-1,)*16)
        record={};handles=[]
        try:
            handles.append(mlp.register_forward_hook(lambda _m,_a,out:record.update(mlp=out.detach().clone())))
            handles.append(attn.register_forward_hook(lambda _m,_a,out:record.update(first=out[1].detach().clone())))
            with ExitStack() as stack:
                if delta is not None:
                    stack.enter_context(D.subtract(mlp,native['mlp'][start:start+16],delta[start:start+16],source_audit))
                if attention_context is not None:
                    assert memory_pair is None
                    stack.enter_context(attention_context(start,start+16))
                elif memory_pair is None:
                    memory=stack.enter_context(M.capture(model))
                else:
                    donor,recipient=memory_pair
                    stack.enter_context(M.replace(model,M.slice_batch(donor,start,start+16),M.slice_batch(recipient,start,start+16),audit))
                with G.capture(model) as g:backend.native(b,capture=False)
            if memory_pair is None and attention_context is None:memory_chunks.append(memory)
            record['logits']=G.endpoint_logits(g['raw_logits'],b).double()
        finally:
            for h in handles:h.remove()
        chunks.append(record)
    result={k:torch.cat([r[k] for r in chunks]) for k in chunks[0]}
    result['memory']=M.join_batches(memory_chunks) if memory_chunks else None
    result['memory_audit']=audit
    return result
