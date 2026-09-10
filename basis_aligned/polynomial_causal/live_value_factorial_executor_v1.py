"""Reusable native per-world executor for the frozen layer9 value operator."""
import torch
import circuit_fast_screen_producer as P
import value_lineage_capture as C
import norm_preserving_response_hooks as H
import source_margin_gradient as G


def run_world(backend, world, projector, reader_ids, *, identity=False):
    rows=world['rows'];assert len(rows)==32 and all(len(r['ids'])==10 and r['semantic_position']==9 for r in rows)
    attn=backend.model.transformer.h[9].attn;record={};baselines=[];inputs=[];values=[]
    foil=1 if world['foil']=='himself' else 2
    def first(_m,_a,y):record['first']=y[1].detach().clone()
    handle=attn.register_forward_hook(first)
    def batch(start):
        part=rows[start:start+16]
        return P.ModelBatch(tuple(r['row_id'] for r in part),'base',tuple(tuple(r['ids']) for r in part),
                            (reader_ids[0],)*16,(reader_ids[foil],)*16,(9,)*16)
    def run(b):
        with H.capture(backend.model,layers=(9,)) as h:
            with G.capture(backend.model) as g:out=backend.native(b,capture=False)
        z=G.endpoint_logits(g['raw_logits'],b)[:,reader_ids].double()
        expected=torch.tensor([a-f for a,f in out.answer_foil],device=z.device,dtype=torch.float64)
        assert torch.equal(z[:,0]-z[:,foil],expected)
        return z,h[9]['factors'],record['first'].clone()
    try:
        with torch.inference_mode():
            for start in (0,16):
                with C.capture(backend.model,source=8,target=9) as c:base=run(batch(start))
                baselines.append(base);inputs.append(c['normalized_input']);values.append(c['value'])
            u=torch.cat(inputs).double();v=torch.cat(values).double()
            mixed=lambda x:torch.einsum('ij,jtd->itd',projector,x)
            delta=mixed(u)@attn.c_v.weight.double().T
            reference=mixed(v);den=float(reference.norm())
            commute=float((delta-reference).norm()/reference.norm().clamp_min(1e-30))
            early=float(mixed(u)[:,:7].norm()/u.norm().clamp_min(1e-30))
            replacement=(v-delta).float().view(32,10,9,128)
            edited=[];identity_ok=True;qk=True;first_ok=True
            for bi,start in enumerate((0,16)):
                b=batch(start);base=baselines[bi]
                if identity:
                    with C.replace_value(attn,values[bi].view(16,10,9,128),(10,)*16,heads=tuple(range(9))):same=run(b)
                    identity_ok &= torch.equal(same[0],base[0])
                    first_ok &= torch.equal(same[2],base[2])
                    qk &= all(torch.equal(same[1][i],base[1][i]) for i in (0,1,3,4))
                with C.replace_value(attn,replacement[start:start+16],(10,)*16,heads=tuple(range(9))):changed=run(b)
                edited.append(changed[0]);first_ok &= torch.equal(changed[2],base[2])
                qk &= all(torch.equal(changed[1][i],base[1][i]) for i in (0,1,3,4))
            z0=torch.cat([b[0] for b in baselines]);z1=torch.cat(edited)
            finite=bool(z0.isfinite().all() and z1.isfinite().all())
    finally:handle.remove()
    return {'world_id':world['world_id'],'panel':world['panel'],'foil':world['foil'],
        'native_logits':z0.cpu().tolist(),'edited_logits':z1.cpu().tolist(),
        'instrument':{'weight_commutation_relative':commute,'early_input_mixed_relative':early,
            'local_mixed_value_norm':den,'identity_run':identity,'identity_passed':identity_ok,
            'qk_unchanged':qk,'first_value_unchanged':first_ok,
            'passed':commute<=1e-4 and early<=1e-6 and identity_ok and qk and first_ok and finite}}
