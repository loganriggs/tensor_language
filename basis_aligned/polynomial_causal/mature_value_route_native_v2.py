"""Native factors plus raw RMS inputs; completed v1 behavior preserved."""
import torch
import circuit_fast_screen_producer as P
import value_lineage_capture as C
import norm_preserving_response_hooks as H
import source_margin_gradient as G


def contract(pattern,value,output_weight,local_scale):
    read=torch.einsum('nhqs,nshd->nqhd',pattern,value)
    return local_scale*(read.flatten(-2)@output_weight.T)


def capture(backend,world,q,mean,readers):
    model=backend.model;attn=model.transformer.h[9].attn;record={};bases=[];inputs=[];values=[];raws=[];recurrence=[]
    length=world['length'];assert world['interaction_start']==length-3
    foil=1 if world['foil']=='himself' else 2
    def output(_m,_a,out):record.update(write=out[0].detach().clone(),first=out[1].detach().clone())
    handle=attn.register_forward_hook(output)
    def batch(start):
        rows=world['rows'][start:start+16]
        return P.ModelBatch(tuple(r['row_id'] for r in rows),'base',tuple(tuple(r['ids']) for r in rows),
            (readers[0],)*16,(readers[foil],)*16,(length-1,)*16)
    def run(b):
        with H.capture(model,layers=(9,)) as h:
            with G.capture(model) as g:backend.native(b,capture=False)
        return {'logits':G.endpoint_logits(g['raw_logits'],b)[:,readers].double(),
                'factors':h[9]['factors'],'write':record['write'],'first':record['first']}
    try:
        with torch.inference_mode():
            for start in (0,16):
                with C.capture(model,source=8,target=9) as c:base=run(batch(start))
                bases.append(base);inputs.append(c['normalized_input']);values.append(c['value'])
                raws.append(c['residual'].detach().clone());recurrence.append(c['recurrence_bitwise'])
            u=torch.cat(inputs).double();local=torch.cat(values).double()
            mixed_u=torch.einsum('ij,jtd->itd',q,u)
            delta=mixed_u@attn.c_v.weight.double().T
            local_mixed=torch.einsum('ij,jtd->itd',q,local)
            commute=float((delta-local_mixed).norm()/local_mixed.norm().clamp_min(1e-30))
            early=float(mixed_u[:,:length-3].norm()/u.norm().clamp_min(1e-30))
            replacement=(local-delta).float().view(32,length,9,128)
            changes=[];qk=True;first=True
            for i,start in enumerate((0,16)):
                with C.replace_value(attn,replacement[start:start+16],(length,)*16,heads=tuple(range(9))):changed=run(batch(start))
                changes.append(changed)
                qk &= all(torch.equal(bases[i]['factors'][k],changed['factors'][k]) for k in (0,1,3,4))
                first &= torch.equal(bases[i]['first'],changed['first'])
            native_write=torch.cat([b['write'] for b in bases])
            difference=native_write.double()-torch.cat([c['write'] for c in changes]).double()
            observed=torch.einsum('ij,jtd->itd',q,difference)[:,-2:]
            f=[torch.cat([b['factors'][i] for b in bases]).double() for i in range(5)]
            pattern=(torch.einsum('nqhd,nshd->nhqs',f[0][:,-2:],f[1][:,-3:])/128)*(
                     torch.einsum('nqhd,nshd->nhqs',f[3][:,-2:],f[4][:,-3:])/128)
            pattern[:,:,0,2]=0
            p0=torch.einsum('ij,jhqs->ihqs',mean,pattern)
            v=delta[:,-3:].view(32,3,9,128)
            scale=1-float(attn.lamb);assert scale==1.65625
            compiled=contract(p0,v,attn.c_proj.weight.double(),scale)
            bridge=float((compiled-observed).norm()/observed.norm().clamp_min(1e-30))
            c=torch.zeros_like(native_write,dtype=torch.float64);c[:,-2:]=observed
            valid=commute<=1e-4 and early<=1e-6 and qk and first and bridge<=1e-4
            valid &= bool(compiled.isfinite().all())
    finally:handle.remove()
    return {'native_logits':torch.cat([b['logits'] for b in bases]).cpu().tolist(),
            'native_write':native_write,'observed_common':c,'pattern':p0,'value':v,
            'raw_inputs':torch.cat(raws),'normalized_inputs':u,'recurrence_bitwise':all(recurrence),
            'compiled_common':compiled,'instrument':{'passed':bool(valid),
                'weight_commutation_relative':commute,'early_input_mixed_relative':early,
                'qk_unchanged':qk,'first_value_unchanged':first,'native_component_bridge_relative':bridge}}


def controls(corners):
    import numpy as np
    import mature_value_route_contract_v1 as M
    rng=np.random.default_rng(910716)
    p=rng.normal(size=(32,2,2,3));v=rng.normal(size=(32,3,2,4));w=rng.normal(size=(5,2,4))
    predicted=contract(torch.tensor(p),torch.tensor(v),torch.tensor(w.reshape(5,8)),1.65625).numpy()
    reference=M.contract(p,v,w,1.65625)
    error=float(np.max(abs(predicted-reference)));assert error<1e-12
    return {'passed':True,'factorized_output_map_error':error,'conditional_contract':M.controls(corners)}
