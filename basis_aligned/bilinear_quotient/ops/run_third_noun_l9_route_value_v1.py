#!/usr/bin/env python3
# BQGATE: all layer9 heads/sources, symmetric routing/value partition, no fits.
"""A instrument; B materiality; C routing; D value; E cross-term fidelity."""
import hashlib,json,os,signal,sys,time
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path.insert(0,str(POLY));sys.path.insert(0,str(ROOT))
import circuit_fast_screen_producer as P
import norm_preserving_response_hooks as H
import attention_route_value_partition_v1 as S
from circuit_fast_screen_managed_runner import atomic_create_json
BINDING=POLY/'THIRD_NOUN_L9_ROUTE_VALUE_V1_BINDING.json'
OUT=POLY/'THIRD_NOUN_L9_ROUTE_VALUE_V1_RESULT.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    binding=json.loads(BINDING.read_text());assert all(sha(p)==v for p,v in binding.items())
    source=json.loads((POLY/'THIRD_NOUN_ANIMACY_V1_ROWS.json').read_text())
    parent=json.loads((POLY/'THIRD_NOUN_ANIMACY_V1_RESULT.json').read_text())
    controls=json.loads((POLY/'ATTENTION_ROUTE_VALUE_PARTITION_V1_CONTROLS.json').read_text())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'gpu_accessed':False,'model_loaded':False,'forwards':16,'sequences':256,
                          'decoder_batches':80,'controls_passed':all(x['passed'] for x in controls.values())}));return
    assert not OUT.exists();signal.alarm(600)
    backend=P.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    import torch.nn.functional as F
    tic=time.perf_counter();counts=[0,0];decoder_batches=0;native=[];factors=[[] for _ in range(5)]
    finals=[];writes=[];rows=[r for w in source['worlds'] for r in w['rows']]
    assert all(len(r['ids'])==10 and r['semantic_position']==9 for r in rows)
    def count(_m,args,_y):counts[0]+=1;counts[1]+=len(args[0])
    handle=backend.model.transformer.h[0].attn.register_forward_hook(count)
    try:
        with torch.inference_mode():
            for start in range(0,256,16):
                part=rows[start:start+16]
                batch=P.ModelBatch(tuple(r['row_id'] for r in part),'base',tuple(tuple(r['ids']) for r in part),
                    (source['answer_id'],)*16,(source['foil_id'],)*16,(9,)*16)
                with H.capture(backend.model,layers=(9,)) as captured:out=backend.native(batch,capture=True)
                native.extend(a-b for a,b in out.answer_foil)
                for i,f in enumerate(captured[9]['factors']):factors[i].append(f)
                finals.append(torch.stack([out.captured[r['row_id'],'resid:18'] for r in part]))
                writes.append(torch.stack([out.captured[r['row_id'],'attn:09'] for r in part]))
            q,k,v,q2,k2=[torch.cat(x).double() for x in factors]
            score=lambda a,b:torch.einsum('bhd,bshd->bhs',a[:,9],b)/a.shape[-1]
            pattern=(score(q,k)*score(q2,k2)).view(8,32,9,10)
            value=v.permute(0,2,1,3).reshape(8,32,9,10,128)
            projections=S.projectors(source['corners']).to('cuda')
            def mixed(x):return torch.einsum('ij,wj...->wi...',projections[3],x)
            parts,edges,full=S.partition(pattern,value,projections)
            factor_closure=float((sum(parts.values())-full).norm()/full.norm().clamp_min(1e-30))
            attn=backend.model.transformer.h[9].attn;output=attn.c_proj.weight.double()
            reconstructed=full.reshape(8,32,1152)@output.T
            actual=mixed(torch.cat(writes).view(8,32,1152).double())
            factor_bridge=float((reconstructed-actual).norm()/actual.norm())
            value_causality=float(mixed(value)[:,:,:,:7].norm()/value.norm())
            carry=1.
            for b in backend.model.transformer.h[10:]:carry*=float(b.lambdas[0])
            branch_writes={name:carry*(x.reshape(8,32,1152)@output.T) for name,x in parts.items()}
            final=torch.cat(finals).view(8,32,1152).double();arms={'native':final,'full':final-carry*actual}
            arms.update({name:final-x for name,x in branch_writes.items()})
            decoded={};margins={};aid,fid=source['answer_id'],source['foil_id']
            for arm,x in arms.items():
                chunks=[]
                for part in x.float().reshape(256,-1).split(16):
                    z=30*torch.tanh(backend.model.lm_head(F.rms_norm(part,(1152,)))/30)
                    chunks.append(z.double());decoder_batches+=1
                z=torch.cat(chunks).view(8,32,-1)
                margins[arm]=z[:,:,aid]-z[:,:,fid];decoded[arm]=z-z.mean(-1,keepdim=True)
            def bridge(x,y):return {'max_abs':float((x-y).abs().max()),'relative':float((x-y).norm()/y.norm().clamp_min(1e-30))}
            original=torch.tensor(native,device='cuda',dtype=torch.float64).view(8,32)
            bridges={'parent':bridge(original,torch.tensor(parent['margins'],device='cuda',dtype=torch.float64)),
                     'decoder':bridge(margins['native'],original)}
            natural=mixed(margins['native']);target=mixed(margins['native']-margins['full'])
            fulltarget=mixed(decoded['native']-decoded['full']);reports=[]
            for w in range(8):
                projection=float((target[w]*natural[w]).sum()/natural[w].square().sum())
                branches={}
                for name in parts:
                    effect=mixed(margins['native']-margins[name])[w]
                    vector=mixed(decoded['native']-decoded[name])[w]
                    me=float((effect-target[w]).norm()/target[w].norm())
                    ve=float((vector-fulltarget[w]).norm()/fulltarget[w].norm())
                    branches[name]={'margin_relative_error':me,'full_vector_relative_error':ve,
                        'local_margin_signed_projection':float((effect*target[w]).sum()/target[w].square().sum()),
                        'passed':me<=.10 and ve<=.10}
                reports.append({'world_id':source['worlds'][w]['world_id'],'natural_margin_projection':projection,
                                'materiality_passed':projection>=.10,'branches':branches})
            readers=backend.model.lm_head.weight[[aid,fid]].double()
            folded=(carry*(readers@output)).view(2,9,128)
            edge_readers={name:torch.einsum('wihsd,rhd->wihsr',e,folded).cpu().tolist() for name,e in edges.items()}
            gram_parts=torch.stack(list(branch_writes.values()),dim=1).flatten(2)
            gram=torch.einsum('wid,wjd->wij',gram_parts,gram_parts)
            finite=all(bool(x.isfinite().all()) for x in list(decoded.values())+[gram])
            final_rms=(final.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt()
    finally:handle.remove()
    restored=not any(m._forward_hooks or m._forward_pre_hooks for m in backend.model.modules()) and all('squared_attention' not in b.attn.__dict__ for b in backend.model.transformer.h)
    a=restored and finite and counts==[16,256] and decoder_batches==80 and factor_closure<=1e-10 and factor_bridge<=1e-4 and value_causality<=1e-6 and all(
        b['max_abs']<=1e-3 and b['relative']<=1e-5 for b in bridges.values())
    preds={'pred_a_instrument':a,'pred_b_materiality':all(r['materiality_passed'] for r in reports),
           'pred_c_routing':all(r['branches']['routing']['passed'] for r in reports),
           'pred_d_value':all(r['branches']['value']['passed'] for r in reports),
           'pred_e_cross':all(r['branches']['cross']['passed'] for r in reports)}
    result={'terminal':'route_value_partition_complete' if a else 'invalid','predictions':preds,'reports':reports,
        'bridges':bridges,'factor_closure_relative':factor_closure,'native_mixed_write_relative':factor_bridge,
        'early_source_value_mixed_relative':value_causality,'branch_order':list(parts),'branch_gram':gram.cpu().tolist(),
        'branch_edge_reader_numerators':edge_readers,'native_final_rms':final_rms.cpu().tolist(),
        'native_margin':margins['native'].cpu().tolist(),'corners':source['corners'],
        'controls':controls,'runner_sha256':sha(RUNNER),'binding_sha256':sha(BINDING),'wall_seconds':time.perf_counter()-tic,
        'price':{'forwards':counts[0],'sequences':counts[1],'decoder_batches':decoder_batches,'decoder_states':1280,
            'fits':0,'native_parameters':sum(p.numel() for p in backend.model.parameters()),'actual_weight_saving':0},
        'scope':'Exact symmetric native layer9 routing/value algebra and partial direct-write decoder effects. No independent factor producers, standalone/OOD circuit or unrelated-control selectivity.'}
    atomic_create_json(OUT,result)
    print(json.dumps({'predictions':preds,'reports':reports,'factor_closure':factor_closure,'native_write_bridge':factor_bridge,'wall_seconds':result['wall_seconds']}));assert a


if __name__=='__main__':main()
