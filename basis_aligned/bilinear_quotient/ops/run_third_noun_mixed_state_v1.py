#!/usr/bin/env python3
# BQGATE: frozen cube, native mixed-state removal, sixteen forwards, no fits.
"""A instrument; B internal mixed-state dependence; C first-attention zero."""
import hashlib,json,os,signal,sys,time
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path.insert(0,str(POLY));sys.path.insert(0,str(ROOT))
import circuit_fast_screen_producer as P
import mixed_state_projector_v1 as M
from circuit_fast_screen_managed_runner import atomic_create_json
ROWS=POLY/'THIRD_NOUN_ANIMACY_V1_ROWS.json'
BINDING=POLY/'THIRD_NOUN_MIXED_STATE_V1_BINDING.json'
OUT=POLY/'THIRD_NOUN_MIXED_STATE_V1_RESULT.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    binding=json.loads(BINDING.read_text());assert all(sha(p)==v for p,v in binding.items())
    source=json.loads(ROWS.read_text());controls=M.controls()
    parent=json.loads((POLY/'THIRD_NOUN_ANIMACY_V1_RESULT.json').read_text())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'gpu_accessed':False,'model_loaded':False,
                          'forwards':16,'sequences':256,'decoder_batches':32,'controls':controls}));return
    assert not OUT.exists();signal.alarm(600)
    backend=P.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    import torch.nn.functional as F
    tic=time.perf_counter();counts=[0,0];decoder_batches=0
    sites=[f'resid:{i:02d}' for i in range(19)]+[f'{kind}:{i:02d}' for i in range(18) for kind in ('attn','mlp')]
    cache={site:[] for site in sites};native=[]
    rows=[r for w in source['worlds'] for r in w['rows']]
    def count(_m,args,_out):counts[0]+=1;counts[1]+=len(args[0])
    handle=backend.model.transformer.h[0].attn.register_forward_hook(count)
    try:
        with torch.inference_mode():
            for start in range(0,256,16):
                part=rows[start:start+16]
                batch=P.ModelBatch(tuple(r['row_id'] for r in part),'base',tuple(tuple(r['ids']) for r in part),
                    (source['answer_id'],)*16,(source['foil_id'],)*16,tuple(r['semantic_position'] for r in part))
                out=backend.native(batch,capture=True)
                native.extend(a-b for a,b in out.answer_foil)
                for site in sites:cache[site].append(torch.stack([out.captured[r['row_id'],site] for r in part]))
            states={k:torch.cat(v).view(8,32,-1).double() for k,v in cache.items()}
            q=torch.tensor(M.projector(source['corners'],2,4),device='cuda',dtype=torch.float64)
            def mixed(x):return torch.einsum('ij,wjd->wid',q,x)
            final=states['resid:18'];piece=mixed(final)
            removed=(final-piece).float()
            residual_ratio=float(mixed(removed.double()).norm()/piece.norm().clamp_min(1e-30))
            decoded={}
            for arm,x in [('native',final.float()),('removed',removed)]:
                chunks=[]
                for part in x.reshape(256,-1).split(16):
                    raw=backend.model.lm_head(F.rms_norm(part,(part.shape[-1],)))
                    chunks.append((30*torch.tanh(raw/30)).double());decoder_batches+=1
                decoded[arm]=torch.cat(chunks).view(8,32,-1)
            aid,fid=source['answer_id'],source['foil_id']
            margins={k:v[:,:,aid]-v[:,:,fid] for k,v in decoded.items()}
            original=torch.tensor(native,device='cuda',dtype=torch.float64).view(8,32)
            prior=torch.tensor(parent['margins'],device='cuda',dtype=torch.float64)
            def bridge(x,y):
                return {'max_abs':float((x-y).abs().max()),'relative':float((x-y).norm()/y.norm().clamp_min(1e-30))}
            bridges={'parent':bridge(original,prior),'decoder':bridge(margins['native'],original)}
            reports=[]
            m0=mixed(margins['native'].unsqueeze(-1));m1=mixed(margins['removed'].unsqueeze(-1))
            z0=mixed(decoded['native']);z1=mixed(decoded['removed'])
            for w in range(8):
                nr=float(m0[w].square().mean().sqrt());rr=float(m1[w].norm()/m0[w].norm().clamp_min(1e-30))
                reports.append({'world_id':source['worlds'][w]['world_id'],'native_mixed_margin_rms':nr,
                    'remaining_mixed_margin_ratio':rr,'removed_signed_projection':float(((m0[w]-m1[w])*m0[w]).sum()/m0[w].square().sum()),
                    'full_mixed_remaining_ratio':float(z1[w].norm()/z0[w].norm().clamp_min(1e-30)),
                    'passed':nr>=.05 and rr<=.25})
            # Transport each actual native write through later residual lambda0s.
            coefficients={};carry=1.
            for layer in reversed(range(18)):
                for kind in ('attn','mlp'):coefficients[f'{kind}:{layer:02d}']=carry
                carry*=float(backend.model.transformer.h[layer].lambdas[0])
            embedding=1.
            for block in backend.model.transformer.h:
                embedding=float(block.lambdas[0])*embedding+float(block.lambdas[1])
            total=embedding*states['resid:00']+sum(coefficients[k]*states[k] for k in coefficients)
            telescope=bridge(total,final)
            keys=sorted(coefficients)
            parts=torch.stack([coefficients[k]*mixed(states[k]) for k in keys],dim=1).flatten(2)
            gram=torch.einsum('wid,wjd->wij',parts,parts)
            mixed_telescope=bridge(parts.sum(1),piece.flatten(1))
            site_stats={k:{'mixed_norm_by_world':mixed(v).flatten(1).norm(dim=1).cpu().tolist(),
                           'whole_norm_by_world':v.flatten(1).norm(dim=1).cpu().tolist()} for k,v in states.items()}
            first_zero=float(mixed(states['attn:00']).norm())<=1e-6*max(1.,float(states['attn:00'].norm()))
            embed_zero=bool((mixed(states['resid:00'])==0).all())
            finite=all(bool(v.isfinite().all()) for v in list(states.values())+list(decoded.values()))
    finally:handle.remove()
    restored=not any(m._forward_hooks or m._forward_pre_hooks for m in backend.model.modules())
    a=restored and counts==[16,256] and decoder_batches==32 and finite and residual_ratio<=1e-4 and telescope['relative']<=1e-5 and all(
        b['max_abs']<=1e-3 and b['relative']<=1e-5 for b in bridges.values())
    preds={'pred_a_instrument':a,'pred_b_internal_mixed_state':all(r['passed'] for r in reports),
           'pred_c_first_attention_zero':first_zero and embed_zero}
    result={'terminal':'mixed_state_screen_complete' if a else 'invalid','predictions':preds,
        'reports':reports,'bridges':bridges,'raw_telescope':telescope,'mixed_telescope':mixed_telescope,
        'removed_raw_mixed_ratio':residual_ratio,'site_statistics':site_stats,
        'source_gram':gram.cpu().tolist(),'source_order':keys,'transport_coefficients':coefficients,
        'final_mixed_norm_squared_by_world':piece.flatten(1).square().sum(1).cpu().tolist(),
        'controls':controls,'runner_sha256':sha(RUNNER),'binding_sha256':sha(BINDING),
        'wall_seconds':time.perf_counter()-tic,'price':{'forwards':counts[0],'sequences':counts[1],
        'decoder_batches':decoder_batches,'decoder_states':512,'fits':0,'native_parameters':sum(p.numel() for p in backend.model.parameters()),'actual_weight_saving':0},
        'scope':'Prospective finite-table final-state edit on opened texts. Native background and decoder retained; no independently generated circuit or selective removal claim.'}
    atomic_create_json(OUT,result)
    print(json.dumps({'predictions':preds,'reports':reports,'bridges':bridges,'raw_telescope':telescope,
                      'mixed_telescope':mixed_telescope,'wall_seconds':result['wall_seconds']}));assert a


if __name__=='__main__':main()
