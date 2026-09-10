#!/usr/bin/env python3
# BQGATE: symmetric native factor partition, fixed eighteen MLPs, no fitting.
"""A instrument; B full-MLP dependence; C new-product; D input-inherited."""
import hashlib,json,os,signal,sys,time
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path.insert(0,str(POLY));sys.path.insert(0,str(ROOT))
import circuit_fast_screen_producer as P
import symmetric_factor_interaction_v1 as S
from circuit_fast_screen_managed_runner import atomic_create_json
BINDING=POLY/'THIRD_NOUN_MLP_FACTOR_PARTITION_V1_BINDING.json'
OUT=POLY/'THIRD_NOUN_MLP_FACTOR_PARTITION_V1_RESULT.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    binding=json.loads(BINDING.read_text());assert all(sha(p)==v for p,v in binding.items())
    source=json.loads((POLY/'THIRD_NOUN_ANIMACY_V1_ROWS.json').read_text())
    parent=json.loads((POLY/'THIRD_NOUN_ANIMACY_V1_RESULT.json').read_text());controls=S.controls()
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'gpu_accessed':False,'model_loaded':False,'forwards':16,
                          'sequences':256,'decoder_batches':64,'controls':controls}));return
    assert not OUT.exists();signal.alarm(600)
    backend=P.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    import torch.nn.functional as F
    assert all(not b.mlp.config.gated for b in backend.model.transformer.h)
    tic=time.perf_counter();counts=[0,0];decoder_batches=0;handles=[];record={}
    sites=['resid:18']+[f'mlp:{i:02d}' for i in range(18)]
    cache={k:[] for k in sites};factors={(i,k):[] for i in range(18) for k in ('Left','Right')};native=[]
    rows=[r for w in source['worlds'] for r in w['rows']]
    assert all(r['semantic_position']==9 and len(r['ids'])==10 for r in rows)
    def count(_m,args,_out):counts[0]+=1;counts[1]+=len(args[0])
    def save(key):
        def hook(_m,_a,y):record[key]=y[:,9,:].detach().clone()
        return hook
    handles.append(backend.model.transformer.h[0].attn.register_forward_hook(count))
    for i,b in enumerate(backend.model.transformer.h):
        for k in ('Left','Right'):handles.append(getattr(b.mlp,k).register_forward_hook(save((i,k))))
    try:
        with torch.inference_mode():
            for start in range(0,256,16):
                part=rows[start:start+16];record.clear()
                batch=P.ModelBatch(tuple(r['row_id'] for r in part),'base',tuple(tuple(r['ids']) for r in part),
                    (source['answer_id'],)*16,(source['foil_id'],)*16,(9,)*16)
                out=backend.native(batch,capture=True);native.extend(a-b for a,b in out.answer_foil)
                assert set(record)==set(factors)
                for k in factors:factors[k].append(record[k])
                for k in sites:cache[k].append(torch.stack([out.captured[r['row_id'],k] for r in part]))
            states={k:torch.cat(v).view(8,32,-1).double() for k,v in cache.items()}
            q=S.projectors(source['corners']).to('cuda')
            def mixed(x):return torch.einsum('ij,wjd->wid',q[3],x)
            carries={};carry=1.
            for i in reversed(range(18)):
                carries[i]=carry;carry*=float(backend.model.transformer.h[i].lambdas[0])
            new=[];inherited=[];audits=[]
            for i,b in enumerate(backend.model.transformer.h):
                l=torch.cat(factors[i,'Left']).view(8,32,-1).double()
                r=torch.cat(factors[i,'Right']).view(8,32,-1).double()
                n,h,full=S.partition(l,r,q)
                algebra=float((n+h-full).norm()/full.norm().clamp_min(1e-30))
                down=b.mlp.Down.weight.double();nw=n@down.T;hw=h@down.T
                actual=mixed(states[f'mlp:{i:02d}'])
                replay=float((nw+hw-actual).norm()/actual.norm().clamp_min(1e-30))
                audits.append({'layer':i,'factor_algebra_relative':algebra,'native_write_relative':replay,
                               'passed':algebra<=1e-10 and replay<=1e-4})
                new.append(carries[i]*nw);inherited.append(carries[i]*hw)
            final=states['resid:18'];n=sum(new);h=sum(inherited)
            native_full=sum(carries[i]*mixed(states[f'mlp:{i:02d}']) for i in range(18))
            arms={'native':final,'full':final-native_full,'new':final-n,'inherited':final-h}
            margins={};full_mixed_norms={}
            aid,fid=source['answer_id'],source['foil_id']
            for arm,x in arms.items():
                chunks=[]
                for part in x.float().reshape(256,-1).split(16):
                    z=30*torch.tanh(backend.model.lm_head(F.rms_norm(part,(part.shape[-1],)))/30)
                    chunks.append(z.double());decoder_batches+=1
                z=torch.cat(chunks).view(8,32,-1)
                margins[arm]=z[:,:,aid]-z[:,:,fid]
                full_mixed_norms[arm]=mixed(z).flatten(1).norm(dim=1).cpu().tolist()
            original=torch.tensor(native,device='cuda',dtype=torch.float64).view(8,32)
            prior=torch.tensor(parent['margins'],device='cuda',dtype=torch.float64)
            def bridge(x,y):return {'max_abs':float((x-y).abs().max()),'relative':float((x-y).norm()/y.norm().clamp_min(1e-30))}
            bridges={'parent':bridge(original,prior),'decoder':bridge(margins['native'],original)}
            reports={};target=mixed(margins['native'].unsqueeze(-1))
            for arm in ('full','new','inherited'):
                remaining=mixed(margins[arm].unsqueeze(-1));cells=[]
                for w in range(8):
                    nr=float(target[w].square().mean().sqrt());ratio=float(remaining[w].norm()/target[w].norm())
                    cells.append({'world_id':source['worlds'][w]['world_id'],'native_mixed_margin_rms':nr,
                        'remaining_mixed_margin_ratio':ratio,'removed_signed_projection':float(((target[w]-remaining[w])*target[w]).sum()/target[w].square().sum()),
                        'passed':nr>=.05 and ratio<=.25})
                reports[arm]={'passed':all(c['passed'] for c in cells),'cells':cells}
            names=[f'{kind}:{i:02d}' for i in range(18) for kind in ('new','inherited')]
            parts=torch.stack([v for pair in zip(new,inherited) for v in pair],dim=1).flatten(2)
            gram=torch.einsum('wid,wjd->wij',parts,parts)
            target_state=mixed(final).flatten(1)
            source_dot_final=torch.einsum('wid,wd->wi',parts,target_state)
            finite=all(bool(x.isfinite().all()) for x in [gram,source_dot_final]+list(margins.values()))
    finally:
        for handle in handles:handle.remove()
    restored=not any(m._forward_hooks or m._forward_pre_hooks for m in backend.model.modules())
    a=restored and finite and counts==[16,256] and decoder_batches==64 and all(x['passed'] for x in audits) and all(
        b['max_abs']<=1e-3 and b['relative']<=1e-5 for b in bridges.values())
    preds={'pred_a_instrument':a,'pred_b_full_mlp_dependence':reports['full']['passed'],
           'pred_c_new_product_dependence':reports['new']['passed'],'pred_d_inherited_dependence':reports['inherited']['passed']}
    result={'terminal':'factor_partition_complete' if a else 'invalid','predictions':preds,'reports':reports,
        'bridges':bridges,'layer_audits':audits,'full_mixed_output_norms':full_mixed_norms,
        'branch_gram':gram.cpu().tolist(),'branch_names':names,'branch_dot_final':source_dot_final.cpu().tolist(),
        'final_mixed_squared_norm':target_state.square().sum(1).cpu().tolist(),
        'controls':controls,'runner_sha256':sha(RUNNER),'binding_sha256':sha(BINDING),'wall_seconds':time.perf_counter()-tic,
        'price':{'forwards':counts[0],'sequences':counts[1],'decoder_batches':decoder_batches,'decoder_states':1024,
            'fits':0,'native_parameters':sum(p.numel() for p in backend.model.parameters()),'actual_weight_saving':0},
        'scope':'Symmetric actual-factor partition, transported native direct-write edits on opened texts. Inherited includes normalization. All producers/decoder retained; no standalone extracted semantic program.'}
    atomic_create_json(OUT,result)
    print(json.dumps({'predictions':preds,'reports':reports,'bridges':bridges,'wall_seconds':result['wall_seconds']}));assert a


if __name__=='__main__':main()
