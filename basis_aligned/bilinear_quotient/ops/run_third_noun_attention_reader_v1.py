#!/usr/bin/env python3
# BQGATE: fixed attention/MLP readout factorial, exact shared-norm statistics, no fits.
"""A instrument; B native attention dependence; C joint decoder composition."""
import hashlib,json,os,signal,sys,time
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path.insert(0,str(POLY));sys.path.insert(0,str(ROOT))
import numpy as np
import circuit_fast_screen_producer as P
import mixed_state_projector_v1 as Q
import rms_softcap_edit_statistics_v1 as S
from circuit_fast_screen_managed_runner import atomic_create_json
BINDING=POLY/'THIRD_NOUN_ATTENTION_READER_V1_BINDING.json'
OUT=POLY/'THIRD_NOUN_ATTENTION_READER_V1_RESULT.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    binding=json.loads(BINDING.read_text());assert all(sha(p)==v for p,v in binding.items())
    source=json.loads((POLY/'THIRD_NOUN_ANIMACY_V1_ROWS.json').read_text());controls=S.controls()
    parent=json.loads((POLY/'THIRD_NOUN_ANIMACY_V1_RESULT.json').read_text())
    pm=json.loads((POLY/'THIRD_NOUN_MLP_FACTOR_PARTITION_V1_RESULT.json').read_text())
    pam=json.loads((POLY/'THIRD_NOUN_MIXED_STATE_V1_RESULT.json').read_text())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'gpu_accessed':False,'model_loaded':False,'forwards':16,
                          'sequences':256,'decoder_batches':64,'statistic_sources':19,'controls':controls}));return
    assert not OUT.exists();signal.alarm(600)
    backend=P.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    import torch.nn.functional as F
    tic=time.perf_counter();counts=[0,0];decoder_batches=0
    sites=['resid:18']+[f'{kind}:{i:02d}' for i in range(18) for kind in ('attn','mlp')]
    cache={k:[] for k in sites};native=[];rows=[r for w in source['worlds'] for r in w['rows']]
    def count(_m,args,_y):counts[0]+=1;counts[1]+=len(args[0])
    handle=backend.model.transformer.h[0].attn.register_forward_hook(count)
    try:
        with torch.inference_mode():
            for start in range(0,256,16):
                part=rows[start:start+16]
                batch=P.ModelBatch(tuple(r['row_id'] for r in part),'base',tuple(tuple(r['ids']) for r in part),
                    (source['answer_id'],)*16,(source['foil_id'],)*16,tuple(r['semantic_position'] for r in part))
                out=backend.native(batch,capture=True);native.extend(a-b for a,b in out.answer_foil)
                for k in sites:cache[k].append(torch.stack([out.captured[r['row_id'],k] for r in part]))
            states={k:torch.cat(v).view(8,32,-1).double() for k,v in cache.items()}
            q=torch.tensor(Q.projector(source['corners'],2,4),device='cuda',dtype=torch.float64)
            def mixed(x):return torch.einsum('ij,wjd->wid',q,x)
            carries={};carry=1.
            for i in reversed(range(18)):
                carries[i]=carry;carry*=float(backend.model.transformer.h[i].lambdas[0])
            attention=[carries[i]*mixed(states[f'attn:{i:02d}']) for i in range(18)]
            mlp=sum(carries[i]*mixed(states[f'mlp:{i:02d}']) for i in range(18))
            final=states['resid:18'];sources=torch.stack(attention+[mlp],dim=2)
            source_names=[f'attn:{i:02d}' for i in range(18)]+['all_mlp']
            doses={'native':np.zeros(19),'A':np.array([-1.]*18+[0.]),
                   'M':np.array([0.]*18+[-1.]),'AM':-np.ones(19)}
            aid,fid=source['answer_id'],source['foil_id'];decoded={};pairs={};margins={}
            for arm,dose in doses.items():
                x=final+torch.einsum('wcsd,s->wcd',sources,torch.tensor(dose,device='cuda',dtype=torch.float64))
                chunks=[]
                for part in x.float().reshape(256,-1).split(16):
                    chunks.append((30*torch.tanh(backend.model.lm_head(F.rms_norm(part,(part.shape[-1],)))/30)).double());decoder_batches+=1
                z=torch.cat(chunks).view(8,32,-1);decoded[arm]=mixed(z)
                pairs[arm]=z[:,:,[aid,fid]].cpu().numpy();margins[arm]=z[:,:,aid]-z[:,:,fid]
            base_cpu=final.reshape(256,-1).cpu().numpy()
            sources_cpu=sources.reshape(256,19,-1).cpu().numpy()
            readers=backend.model.lm_head.weight[[aid,fid]].double().cpu().numpy()
            statistics=[S.compile_statistics(x,d,readers,eps=float(np.finfo(np.float32).eps)) for x,d in zip(base_cpu,sources_cpu)]
            def bridge(x,y):
                x=np.asarray(x);y=np.asarray(y)
                return {'max_abs':float(np.max(abs(x-y))),'relative':float(np.linalg.norm(x-y)/max(1e-30,np.linalg.norm(y)))}
            bridges={'parent':bridge(np.array(native).reshape(8,32),parent['margins'])}
            for arm,dose in doses.items():
                prediction=np.array([S.decode(s,dose) for s in statistics]).reshape(8,32,2)
                bridges[f'statistics_{arm}']=bridge(prediction,pairs[arm])
            target=mixed(margins['native'].unsqueeze(-1));reports=[];native_margin=margins['native'].cpu().numpy()
            for w in range(8):
                t=target[w];den=t.norm();ratios={arm:float(mixed(m.unsqueeze(-1))[w].norm()/den) for arm,m in margins.items()}
                interaction=mixed((margins['AM']-margins['A']-margins['M']+margins['native']).unsqueeze(-1))[w]
                ic=float(interaction.norm()/den);nr=float(t.square().mean().sqrt())
                conditional=mixed((margins['M']-margins['AM']).unsqueeze(-1))[w]
                effect=mixed((margins['native']-margins['A']).unsqueeze(-1))[w]
                reports.append({'world_id':source['worlds'][w]['world_id'],'native_mixed_margin_rms':nr,
                    'remaining_ratios':ratios,'attention_native_signed_projection':float((effect*t).sum()/t.square().sum()),
                    'attention_conditional_signed_projection':float((conditional*t).sum()/t.square().sum()),
                    'mixed_decoder_interaction_ratio':ic,
                    'full_vector_mixed_interaction_ratio':float((decoded['AM'][w]-decoded['A'][w]-decoded['M'][w]+decoded['native'][w]).norm()/decoded['native'][w].norm()),
                    'attention_passed':nr>=.05 and ratios['A']<=.25,'composition_passed':ic<=.10})
            replays={'M_ratio_max_abs':max(abs(r['remaining_ratios']['M']-s['remaining_mixed_margin_ratio']) for r,s in zip(reports,pm['reports']['full']['cells'])),
                     'AM_ratio_max_abs':max(abs(r['remaining_ratios']['AM']-s['remaining_mixed_margin_ratio']) for r,s in zip(reports,pam['reports']))}
            finite=all(bool(z.isfinite().all()) for z in decoded.values())
    finally:handle.remove()
    restored=not any(m._forward_hooks or m._forward_pre_hooks for m in backend.model.modules())
    a=restored and finite and counts==[16,256] and decoder_batches==64 and max(replays.values())<=1e-3 and all(
        b['max_abs']<=1e-3 and b['relative']<=1e-5 for b in bridges.values())
    preds={'pred_a_instrument':a,'pred_b_attention_dependence':all(r['attention_passed'] for r in reports),
           'pred_c_decoder_composition':all(r['composition_passed'] for r in reports)}
    result={'terminal':'attention_reader_complete' if a else 'invalid','predictions':preds,'reports':reports,
        'bridges':bridges,'parent_replays':replays,'source_names':source_names,'corners':source['corners'],
        'row_ids':[r['row_id'] for r in rows],'native_margin':native_margin.tolist(),
        'reader_token_ids':[aid,fid],'statistics':[{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in s.items()} for s in statistics],
        'controls':controls,'runner_sha256':sha(RUNNER),'binding_sha256':sha(BINDING),'wall_seconds':time.perf_counter()-tic,
        'price':{'forwards':counts[0],'sequences':counts[1],'decoder_batches':decoder_batches,'decoder_states':1024,
            'fits':0,'native_parameters':sum(p.numel() for p in backend.model.parameters()),'actual_weight_saving':0},
        'scope':'Two-reader finite source-edit statistics on opened native backgrounds. No upstream recomputation, independently produced circuit, unrelated control selectivity or OOD semantic identification.'}
    atomic_create_json(OUT,result)
    print(json.dumps({'predictions':preds,'reports':reports,'bridges':bridges,'replays':replays,'wall_seconds':result['wall_seconds']}));assert a


if __name__=='__main__':main()
