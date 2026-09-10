#!/usr/bin/env python3
# BQGATE: frozen factorial population, six predictions and eight native forwards; no fitting.
"""Native controller selector screen; registered task and first/nearest alternatives.

A instrumentation; B controller capability; C inactive-number selectivity;
D complementary full-output interactions; E nearest-only; F first-only.
Capability per split/corner >=.75 correct and mean signed margin>=1.
No outcome filtering or phrase/threshold rescue; all weights retained.
"""
import hashlib,json,os,signal,sys,time
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path.insert(0,str(POLY));sys.path.insert(0,str(ROOT))
import controller_selector_v1 as S
import source_margin_gradient as G
import circuit_fast_screen_producer as P
from circuit_fast_screen_managed_runner import atomic_create_json
ROWS=POLY/'SUBJECT_OBJECT_CONTROLLER_V1_ROWS.json'
BINDING=POLY/'SUBJECT_OBJECT_CONTROLLER_V1_BINDING.json'
OUT=POLY/'SUBJECT_OBJECT_CONTROLLER_V1_RESULT.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    binding=json.loads(BINDING.read_text());assert all(sha(p)==v for p,v in binding.items())
    source=json.loads(ROWS.read_text());assert source['builder_sha256']==sha(S.__file__)
    assert S.controls()['passed'];worlds=source['worlds']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'gpu_accessed':False,'model_loaded':False,
                          'forwards':8,'sequences':128,'backwards':0,'audit':source['audit']}));return
    assert not OUT.exists();signal.alarm(600)
    backend=P.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    tic=time.perf_counter();outputs=[];audits=[];counts=[0,0]
    def count(_m,args,_out):counts[0]+=1;counts[1]+=len(args[0])
    handle=backend.model.transformer.h[0].attn.register_forward_hook(count)
    try:
        with torch.inference_mode():
            for corner in range(8):
                rows=[w['rows'][corner] for w in worlds]
                batch=P.ModelBatch(tuple(r['row_id'] for r in rows),'base',tuple(tuple(r['ids']) for r in rows),
                    (source['positive_answer_id'],)*16,(source['negative_answer_id'],)*16,
                    tuple(r['semantic_position'] for r in rows))
                with G.capture(backend.model) as log:out=backend.native(batch,capture=False)
                z=G.endpoint_logits(log['raw_logits'],batch).double()
                m=z[:,source['positive_answer_id']]-z[:,source['negative_answer_id']]
                native=torch.tensor([a-b for a,b in out.answer_foil],device='cuda',dtype=torch.float64)
                delta=m-native;absolute=float(delta.abs().max());relative=float(delta.norm()/native.norm().clamp_min(1e-30))
                audits.append({'corner':corner,'max_abs':absolute,'relative':relative,'passed':absolute<=1e-3 and relative<=1e-5})
                outputs.append(z-z.mean(-1,keepdim=True))
            z=torch.stack(outputs,dim=1)  # world,corner,vocabulary
            h=torch.tensor(S.design(),device='cuda',dtype=torch.float64)
            coefficients=torch.einsum('ct,wcv->wtv',h,z)/8
            reconstruction=torch.einsum('ct,wtv->wcv',h,coefficients)
            closure=float((reconstruction-z).abs().max())
            margins=z[:,:,source['positive_answer_id']]-z[:,:,source['negative_answer_id']]
            reports={}
            for split in ('fit','held'):
                ix=[i for i,w in enumerate(worlds) if w['split']==split];m=margins[ix];coef=coefficients[ix]
                labels={'controller':[float(S.selector(*v)) for v in S.CORNERS],
                        'nearest':[v[2] for v in S.CORNERS],'first':[v[1] for v in S.CORNERS]}
                capabilities={}
                for rule,signs in labels.items():
                    cells=[]
                    for j,sign in enumerate(signs):
                        signed=m[:,j]*sign;accuracy=float((signed>0).double().mean());mean=float(signed.mean())
                        cells.append({'corner':S.CORNERS[j],'accuracy':accuracy,'signed_mean_margin':mean,
                                      'passed':accuracy>=.75 and mean>=1.})
                    capabilities[rule]={'cells':cells,'passed':all(c['passed'] for c in cells)}
                lookup={v:j for j,v in enumerate(S.CORNERS)};selectivity={}
                for c in (-1,1):
                    active=[];inactive=[]
                    for other in (-1,1):
                        ap=(c,1,other) if c==1 else (c,other,1)
                        am=(c,-1,other) if c==1 else (c,other,-1)
                        ip=(c,other,1) if c==1 else (c,1,other)
                        im=(c,other,-1) if c==1 else (c,-1,other)
                        active.append((m[:,lookup[ap]]-m[:,lookup[am]])/2)
                        inactive.append((m[:,lookup[ip]]-m[:,lookup[im]])/2)
                    active=torch.stack(active);inactive=torch.stack(inactive)
                    means=active.mean(-1);ar=float(active.square().mean().sqrt());ir=float(inactive.square().mean().sqrt())
                    selectivity[str(c)]={'active_means':means.cpu().tolist(),'active_rms':ar,'inactive_rms':ir,
                                        'ratio':ir/max(ar,1e-8),'passed':bool((means>=1).all()) and ir<=.25*ar}
                cs,co=coef[:,S.TERMS.index('cs')],coef[:,S.TERMS.index('co')]
                a,b=float(cs.norm()),float(co.norm());cos=-float((cs*co).sum())/max(a*b,1e-30);ratio=a/max(b,1e-30)
                interaction={'cs_norm':a,'co_norm':b,'opposite_cosine':cos,'norm_ratio':ratio,
                             'passed':a>1e-8 and b>1e-8 and cos>=.90 and .8<=ratio<=1.25}
                reports[split]={'capability':capabilities,'selectivity':selectivity,'interaction':interaction,
                    'margin_coefficients':(coef[:,:,source['positive_answer_id']]-coef[:,:,source['negative_answer_id']]).cpu().tolist(),
                    'full_coefficient_norms':coef.norm(dim=-1).cpu().tolist(),'world_ids':[worlds[i]['world_id'] for i in ix]}
    finally:handle.remove()
    restored=not any(m._forward_hooks or m._forward_pre_hooks for m in backend.model.modules())
    a=restored and counts==[8,128] and closure<=1e-9 and bool(z.isfinite().all()) and all(x['passed'] for x in audits)
    preds={'pred_a_instrument':a,
           'pred_b_controller_capability':all(r['capability']['controller']['passed'] for r in reports.values()),
           'pred_c_number_selectivity':all(v['passed'] for r in reports.values() for v in r['selectivity'].values()),
           'pred_d_shared_output_interaction':all(r['interaction']['passed'] for r in reports.values()),
           'pred_e_nearest_noun':all(r['capability']['nearest']['passed'] for r in reports.values()),
           'pred_f_first_noun':all(r['capability']['first']['passed'] for r in reports.values())}
    result={'terminal':'controller_capability_complete' if a else 'invalid','predictions':preds,'reports':reports,
            'margins':margins.cpu().tolist(),'audits':audits,'coefficient_reconstruction_max_abs':closure,
            'runner_sha256':sha(RUNNER),'binding_sha256':sha(BINDING),'wall_seconds':time.perf_counter()-tic,
            'price':{'forwards':counts[0],'sequences':counts[1],'backwards':0,'fits':0,
                     'native_parameters':sum(p.numel() for p in backend.model.parameters()),'actual_weight_saving':0},
            'scope':'Native capability on frozen factorial text; exact finite-table expansion only; no identified neural circuit.'}
    atomic_create_json(OUT,result);print(json.dumps({'predictions':preds,'wall_seconds':result['wall_seconds'],'reports':reports}));assert a


if __name__=='__main__':main()
