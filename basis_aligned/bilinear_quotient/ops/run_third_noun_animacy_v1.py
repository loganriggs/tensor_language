#!/usr/bin/env python3
# BQGATE: fixed third-noun factorial, four competing rules, no fits, sixteen forwards.
"""A instrument; B controller; C object; D nearest noun; E nearest human.
Every rule requires >=.75 sign accuracy and >=1 mean signed margin in each
of32 cells. No outcome filtering or rule/template/threshold rescue.
"""
import hashlib,json,os,signal,sys,time
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path.insert(0,str(POLY));sys.path.insert(0,str(ROOT))
import third_noun_animacy_v1 as S
import source_margin_gradient as G
import circuit_fast_screen_producer as P
from circuit_fast_screen_managed_runner import atomic_create_json
ROWS=POLY/'THIRD_NOUN_ANIMACY_V1_ROWS.json';BINDING=POLY/'THIRD_NOUN_ANIMACY_V1_BINDING.json'
OUT=POLY/'THIRD_NOUN_ANIMACY_V1_RESULT.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    binding=json.loads(BINDING.read_text());assert all(sha(p)==v for p,v in binding.items())
    source=json.loads(ROWS.read_text());assert source['builder_sha256']==sha(S.__file__)
    worlds=source['worlds'];rows=[r for w in worlds for r in w['rows']]
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'gpu_accessed':False,'model_loaded':False,
                          'forwards':16,'sequences':256,'fits':0,'audit':source['audit']}));return
    assert not OUT.exists();signal.alarm(600)
    backend=P.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    tic=time.perf_counter();outputs=[];audits=[];counts=[0,0]
    def count(_m,args,_out):counts[0]+=1;counts[1]+=len(args[0])
    handle=backend.model.transformer.h[0].attn.register_forward_hook(count)
    try:
        with torch.inference_mode():
            for start in range(0,len(rows),16):
                part=rows[start:start+16]
                batch=P.ModelBatch(tuple(r['row_id'] for r in part),'base',tuple(tuple(r['ids']) for r in part),
                    (source['answer_id'],)*16,(source['foil_id'],)*16,tuple(r['semantic_position'] for r in part))
                with G.capture(backend.model) as log:out=backend.native(batch,capture=False)
                z=G.endpoint_logits(log['raw_logits'],batch).double()
                m=z[:,source['answer_id']]-z[:,source['foil_id']]
                native=torch.tensor([a-b for a,b in out.answer_foil],device='cuda',dtype=torch.float64)
                delta=m-native;absolute=float(delta.abs().max());relative=float(delta.norm()/native.norm().clamp_min(1e-30))
                audits.append({'batch':start//16,'max_abs':absolute,'relative':relative,'passed':absolute<=1e-3 and relative<=1e-5})
                outputs.append(z-z.mean(-1,keepdim=True))
            z=torch.cat(outputs).view(8,32,-1)
            h=torch.tensor([S.features(c) for c in S.CORNERS],device='cuda',dtype=torch.float64)
            assert torch.equal(h.T@h,32*torch.eye(32,device='cuda',dtype=torch.float64))
            coef=torch.einsum('ct,wcv->wtv',h,z)/32
            reconstructed=torch.einsum('ct,wtv->wcv',h,coef)
            closure=float((reconstructed-z).abs().max())
            margins=z[:,:,source['answer_id']]-z[:,:,source['foil_id']]
            reports={}
            for rule in ('controller','object','nearest_noun','nearest_human'):
                cells=[]
                for j,corner in enumerate(S.CORNERS):
                    signed=margins[:,j]*S.labels(*corner)[rule]
                    acc=float((signed>0).double().mean());mean=float(signed.mean())
                    cells.append({'corner':corner,'accuracy':acc,'signed_mean_margin':mean,'passed':acc>=.75 and mean>=1.})
                reports[rule]={'cells':cells,'passed':all(c['passed'] for c in cells),
                    'by_animacy':{str(human):all(c['passed'] for c in cells if c['corner'][4]==human) for human in (-1,1)},
                    'raw_correct':sum(round(c['accuracy']*8) for c in cells),'total':256}
    finally:handle.remove()
    restored=not any(m._forward_hooks or m._forward_pre_hooks for m in backend.model.modules())
    a=restored and counts==[16,256] and closure<=1e-9 and bool(z.isfinite().all()) and all(x['passed'] for x in audits)
    preds={'pred_a_instrument':a,'pred_b_controller':reports['controller']['passed'],
           'pred_c_object':reports['object']['passed'],'pred_d_nearest_noun':reports['nearest_noun']['passed'],
           'pred_e_nearest_human':reports['nearest_human']['passed']}
    result={'terminal':'third_noun_screen_complete' if a else 'invalid','predictions':preds,'reports':reports,
        'margins':margins.cpu().tolist(),'terms':source['terms'],'world_ids':[w['world_id'] for w in worlds],
        'margin_coefficients':(coef[:,:,source['answer_id']]-coef[:,:,source['foil_id']]).cpu().tolist(),
        'full_coefficient_norms':coef.norm(dim=-1).cpu().tolist(),'audits':audits,
        'coefficient_reconstruction_max_abs':closure,'runner_sha256':sha(RUNNER),'binding_sha256':sha(BINDING),
        'wall_seconds':time.perf_counter()-tic,'price':{'forwards':counts[0],'sequences':counts[1],'fits':0,
        'native_parameters':sum(p.numel() for p in backend.model.parameters()),'actual_weight_saving':0},
        'scope':'Structural generalization of frozen behavioral hypotheses; no neural circuit or independent full-output executor.'}
    atomic_create_json(OUT,result)
    print(json.dumps({'predictions':preds,'counts':counts,'wall_seconds':result['wall_seconds'],
                      'rules':{k:{x:v[x] for x in ('passed','by_animacy','raw_correct','total')} for k,v in reports.items()}}));assert a


if __name__=='__main__':main()
