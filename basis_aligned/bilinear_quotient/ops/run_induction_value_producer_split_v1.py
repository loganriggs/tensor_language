#!/usr/bin/env python3
# BQGATE: induction shared/local value screen;63forwards2016seq,0fits.
"""A instruments; B equality specificity; C shared/D local .10 fidelity; E composition .10."""
import hashlib,json,os,signal,sys,time
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path.insert(0,str(POLY));sys.path.insert(0,str(ROOT))
import numpy as np
import circuit_fast_screen_producer as P
import induction_value_producer_split_v1 as V
import source_margin_gradient as G
import attention_write_factorial_executor_v1 as S
from circuit_fast_screen_managed_runner import atomic_create_json
BINDING=POLY/'INDUCTION_VALUE_PRODUCER_SPLIT_V1_BINDING.json'
OUT=POLY/'INDUCTION_VALUE_PRODUCER_SPLIT_V1_RESULT.json'
ROWS=ROOT/'basis_aligned/bilinear_quotient/induction_selector_payload_three_source_rows_rung578.json'
CONDITIONS=('s0p0','s0p1','s1p0','s1p1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rows():
    bank=json.loads(ROWS.read_text());groups=[g for g in bank['groups'] if g['split']=='FIT']
    assert len(groups)==72
    out=[]
    for g in groups:
        for name in CONDITIONS:
            c=g['factorial_conditions'][name]
            q=c['query_position'];ids=c['ids'];sources=c['payload_positions']
            assert q==len(ids)-1 and len(ids)<=30
            assert sum(ids[t-1]==ids[q] for t in sources)==1
            neutral=c['N_payload_position'];assert ids[neutral-1]!=ids[q]
            out.append(dict(group_id=g['group_id'],condition=name,ids=ids,
                position=q,answer=c['answer_id'],foil=c['other_answer_id'],neutral_id=c['neutral_payload_id'],
                equality=[t for t in sources if ids[t-1]==ids[q]],neutral=neutral))
    return out


def main():
    binding=json.loads(BINDING.read_text());assert all(sha(p)==v for p,v in binding.items())
    bank=rows()
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(dryrun=True,model_loaded=False,gpu_accessed=False,forwards=63,sequences=2016,groups=72,rows=288,fits=0)));return
    assert not OUT.exists();signal.alarm(600)
    backend=P.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2);model=backend.model
    counts=[0,0];audits=[];firsts=[];tic=time.perf_counter();saved={}
    def counter(_m,a,_o):counts[0]+=1;counts[1]+=len(a[0])
    handle=model.transformer.h[0].attn.register_forward_hook(counter)
    try:
        with torch.inference_mode():
            for mode in ('native','zero','shared','local','full','joint','neutral'):
                chunks=[];first_chunks=[]
                for start in range(0,len(bank),32):
                    part=bank[start:start+32]
                    b=P.ModelBatch(tuple(f"{r['group_id']}:{r['condition']}" for r in part),'base',
                        tuple(tuple(r['ids']+[50256]*(30-len(r['ids']))) for r in part),
                        tuple(r['answer'] for r in part),tuple(r['foil'] for r in part),tuple(r['position'] for r in part))
                    pos=torch.tensor(b.semantic_positions);eq=torch.zeros(32,30,dtype=torch.bool);neutral=eq.clone()
                    for i,r in enumerate(part):eq[i,r['equality']]=True;neutral[i,r['neutral']]=True
                    capture={};h=model.transformer.h[8].attn.register_forward_hook(lambda _m,_a,o:capture.update(first=o[1].detach().cpu()))
                    try:
                        from contextlib import nullcontext
                        with (nullcontext() if mode=='native' else V.remove(model,pos,eq,neutral,mode,audits)):
                            with G.capture(model) as g:backend.native(b,capture=False)
                        chunks.append(G.endpoint_logits(g['raw_logits'],b).double().cpu().numpy())
                        first_chunks.append(capture['first'])
                    finally:h.remove()
                saved[mode]=np.concatenate(chunks);firsts.append(torch.cat(first_chunks))
    finally:handle.remove()
    restored=not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
    restored &= all('squared_attention' not in b.attn.__dict__ for b in model.transformer.h)
    z=saved['native'];effects={m:z-saved[m] for m in ('shared','local','full','neutral')}
    bridges={'zero':S.bridge(saved['zero'],z),'joint_full':S.bridge(saved['joint'],saved['full'])}
    idx=np.arange(len(bank));answer=np.array([r['answer'] for r in bank]);foil=np.array([r['foil'] for r in bank])
    readers=np.array([[r['answer'],r['foil'],r['neutral_id']] for r in bank])
    def read(a):return np.take_along_axis(a,readers,axis=1)
    def goldp(a):
        shifted=a-a.max(1,keepdims=True);return np.exp(shifted[idx,answer])/np.exp(shifted).sum(1)
    probabilities={m:goldp(a) for m,a in saved.items()};reports=[]
    for ci,name in enumerate(CONDITIONS):
        take=np.arange(ci,len(bank),4)
        def norms(a,b):
            ar,br=read(a)[take],read(b)[take]
            return dict(margin=S.ratio(ar[:,0]-ar[:,1],br[:,0]-br[:,1]),
                readers=S.ratio(ar-ar.mean(1,keepdims=True),br-br.mean(1,keepdims=True)),
                vocabulary=S.ratio(a[take]-a[take].mean(1,keepdims=True),b[take]-b[take].mean(1,keepdims=True)))
        loss={m:float((probabilities['native'][take]-probabilities[m][take]).mean()) for m in ('full','neutral','shared','local')}
        branch={m:norms(effects[m]-effects['full'],effects['full']) for m in ('shared','local')}
        composition=norms(effects['full']-effects['shared']-effects['local'],effects['full'])
        specificity=loss['full']>=.10 and loss['full']-loss['neutral']>=.05
        reports.append(dict(condition=name,native_accuracy=float((z[idx,answer][take]>z[idx,foil][take]).mean()),
            native_gold_probability=float(probabilities['native'][take].mean()),gold_probability_loss=loss,
            branch_errors=branch,composition_errors=composition,specificity_passed=specificity,
            shared_passed=specificity and all(S.passed(v,.10) for v in branch['shared'].values()),
            local_passed=specificity and all(S.passed(v,.10) for v in branch['local'].values()),
            composition_passed=all(S.passed(v,.10) for v in composition.values())))
    instrument=restored and counts==[63,2016] and len(audits)==162
    instrument &= all(torch.equal(firsts[0],f) for f in firsts)
    instrument &= all(np.isfinite(v).all() for v in saved.values())
    instrument &= all(a['untouched_outputs_bitwise'] and a['finite'] and a['value_mix_relative']<=1e-6 for a in audits)
    instrument &= all(b['max_abs']<=1e-3 and b['relative']<=1e-5 for b in bridges.values())
    capable=all(r['native_accuracy']>=.85 for r in reports)
    preds=dict(pred_a_instrument=bool(instrument),pred_b_equality_specificity=capable and all(r['specificity_passed'] for r in reports),
        pred_c_shared=capable and all(r['shared_passed'] for r in reports),pred_d_local=capable and all(r['local_passed'] for r in reports),
        pred_e_composition=all(r['composition_passed'] for r in reports))
    out=dict(terminal='complete' if instrument else 'invalid',predictions=preds,reports=reports,bridges=bridges,
        native_capability=capable,
        reader_rows=[{k:r[k] for k in ('group_id','condition','answer','foil','neutral_id')} for r in bank],
        reader_logits={m:read(a).tolist() for m,a in saved.items()},
        maximum_value_mix_relative=max(a['value_mix_relative'] for a in audits),
        price=dict(forwards=counts[0],sequences=counts[1],attention_patches=len(audits),fits=0,
                   native_parameters=sum(p.numel() for p in model.parameters()),weight_saving=0),
        runner_sha256=sha(RUNNER),binding_sha256=sha(BINDING),wall_seconds=time.perf_counter()-tic,
        scope='FIT-only live value-producer operation screen. Full native weights/producers retained. Not R593 donor interchange, OOD, independent extraction or induction-only collateral certification.')
    atomic_create_json(OUT,out);print(json.dumps({k:out[k] for k in ('predictions','reports','bridges','wall_seconds','native_capability')}))
    assert instrument


if __name__=='__main__':main()
