"""Bounded native composition screen; no fitted amplitudes or selected heads."""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path('/workspace/tensor_language'); BASE=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(ROOT),str(BASE)]
import torch
import exact_source_edit_reference as E
import join_contribution_context_reference as J
import overlapping_join_normalizer_reference as N
import field_intervention_metrics as M

SCALES=(0.,.5,1.)
def key(a,b): return f'{a:g}_{b:g}'
def center(x): return x-x.mean(-1,keepdim=True)
def rms(x): return float(x.square().mean().sqrt())
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    signal.alarm(180); torch.set_num_threads(2); started=time.perf_counter()
    out=BASE/'OVERLAPPING_JOIN_NORMALIZER_V1_RESULT.json'; rows=BASE/'OVERLAPPING_JOIN_NORMALIZER_V1_ROWS.pt'
    assert not out.exists() and not rows.exists(); controls=N.controls(); assert controls['passed']
    package=BASE/'EXACT_SOURCE_EDIT_V1_PROGRAM.pt'
    assert digest(package)=='e69bff6562c48bd1b59366f118157720734d353626b0a1576626bdd437a1421a'
    program=E.load_package(torch.load(package,map_location='cpu',weights_only=True))
    from hop_ablate import load
    os.chdir(ROOT); model,_=load('attn4-rms-seed0'); model=model.double().eval()
    audits=[]; axes=[]; reuse=[]; results={}; saved={}
    with torch.inference_mode():
        for pop, worlds in N.populations().items():
            exact={key(a,b):[] for a in SCALES for b in SCALES}; predicted={k:[] for k in exact}; meta=[]
            for world in worlds:
                tok=world['tokens']; masks=world['masks']; heads=world['heads']
                context=program.prepare(tok); writes=J.contributions(program.background,tok,masks,heads)
                for w in writes.values(): reuse.append(M.correspondence(w,w[:1].expand_as(w)))
                pair=N.compile_pair(program,context,writes[0],writes[1])
                for a in SCALES:
                    for b in SCALES:
                        actual=program.edit(context,(a-1)*writes[0]+(b-1)*writes[1])
                        with J.intervene(model,masks,heads,(0,1),{0:a*writes[0],1:b*writes[1]}): native=model(tok)
                        audits.append(M.correspondence(actual,native))
                        pred=N.evaluate(pair,a,b); exact[key(a,b)].append(actual[:,-1].clone()); predicted[key(a,b)].append(pred.clone())
                        if a==1 or b==1: axes.append(M.correspondence(pred,actual[:,-1]))
                meta.extend(world['metadata'])
            exact={k:torch.cat(v) for k,v in exact.items()}; predicted={k:torch.cat(v) for k,v in predicted.items()}
            native=exact['1_1']; panels={}; groups={}; gold=torch.tensor([m['answer'] for m in meta])
            for query in (0,1):
                own='0_1' if query==0 else '1_0'; selected=torch.tensor([m['query']==query and m['hop']==3 for m in meta])
                other=torch.tensor([m['query']!=query and m['hop']==3 for m in meta])
                report=M.panel({'native':native,'cut':exact[own]},'cut',meta,selected)
                changes=exact[own][other].softmax(-1)-native[other].softmax(-1)
                cross=float(changes.gather(1,gold[other,None]).abs().mean())
                panels[str(query)]={**report,'other_query_mean_abs_gold_change':cross,
                    'eligible':report['native_accuracy']>=.9 and report['removal_gold_loss']>=.5 and cross<=.05}
                groups[str(query)]={}
                for hop in range(4):
                    sel=torch.tensor([m['query']==query and m['hop']==hop for m in meta]); scales={}
                    for a in (0.,.5):
                        for b in (0.,.5):
                            actual=exact[key(a,b)][sel]; pred=predicted[key(a,b)][sel]
                            interaction=center(actual-exact[key(a,1.)][sel]-exact[key(1.,b)][sel]+native[sel])
                            effect=center(actual-native[sel]); error=center(pred-actual)
                            den=rms(interaction); rel=rms(error)/max(den,1e-6)
                            lp=actual.log_softmax(-1); kl=(lp.exp()*(lp-pred.log_softmax(-1))).sum(-1).clamp_min(0)
                            scales[key(a,b)]={'interaction_rms':den,'joint_effect_rms':rms(effect),
                                'interaction_relative_error':rel,'joint_effect_relative_error':rms(error)/max(rms(effect),1e-6),
                                'interaction_denominator_floored':den<1e-6,'query_mean_kl':float(kl.mean()),
                                'query_p99_kl':float(torch.quantile(kl,.99)),'query_max_kl':float(kl.max()),'composition_passed':rel<=.01}
                    panels_hop={arm:M.panel({'native':native,arm:exact[arm]},arm,meta,sel) for arm in ('0_1','1_0','0_0')}
                    groups[str(query)][str(hop)]={'requests':int(sel.sum()),'scales':scales,'cut_panels':panels_hop}
            results[pop]={'eligibility':panels,'groups':groups}; saved[pop]={'metadata':meta,'exact_query_logits':exact,'predicted_query_logits':predicted}
    predictions={'pred_a_instrument':all(a['passed'] for a in audits+axes+reuse),
        'pred_b_causal_eligibility':all(g['eligible'] for p in results.values() for g in p['eligibility'].values()),
        'pred_c_shared_normalizer':all(s['composition_passed'] for p in results.values() for q in p['groups'].values() for h in q.values() for s in h['scales'].values())}
    torch.save(saved,rows)
    result={'scope':'Opened overlapping native join composition; no model reduction or invariant amplitude assumption.',
        'predictions':predictions,'controls':controls,'populations':results,'oracle_max_abs':max(a['max_abs'] for a in audits),
        'axis_max_abs':max(a['max_abs'] for a in axes),'reuse_max_abs':max(a['max_abs'] for a in reuse),
        'independent_worlds':32,'requests':512,'opaque_export_constants':program.independent_constant_count(),
        'rows_sha256':digest(rows),'runner_sha256':digest(Path(__file__)),
        'prereg_sha256':digest(BASE/'OVERLAPPING_JOIN_NORMALIZER_V1_PREREGISTRATION.md'),
        'reference_sha256':digest(BASE/'overlapping_join_normalizer_reference.py'),'wall_seconds':time.perf_counter()-started}
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='populations'},indent=2))
    print(json.dumps({p:{'eligibility':v['eligibility'],'hop3':{q:g['3']['scales'] for q,g in v['groups'].items()}} for p,v in results.items()},indent=2))


if __name__=='__main__': main()
