"""Native source-port factorial for the overlapping backward join's two consumers."""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
ROOT=Path('/workspace/tensor_language');BASE=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(ROOT),str(BASE)]
import torch
import exact_source_edit_reference as E
import join_contribution_context_reference as J
import overlapping_join_normalizer_reference as N
import source_port_interchange_reference as P
import field_intervention_metrics as M
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def center(x): return x-x.mean(-1,keepdim=True)
def rms(x): return float(x.square().mean().sqrt())


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    signal.alarm(180);torch.set_num_threads(2);started=time.perf_counter()
    source=BASE/'OVERLAPPING_JOIN_NORMALIZER_V1_ROWS.pt';out=BASE/'OVERLAPPING_JOIN_PORTS_V1_RESULT.json';rows=BASE/'OVERLAPPING_JOIN_PORTS_V1_ROWS.pt'
    assert digest(source)=='fb74b8938f81f3f000fbdc5be8e35d431632bfaaaeb2fd8d111d8311db0769e0'
    assert not out.exists() and not rows.exists();previous=torch.load(source,map_location='cpu',weights_only=True)
    controls=P.controls();assert controls['passed']
    package=BASE/'EXACT_SOURCE_EDIT_V1_PROGRAM.pt'
    assert digest(package)=='e69bff6562c48bd1b59366f118157720734d353626b0a1576626bdd437a1421a'
    program=E.load_package(torch.load(package,map_location='cpu',weights_only=True))
    from hop_ablate import load
    os.chdir(ROOT);model,_=load('attn4-rms-seed0');model=model.double().eval()
    audits=[];replays=[];reuse=[];results={};saved={}
    with torch.inference_mode():
        for pop,worlds in N.populations().items():
            logits={str(s):[] for s in range(8)};meta=[]
            for world in worlds:
                tok=world['tokens'];context=program.prepare(tok)
                write=J.contributions(program.background,tok,world['masks'],world['heads'])[1]
                reuse.append(M.correspondence(write,write[:1].expand_as(write)))
                for s in range(8):
                    actual=P.execute(program,context,P.ports(program,context,-write,s))
                    native=P.native(model,context['x'],-write,s);audits.append(M.correspondence(actual,native))
                    logits[str(s)].append(actual[:,-1].clone())
                meta.extend(world['metadata'])
            assert meta==previous[pop]['metadata'];logits={k:torch.cat(v) for k,v in logits.items()}
            replays.extend((M.correspondence(logits['0'],previous[pop]['exact_query_logits']['1_1']),
                            M.correspondence(logits['7'],previous[pop]['exact_query_logits']['1_0'])))
            panels={};groups={}
            for query in (0,1):
                groups[str(query)]={}
                for hop in range(4):
                    sel=torch.tensor([m['query']==query and m['hop']==hop for m in meta])
                    factorial={s:M.panel({'native':logits['0'],s:v},s,meta,sel) for s,v in logits.items()}
                    full=factorial['7']['gold_probability_change'];keys=factorial['3']['gold_probability_change'];value=factorial['4']['gold_probability_change']
                    eligible=abs(full)>=.001;keyratio=keys/full if eligible else None;vratio=abs(value/full) if eligible else None
                    lp=logits['7'][sel].log_softmax(-1);kl=(lp.exp()*(lp-logits['3'][sel].log_softmax(-1))).sum(-1).clamp_min(0)
                    effect=center(logits['7'][sel]-logits['0'][sel]);rel=rms(center(logits['3'][sel]-logits['7'][sel]))/max(rms(effect),1e-6)
                    groups[str(query)][str(hop)]={'factorial':factorial,'full_effect_rms':rms(effect),
                        'keys_full_effect_relative_error':rel,'keys_full_query_mean_kl':float(kl.mean()),'keys_full_query_p99_kl':float(torch.quantile(kl,.99)),
                        'keys_full_query_max_kl':float(kl.max()),'sufficiency_passed':rel<=.01 and float(kl.mean())<=.001 and float(torch.quantile(kl,.99))<=.01}
                    if hop==3:panels[str(query)]={'full_gold_change':full,'keys_gold_change':keys,'value_gold_change':value,
                        'key_ratio':keyratio,'absolute_value_ratio':vratio,'gold_probability_interaction':full-keys-value,
                        'denominator_valid':eligible,'nomination_passed':eligible and .9<=keyratio<=1.1 and vratio<=.1}
            results[pop]={'nomination':panels,'groups':groups};saved[pop]={'metadata':meta,'query_logits':logits}
    predictions={'pred_a_instrument':all(a['passed'] for a in audits+replays+reuse),
        'pred_b_shared_address':all(g['nomination_passed'] for p in results.values() for g in p['nomination'].values()),
        'pred_c_sufficiency':all(g['sufficiency_passed'] for p in results.values() for q in p['groups'].values() for g in q.values())}
    torch.save(saved,rows);result={'scope':'Opened cross-consumer source-port screen; native keys/values remain explicit and priced.',
        'predictions':predictions,'controls':controls,'populations':results,'oracle_max_abs':max(a['max_abs'] for a in audits),
        'saved_replay_max_abs':max(a['max_abs'] for a in replays),'reuse_max_abs':max(a['max_abs'] for a in reuse),
        'independent_worlds':32,'requests':512,'opaque_export_constants':program.independent_constant_count(),
        'source_sha256':digest(source),'rows_sha256':digest(rows),'runner_sha256':digest(Path(__file__)),
        'prereg_sha256':digest(BASE/'OVERLAPPING_JOIN_PORTS_V1_PREREGISTRATION.md'),'wall_seconds':time.perf_counter()-started}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('populations','controls')},indent=2))
    print(json.dumps({p:{'nomination':v['nomination'],'hop3_errors':{q:g['3']['keys_full_effect_relative_error'] for q,g in v['groups'].items()}} for p,v in results.items()},indent=2))


if __name__=='__main__':main()
