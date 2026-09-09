"""Circuit screen for an origin-key x endpoint-value interaction node."""
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
import join_origin_writer_reference as O
import forward_endpoint_program_reference as F
import overlapping_join_normalizer_reference as N
import key_payload_interaction_reference as I
import field_intervention_metrics as M
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    signal.alarm(180);torch.set_num_threads(2);started=time.perf_counter()
    out=BASE/'ORIGIN_ENDPOINT_INTERACTION_V1_RESULT.json';rows=BASE/'ORIGIN_ENDPOINT_INTERACTION_V1_ROWS.pt'
    source=BASE/'OVERLAPPING_JOIN_KEY_GAIN_V1_ROWS.pt'
    assert not out.exists() and not rows.exists()
    assert digest(source)=='83f2ac14394f78c7b6e9e084328c617f9b631bd0c865fb347c29bcdf2fda23a7'
    previous=torch.load(source,map_location='cpu',weights_only=True);controls=I.controls();assert controls['passed']
    package=BASE/'EXACT_SOURCE_EDIT_V1_PROGRAM.pt'
    assert digest(package)=='e69bff6562c48bd1b59366f118157720734d353626b0a1576626bdd437a1421a'
    program=E.load_package(torch.load(package,map_location='cpu',weights_only=True))
    from hop_ablate import load
    os.chdir(ROOT);model,_=load('attn4-rms-seed0');model=model.double().eval()
    audits=[];replays=[];results={};saved={}
    with torch.inference_mode():
        for pop,worlds in N.populations().items():
            logits={k:[] for k in ('native','without_interaction','key_cut','value_cut','both_cut')};meta=[]
            for w in worlds:
                tok=w['tokens'];context=program.prepare(tok)
                key=O.origin_write(program.background,tok,{1:w['masks'][1][0]},{1:2})[1]
                mask=w['masks'][0].clone();mask[:,:,:48:2]=False
                payload=F.messages(program.background,tok,mask)
                term=I.interaction(program,context,key,payload);arms=I.native_arms(model,context['x'],key,payload)
                a,b,c,d=(arms[k] for k in ((False,False),(True,False),(False,True),(True,True)))
                audits.extend((M.correspondence(a+d,b+c+term),M.correspondence(context['logits']-term,b+c-d),M.correspondence(context['logits'],a)))
                for name,value in (('native',a),('without_interaction',context['logits']-term),('key_cut',b),('value_cut',c),('both_cut',d)):
                    logits[name].append(value[:,-1].clone())
                meta.extend(w['metadata'])
            assert meta==previous[pop]['metadata'];logits={k:torch.cat(v) for k,v in logits.items()}
            replays.append(M.correspondence(logits['native'],previous[pop]['query_logits']['native']))
            groups={};gold=torch.tensor([m['answer'] for m in meta])
            for query in (0,1):
                groups[str(query)]={}
                for hop in range(4):
                    sel=torch.tensor([m['query']==query and m['hop']==hop for m in meta])
                    panel=M.panel(logits,'without_interaction',meta,sel)
                    lp=logits['native'][sel].log_softmax(-1);candidate=logits['without_interaction'][sel]
                    kl=(lp.exp()*(lp-candidate.log_softmax(-1))).sum(-1).clamp_min(0)
                    changes=candidate.softmax(-1)-lp.exp();abs_gold=float(changes.gather(1,gold[sel,None]).abs().mean())
                    target=query==0 and hop==3
                    passed=panel['removal_gold_loss']>=.5 if target else abs_gold<=.05
                    capability=panel['native_accuracy']>=.9 if hop==3 else True
                    groups[str(query)][str(hop)]={**panel,'mean_absolute_gold_change':abs_gold,'target':target,
                        'query_mean_kl':float(kl.mean()),'query_p99_kl':float(torch.quantile(kl,.99)),'query_max_kl':float(kl.max()),
                        'nomination_panel_passed':passed and capability}
            results[pop]=groups;saved[pop]={'metadata':meta,'query_logits':logits}
    predictions={'pred_a_instrument':all(a['passed'] for a in audits+replays),
        'pred_b_selective_circuit':all(g['nomination_panel_passed'] for p in results.values() for q in p.values() for g in q.values())}
    torch.save(saved,rows);result={'scope':'Opened additive interaction-node knockout; not a physical L2 source removal or structural reduction.',
        'predictions':predictions,'controls':controls,'populations':results,'oracle_max_abs':max(a['max_abs'] for a in audits),
        'saved_replay_max_abs':max(a['max_abs'] for a in replays),'independent_worlds':32,'requests':512,
        'opaque_export_constants':program.independent_constant_count(),'rows_sha256':digest(rows),'source_sha256':digest(source),
        'runner_sha256':digest(Path(__file__)),'reference_sha256':digest(BASE/'key_payload_interaction_reference.py'),
        'prereg_sha256':digest(BASE/'ORIGIN_ENDPOINT_INTERACTION_V1_PREREGISTRATION.md'),'wall_seconds':time.perf_counter()-started}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
