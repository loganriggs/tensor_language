"""Registered opened-case content-path mediation; CPU, two threads, 180 seconds."""
import hashlib
import json
from pathlib import Path
import signal
import time
import torch
import bad_source_content_reference as B
import exact_source_edit_reference as E
import final_source_failure_reference as F
import forward_endpoint_program_reference as J
import field_intervention_metrics as M
from hop_ablate import load

ROOT=Path('/workspace/tensor_language');BASE=ROOT/'basis_aligned/polynomial_causal'
OUT=BASE/'BAD_SOURCE_CONTENT_MEDIATION_V1_RESULT.json'


def main():
    signal.alarm(180);torch.set_num_threads(2);started=time.perf_counter();assert not OUT.exists()
    controls=B.controls();assert controls['passed']
    cases=B.manifest(json.loads((BASE/'FINAL_SOURCE_FAILURE_ATLAS_V1.json').read_text()),
                     json.loads((BASE/'ENTITY_SYMMETRY_COUNTEREXAMPLES_V1.json').read_text()))
    assert cases==json.loads((BASE/'BAD_SOURCE_CONTENT_MEDIATION_V1_CASES.json').read_text())
    package=BASE/'EXACT_SOURCE_EDIT_V1_PROGRAM.pt';checkpoint=ROOT/'runs_hop/attn4-rms-seed0/model.pt'
    assert hashlib.sha256(package.read_bytes()).hexdigest()=='e69bff6562c48bd1b59366f118157720734d353626b0a1576626bdd437a1421a'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()=='c9388cc8b1ba7e95c6a043cbf4d01592987447867a50781d3babe31ce3f8358b'
    program=E.load_package(torch.load(package,map_location='cpu',weights_only=True))
    model,_=load('attn4-rms-seed0');model=model.double().eval()
    rows=[];saved={k:[] for k in ('native','whole','nominated','opposite','joint','identity')}
    checks=[];composition=0.
    with torch.inference_mode():
        for case in cases:
            tok=torch.tensor(case['tokens'])[None];j=case['binding'];context=program.prepare(tok)
            base=model.embed(tok)
            for layer in model.layers[:3]:base=layer(base)
            raw=torch.zeros_like(context['x']);raw[:,2*j+1]=program.background.embed(tok)[:,2*j+1]/8
            mask,_=J.joins(tok);mask[:,:2*j]=False;mask[:,2*j+2:]=False
            joined=J.messages(program.background,tok,mask)
            native_join=J.native_messages(model,tok,mask)
            native_raw=torch.zeros_like(base);native_raw[:,2*j+1]=model.embed(tok)[:,2*j+1]/8
            content={'raw':B.execute(program,context,raw),'join':B.execute(program,context,joined)}
            joint=B.execute(program,context,raw+joined);identity=B.execute(program,context,raw*0)
            for actual,expected in ((context['logits'],model(tok)),(joined,native_join),
                                    (content['raw'],B.native(model,base,native_raw)),
                                    (content['join'],B.native(model,base,native_join)),
                                    (joint,B.native(model,base,native_raw+native_join)),
                                    (identity,context['logits'])):
                checks.append(M.correspondence(actual,expected))
            composition=max(composition,float((joint-content['raw']-content['join']+context['logits']).abs().max()))
            source,residual=F.messages(program,context)
            whole=context['logits'][:,-1]-source[:,2*j:2*j+2].sum(1)
            checks.append(M.correspondence(whole,F.native_binding_cuts(model,base)[j:j+1]))
            checks.append(M.correspondence(source.sum(1)+residual,context['logits'][:,-1]))
            outputs={'native':context['logits'][:,-1],'whole':whole,
                     'nominated':content[case['field']][:,-1],
                     'opposite':content['join' if case['field']=='raw' else 'raw'][:,-1],
                     'joint':joint[:,-1],'identity':identity[:,-1]}
            for arm,value in outputs.items():saved[arm].append(value[0].clone())
            p=outputs['native'].softmax(-1)[0];answer=case['answer'];foil=case['foil']
            metrics={}
            for arm,value in outputs.items():
                q=value.softmax(-1)[0]
                metrics[arm]={'prediction':int(value.argmax()),'gold_probability_change':float(q[answer]-p[answer]),
                              'wrong_probability_loss':float(p[foil]-q[foil]),
                              'margin_reduction':float(outputs['native'][0,foil]-outputs['native'][0,answer]-value[0,foil]+value[0,answer])}
            assert not case['is_primary'] or int(outputs['native'].argmax())==foil
            rows.append({**case,'join_path_present':bool(mask.any()),'native_correct':int(outputs['native'].argmax())==answer,'arms':metrics})
    logits={k:torch.stack(v) for k,v in saved.items()};groups={};b_gates=[]
    for field in ('raw','join'):
        selected=[c['is_primary'] and c['field']==field for c in rows];group=[r for r,s in zip(rows,selected) if s]
        mean=lambda arm,key:sum(r['arms'][arm][key] for r in group)/len(group)
        whole_margin=mean('whole','margin_reduction');nominated_margin=mean('nominated','margin_reduction')
        repairable=[r for r in group if r['arms']['whole']['prediction']==r['answer']]
        repaired=sum(r['arms']['nominated']['prediction']==r['answer'] for r in repairable)
        gate=whole_margin>0 and nominated_margin>=.5*whole_margin and mean('nominated','wrong_probability_loss')>=.25 and bool(repairable) and repaired>=.5*len(repairable)
        b_gates.append(gate)
        groups[field]={'n':len(group),'whole_margin_reduction':whole_margin,'nominated_margin_reduction':nominated_margin,
                       'wrong_probability_loss':mean('nominated','wrong_probability_loss'),'whole_repairable':len(repairable),
                       'also_repaired_by_content':repaired,'repair_gate_tested':bool(repairable),'passed':gate,
                       'arms':{arm:M.panel(logits,arm,cases,selected) for arm in saved}}
    peer=[not r['is_primary'] and r['native_correct'] for r in rows];assert sum(peer)==19
    specificity=M.panel(logits,'nominated',cases,peer)
    specificity['subgroups']={field:M.panel(logits,'nominated',cases,[s and r['field']==field for s,r in zip(peer,rows)]) for field in ('raw','join')}
    result={'scope':'opened-case discovery; no heldout identification or adopted deletion program','controls':controls,
            'prediction_a':all(c['passed'] for c in checks),'prediction_b':all(b_gates),
            'prediction_c':abs(specificity['gold_probability_change'])<=.10,'prediction_d':composition<=1e-9,
            'oracle_max_abs':max(c['max_abs'] for c in checks),'oracle_max_relative_rms':max(c['relative_rms'] for c in checks),
            'composition_max_abs':composition,'groups':groups,'correct_peer_specificity':specificity,'cases':rows,
            'independent_constants':program.independent_constant_count(),'wall_seconds':time.perf_counter()-started,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'reference_sha256':hashlib.sha256(Path(B.__file__).read_bytes()).hexdigest()}
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))


if __name__=='__main__':main()
