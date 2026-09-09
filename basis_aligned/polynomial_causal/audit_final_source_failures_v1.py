"""All20 opened failure pairs,40 states,960 final binding-read interventions."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
import torch
import exact_source_edit_reference as E
import final_source_failure_reference as F
import field_intervention_metrics as M

ROOT=Path('/workspace/tensor_language');BASE=ROOT/'basis_aligned/polynomial_causal'
OUT=BASE/'FINAL_SOURCE_FAILURE_ATLAS_V1.json';CORPUS=BASE/'ENTITY_SYMMETRY_COUNTEREXAMPLES_V1.json'


def main():
    signal.alarm(180);torch.set_num_threads(2);started=time.perf_counter();assert not OUT.exists()
    checks=F.controls();assert checks['passed'];cases=json.loads(CORPUS.read_text())['all_pairs']
    cases=[c for c in cases if c['old_prediction']!=c['renamed_prediction_original_labels']];assert len(cases)==20
    package=BASE/'EXACT_SOURCE_EDIT_V1_PROGRAM.pt';checkpoint=ROOT/'runs_hop/attn4-rms-seed0/model.pt'
    assert hashlib.sha256(package.read_bytes()).hexdigest()=='e69bff6562c48bd1b59366f118157720734d353626b0a1576626bdd437a1421a'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()=='c9388cc8b1ba7e95c6a043cbf4d01592987447867a50781d3babe31ce3f8358b'
    program=E.load_package(torch.load(package,map_location='cpu',weights_only=True))
    sys.path.insert(0,str(ROOT));from hop_ablate import load
    model,_=load('attn4-rms-seed0');model=model.double().eval();records=[];maximum=relative=0.;summary=Counter();roles=Counter()
    with torch.inference_mode():
        for case in cases:
            sigma=list(case['entity_permutation'])+list(range(24,29))
            for side in ('old','renamed'):
                tok=torch.tensor(case[side+'_tokens'])[None];context=program.prepare(tok);source,residual=F.messages(program,context)
                native=model(tok)[:,-1];base=model.embed(tok)
                for layer in model.layers[:3]:base=layer(base)
                pairs=source[:,:48].reshape(1,24,2,29).sum(2)[0];predicted=context['logits'][:,-1]-pairs;oracle=F.native_binding_cuts(model,base)
                for a,b in ((source.sum(1)+residual,native),(predicted,oracle)):
                    audit=M.correspondence(a,b);assert audit['finite'];maximum=max(maximum,audit['max_abs']);relative=max(relative,audit['relative_rms'])
                answer=case['correct_answer_original_labels'];old_prediction=case['old_prediction'];new_prediction=case['renamed_prediction_original_labels']
                original_foil=(old_prediction if old_prediction!=answer else new_prediction) if side=='old' else (new_prediction if new_prediction!=answer else old_prediction)
                if side=='renamed':answer=sigma[answer];foil=sigma[original_foil]
                else:foil=original_foil
                winner=int(native.argmax());wrong=winner!=answer;summary['wrong_states' if wrong else 'correct_states']+=1
                f=torch.empty(24,dtype=torch.long).scatter_(0,tok[0,:48:2],tok[0,1:48:2]);positions={int(k):j for j,k in enumerate(tok[0,:48:2])}
                chain=[int(tok[0,49])]
                for _ in range(case['hop']):chain.append(int(f[chain[-1]]))
                chain_slots={positions[k] for k in chain[:-1]};answer_slot=positions[chain[-2]]
                carrier=max(positions[chain[1]],positions[chain[2]]) if case['hop']==3 else None
                p=native.softmax(-1)[0];q=predicted.softmax(-1);per_source=[]
                for j in range(24):
                    role='suffix_carrier' if j==carrier else 'answer_binding' if j==answer_slot else 'other_chain_binding' if j in chain_slots else 'off_chain'
                    gain=float(q[j,answer]-p[answer]);rescue=int(predicted[j].argmax())==answer
                    per_source.append({'binding':j,'key':int(tok[0,2*j]),'value':int(tok[0,2*j+1]),'role':role,
                                       'wrong_vs_gold_message_margin':float(pairs[j,foil]-pairs[j,answer]),
                                       'gold_probability_gain_after_cut':gain,'prediction_after_cut':int(predicted[j].argmax()),'gold_after_cut':rescue})
                top=max(per_source,key=lambda s:s['wrong_vs_gold_message_margin'])
                if wrong:
                    summary['any_answer_rescue']+=int(any(s['gold_after_cut'] for s in per_source))
                    summary['rescue_with_gain_ge_025']+=int(any(s['gold_after_cut'] and s['gold_probability_gain_after_cut']>=.25 for s in per_source))
                    summary['off_chain_answer_rescue']+=int(any(s['role']=='off_chain' and s['gold_after_cut'] for s in per_source))
                    roles[top['role']]+=1
                records.append({'population':case['population'],'row':case['row'],'world':case['world'],'side':side,'hop':case['hop'],
                                'answer':answer,'native_prediction':winner,'foil':foil,'native_wrong':wrong,'gold_probability':float(p[answer]),
                                'top_wrong_margin_source':top['binding'],'sources':per_source})
    receipt={'scope':'opened-case discovery, all20 disagreements/bothcontexts/all24binding cuts; no selected deletion program or heldout claim',
             'controls':checks,'oracle_max_abs':maximum,'oracle_max_relative_rms':relative,'passed':maximum<=1e-9 and relative<=1e-10,
             'summary':dict(summary),'top_wrong_support_source_roles':dict(roles),'states':records,
             'case_corpus_sha256':hashlib.sha256(CORPUS.read_bytes()).hexdigest(),'independent_constants':program.independent_constant_count(),
             'reference_sha256':hashlib.sha256(Path(F.__file__).read_bytes()).hexdigest(),'wall_seconds':time.perf_counter()-started}
    OUT.write_text(json.dumps(receipt,indent=2)+'\n');assert receipt['passed'];print(json.dumps({k:v for k,v in receipt.items() if k!='states'},indent=2))


if __name__=='__main__':main()
