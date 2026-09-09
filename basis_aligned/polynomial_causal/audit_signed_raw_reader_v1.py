"""Signed all-head raw-reader accounting on40 opened primary/paired states."""
import hashlib
import json
from pathlib import Path
import signal
import time
import torch
import exact_source_edit_reference as E
import field_intervention_metrics as M

BASE=Path(__file__).resolve().parent;OUT=BASE/'SIGNED_RAW_READER_AUDIT_V1.json';ROWS=BASE/'FRESH_ERROR_CONTENT_V1_ROWS.pt'


def main():
    signal.alarm(180);torch.set_num_threads(2);started=time.perf_counter();assert not OUT.exists()
    data=torch.load(ROWS,map_location='cpu',weights_only=True);package=BASE/'EXACT_SOURCE_EDIT_V1_PROGRAM.pt'
    assert hashlib.sha256(package.read_bytes()).hexdigest()=='e69bff6562c48bd1b59366f118157720734d353626b0a1576626bdd437a1421a'
    program=E.load_package(torch.load(package,map_location='cpu',weights_only=True));layer=program.background.layers[-1]
    h=layer.n_head;d=layer.d_head
    lookup={(pop,c['world'],c['query'],c['hop'],c['side']):c for pop,cohort in data['all_native'].items() for c in cohort}
    cases=[];audits=[]
    with torch.inference_mode():
        # D[h,e,c] is the raw embedding-to-logit dictionary, including both lerp factors.
        dictionary=torch.einsum('chd,hdi,ei->hec',program.folded.reshape(29,h,d),layer.v.weight.reshape(h,d,-1),program.background.embed.weight[:24])/16
        centered_dictionary=dictionary-dictionary.mean(-1,keepdim=True)
        for i,case in enumerate(data['error_metadata']):
            row=lookup[(case['population'],case['world'],case['query'],case['hop'],case['side'])]
            tok=row['tokens'][None];context=program.prepare(tok);source=2*case['binding']+1;entity=int(tok[0,source]);gold=case['answer'];wrong=case['foil']
            eps=torch.finfo(context['x'].dtype).eps if layer.norm.eps is None else layer.norm.eps
            gain=torch.rsqrt(context['x'].square().mean(-1,keepdim=True)+eps)[0,source,0]
            gates=layer.pattern(context['x'])[0,:,-1,source]*gain
            terms=gates[:,None]*dictionary[:,entity];raw=terms.sum(0)
            expected=data['error_query_logits']['native'][i]-data['error_query_logits']['raw'][i]
            audits.append(M.correspondence(raw,expected));audits.append(M.correspondence(context['logits'][0,-1],data['error_query_logits']['native'][i]))
            centered=terms-terms.mean(-1,keepdim=True);parts={}
            for sign,sel in (('positive',gates>0),('negative',gates<0)):
                value=centered[sel].sum(0)
                parts[sign]={'heads':sel.nonzero().flatten().tolist(),'gold_logit_contribution':float(value[gold]),'wrong_logit_contribution':float(value[wrong]),
                             'wrong_minus_gold_margin':float(value[wrong]-value[gold]),'source_value_logit_contribution':float(value[entity])}
            heads=[]
            for head in range(h):
                dec=centered_dictionary[head,entity]
                heads.append({'head':head,'signed_gate':float(gates[head]),'decoder_argmax':int(dec.argmax()),'decoder_argmin':int(dec.argmin()),
                    'decoder_source_value':float(dec[entity]),'decoder_gold':float(dec[gold]),'decoder_wrong':float(dec[wrong]),
                    'gold_logit_contribution':float(centered[head,gold]),'wrong_logit_contribution':float(centered[head,wrong]),
                    'wrong_minus_gold_margin':float(centered[head,wrong]-centered[head,gold])})
            cases.append({k:case[k] for k in ('population','world','query','hop','primary_side','side','is_primary','binding','field','answer','foil')})
            cases[-1].update({'source_value':entity,'source_value_is_gold':entity==gold,'source_value_is_wrong':entity==wrong,
                             'signed_parts':parts,'heads':heads,'raw_removal_prediction':case['arms']['raw']['prediction']})
    descriptor=[]
    for head in range(h):
        descriptor.append({'head':head,'tokens_argmax_equals_source':int((dictionary[head].argmax(-1)==torch.arange(24)).sum()),
                           'tokens_argmin_equals_source':int((dictionary[head].argmin(-1)==torch.arange(24)).sum())})
    result={'scope':'opened signed raw-reader diagnostic; no replacement, fitting, selected head or field expansion',
        'passed':all(c['passed'] for c in audits),'replay_max_abs':max(c['max_abs'] for c in audits),'replay_max_relative_rms':max(c['relative_rms'] for c in audits),
        'dictionary_formula':'D[h,e,c]=folded[c,h,:] V[h,:,:] embedding[e,:] /16; signed gate=P[head,query,source]*native_RMS_gain[source]',
        'dictionary_values':dictionary.numel(),'dictionary_bytes':dictionary.numel()*dictionary.element_size(),
        'dictionary':dictionary.tolist(),'dictionary_descriptors':descriptor,'cases':cases,'independent_constants':program.independent_constant_count(),
        'native_coefficients_removed':0,'input_rows_sha256':hashlib.sha256(ROWS.read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'wall_seconds':time.perf_counter()-started}
    OUT.write_text(json.dumps(result,indent=2)+'\n');assert result['passed']
    print(json.dumps({k:v for k,v in result.items() if k not in ('dictionary','cases')},indent=2))
    print(json.dumps([c for c in cases if c['is_primary'] and c['source_value_is_gold']],indent=2))


if __name__=='__main__':main()
