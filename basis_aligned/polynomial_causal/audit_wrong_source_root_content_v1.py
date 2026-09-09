"""All51 embedding roots of each fixed nominated source in20 opened errors."""
import hashlib
import json
from pathlib import Path
import signal
import time
import torch
import bad_source_content_reference as B
import exact_source_edit_reference as E
import final_source_failure_reference as F
import field_intervention_metrics as M
import frozen_payload_lineage_reference as L
from hop_ablate import load

ROOT=Path('/workspace/tensor_language');BASE=ROOT/'basis_aligned/polynomial_causal'
OUT=BASE/'WRONG_SOURCE_ROOT_CONTENT_AUDIT_V1.json';ROWS=BASE/'FRESH_ERROR_CONTENT_V1_ROWS.pt'


def main():
    signal.alarm(180);torch.set_num_threads(2);started=time.perf_counter();assert not OUT.exists()
    controls=L.controls();assert controls['passed'];data=torch.load(ROWS,map_location='cpu',weights_only=True)
    package=BASE/'EXACT_SOURCE_EDIT_V1_PROGRAM.pt';assert hashlib.sha256(package.read_bytes()).hexdigest()=='e69bff6562c48bd1b59366f118157720734d353626b0a1576626bdd437a1421a'
    program=E.load_package(torch.load(package,map_location='cpu',weights_only=True));model,_=load('attn4-rms-seed0');model=model.double().eval()
    lookup={(pop,c['world'],c['query'],c['hop'],c['side']):c for pop,cohort in data['all_native'].items() for c in cohort}
    records=[];audits=[];gatebytes=set()
    with torch.inference_mode():
        for case in data['error_metadata']:
            if not case['is_primary']:continue
            row=lookup[(case['population'],case['world'],case['query'],case['hop'],case['side'])]
            tok=row['tokens'][None];context=program.prepare(tok);cache=L.prepare(program.background,tok);e=cache['embedding'];n=tok.shape[1];j=case['binding']
            roots=torch.zeros(n,n,e.shape[-1],dtype=e.dtype);index=torch.arange(n);roots[index,index]=e[0]
            transported=L.propagate(program.background,roots,cache)
            audits.append(M.correspondence(transported.sum(0,keepdim=True),context['x']))
            selected=torch.zeros_like(transported);selected[:,2*j:2*j+2]=transported[:,2*j:2*j+2]
            layer=program.background.layers[-1];gain=layer.norm(context['x'])
            eps=torch.finfo(e.dtype).eps if layer.norm.eps is None else layer.norm.eps
            gain=torch.rsqrt(context['x'].square().mean(-1,keepdim=True)+eps)
            value=layer.v(selected*gain).reshape(n,n,layer.n_head,layer.d_head);pattern=layer.pattern(context['x'])[:,:,-1]
            messages=.5*torch.einsum('bhs,bshd->bhd',pattern,value).flatten(-2)@program.folded.T
            source,_=F.messages(program,context);whole=source[:,2*j:2*j+2].sum(1)
            audits.append(M.correspondence(messages.sum(0,keepdim=True),whole))
            nativebase=model.embed(tok)
            for native_layer in model.layers[:3]:nativebase=native_layer(nativebase)
            nativebefore=model.head(model.layers[-1](nativebase))[:,-1]
            oracle=nativebefore-B.native(model,nativebase.expand(n,-1,-1),selected)[:,-1]
            audits.append(M.correspondence(messages,oracle));audits.append(M.correspondence(context['logits'][:,-1],row['logits'][None]))
            assert not selected[2*j+2:].ne(0).any()
            gold=case['answer'];wrong=case['foil'];margins=messages[:,wrong]-messages[:,gold];wholemargin=float(whole[0,wrong]-whole[0,gold])
            perroot=[]
            for s in range(n):
                perroot.append({'position':s,'token':int(tok[0,s]),'role':'own_key' if s==2*j else 'own_value' if s==2*j+1 else 'other_key' if s<48 and s%2==0 else 'other_value' if s<48 else 'query',
                    'token_equals_wrong':int(tok[0,s])==wrong,'wrong_minus_gold_margin':float(margins[s]),'logit_rms':float(messages[s].square().mean().sqrt())})
            top=max(perroot,key=lambda r:r['wrong_minus_gold_margin']);same=sum(r['wrong_minus_gold_margin'] for r in perroot if r['token_equals_wrong'])
            records.append({k:case[k] for k in ('population','world','query','hop','side','binding','field','answer','foil')})
            records[-1].update({'nominated_repaired':case['arms']['nominated']['prediction']==gold,'whole_repaired':case['arms']['whole']['prediction']==gold,
                'whole_wrong_minus_gold_margin':wholemargin,'wrong_identity_root_margin':same,'other_identity_root_margin':wholemargin-same,
                'wrong_identity_signed_margin_ratio':same/wholemargin if abs(wholemargin)>1e-12 else None,'strongest_root':top,'roots':perroot})
            gatebytes.add(sum(v.numel()*v.element_size() for gate in cache['gates'] for v in gate.values()))
    result={'scope':'opened all20 primary errors; exact frozen-gate root attribution, not token-input intervention or enlarged field adoption',
        'controls':controls,'passed':all(c['passed'] for c in audits),'max_abs':max(c['max_abs'] for c in audits),'max_relative_rms':max(c['relative_rms'] for c in audits),
        'cases':records,'strongest_root_matches_wrong_identity':sum(c['strongest_root']['token_equals_wrong'] for c in records),
        'independent_constants':program.independent_constant_count(),'prefix_gate_cache_bytes':sorted(gatebytes),'native_coefficients_removed':0,
        'input_rows_sha256':hashlib.sha256(ROWS.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'reference_sha256':hashlib.sha256(Path(L.__file__).read_bytes()).hexdigest(),'wall_seconds':time.perf_counter()-started}
    OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2));assert result['passed']


if __name__=='__main__':main()
