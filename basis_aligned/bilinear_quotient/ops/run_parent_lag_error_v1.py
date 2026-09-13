#!/usr/bin/env python3
# BQGATE:40prefixes;200suffixes;240seconds.
"""pred_a field partition and native reencoding relative error <=1e-10.
pred_b long-lag >=70% weighted error alignment in FineWeb and legal/patent.
pred_c weighted output-error additive closure <=10% in all domains.
pred_d original full-candidate weighted error replay <=1e-6 relative.
40prefixes;200suffixes;240seconds. Same frozen corpus panel, native unit-parent
removal. Hybrid errors at lag<64 and >=64 are diagnostics, not adopted circuits.
"""
import sys,os,json,time,signal
from pathlib import Path
from hashlib import sha256
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parents[2]/'polynomial_causal';sys.path.insert(0,str(P))
from fastload import load_model_fast
from live_crossfirst_prefix_v1 import prepare as live_prefix,assembled
from regional_even_routing_v1 import routing
from regional_cue_row_check_v1 import validate

@torch.no_grad()
def main():
    binding=json.loads((P/'PARENT_LAG_ERROR_V1_BINDING.json').read_text())
    assert all(sha256(Path(k).read_bytes()).hexdigest()==v for k,v in binding['files'].items())
    panel=json.loads((P/'PARENT_CORPUS_TRANSFER_V1_ROWS.json').read_text());rows=panel['rows']
    assert len(rows)==40
    assert all(sha256(Path(k).read_bytes()).hexdigest()==v for k,v in panel['sources'].items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print('40prefixes;200suffixes;240seconds');return
    out=P/'PARENT_LAG_ERROR_V1_RESULT.json';assert not out.exists()
    signal.alarm(240);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter()
    model=load_model_fast().cuda().eval();weights=assembled.load_weights(model.state_dict(),'cuda')
    native=weights['routing'];candidates={}
    for name,file in [('query_product','SHARED_QUERY_PRODUCT_FIT_V1_PROGRAM.pt')]:
        rot=torch.load(P/file,weights_only=True)['basis_rotation'].cuda().double()
        candidates[name]=dict(native,key_basis=[native['key_basis'][0],native['key_basis'][1]@rot])
    w=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)['direction'].cuda()
    corpora={name:torch.load(name,weights_only=True,mmap=True) for name in panel['sources']}
    domains=['fineweb','discussion','reference','biomedical','legal_patent']
    sign=torch.ones(64,device='cuda',dtype=torch.float64);sign[::2]=-1
    exact_reencoding=dict(native,key_basis=[native['key_basis'][0],native['key_basis'][1]*sign])
    cells=[];replay=0.;partition=0.
    methods=['query_product','long','short']
    long=(torch.arange(128,device='cuda')[:,None]-torch.arange(128,device='cuda')[None,:])>=64
    for index,row in enumerate(rows):
        ids=corpora[row['source']][row['source_row']:row['source_row']+1,:128].cuda()
        x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
        for block in model.transformer.h[:9]:x,v1=block(x,v1,x0)
        block=model.transformer.h[9];raw=block.lambdas[0]*x+block.lambdas[1]*x0
        att,v1=block.attn(F.rms_norm(raw,(1152,)),v1);z=raw+att
        live=dict(raw9=raw,att9=att,h9=z+block.mlp(F.rms_norm(z,(1152,))),v1=v1,x0=x0)
        norm=F.rms_norm(live['raw9'],(1152,));values=(norm.double()@native['current_value_reader'])[...,None]
        ng=routing(norm,native,1);cg=routing(norm,candidates['query_product'],1);delta=cg-ng
        parent=ng@values;long_error=(delta*long)@values;short_error=(delta*(~long))@values
        parents={'native':parent,'query_product':cg@values,'long':parent+long_error,'short':parent+short_error}
        partition=max(partition,float(((parents['query_product']-parent)-long_error-short_error).norm()/(parents['query_product']-parent).norm().clamp_min(1e-30)))
        replay=max(replay,float(((routing(norm,exact_reencoding,1)@values)-parents['native']).norm()/parents['native'].norm().clamp_min(1e-30)))
        def score(h9):
            state=h9
            for block in model.transformer.h[10:]:state,_=block(state,live['v1'],live['x0'])
            return (30*torch.tanh(model.lm_head(F.rms_norm(state[:,-1],(1152,)))/30))[0].double()
        baseline=score(live['h9']);logbase=baseline.log_softmax(-1);prob=logbase.exp();outputs={}
        for name,amplitude in parents.items():
            z=live['raw9']+(live['att9']-(amplitude*w).to(live['att9'].dtype))
            h=z+model.transformer.h[9].mlp(F.rms_norm(z,(1152,)));outputs[name]=score(h)
        reference=baseline-outputs['native'];centered=reference-reference.mean();weighted=reference-(prob*reference).sum()
        logref=outputs['native'].log_softmax(-1);pref=logref.exp();klref=(pref*(logref-logbase)).sum()
        full_error=outputs['native']-outputs['query_product'];full_weighted=full_error-(prob*full_error).sum()
        closure=full_error-(outputs['native']-outputs['long'])-(outputs['native']-outputs['short']);closure-= (prob*closure).sum()
        for method in methods:
            error=(baseline-outputs[method])-reference;ce=error-error.mean();we=error-(prob*error).sum()
            kl=(pref*(logref-outputs[method].log_softmax(-1))).sum()
            cells.append(dict(row=index,family=domains.index(row['domain']),method=method,
                centered_error2=float(ce.square().sum()),centered_reference2=float(centered.square().sum()),
                weighted_error2=float((prob*we.square()).sum()),weighted_reference2=float((prob*weighted.square()).sum()),
                candidate_kl=float(kl),native_removal_kl=float(klref),
                alignment=float((prob*we*full_weighted).sum()),full_error2=float((prob*full_weighted.square()).sum()),closure2=float((prob*closure.square()).sum())))
    groups=[]
    for family in range(5):
        for method in methods:
            sub=[c for c in cells if c['family']==family and c['method']==method]
            total=lambda key:sum(c[key] for c in sub)
            groups.append(dict(family=family,domain=domains[family],method=method,
                centered_relative_error=(total('centered_error2')/total('centered_reference2'))**.5,
                weighted_relative_error=(total('weighted_error2')/total('weighted_reference2'))**.5,
                relative_kl=total('candidate_kl')/total('native_removal_kl'),
                weighted_alignment=total('alignment')/total('full_error2'),weighted_closure=(total('closure2')/total('full_error2'))**.5,mean_candidate_kl=total('candidate_kl')/len(sub),mean_native_removal_kl=total('native_removal_kl')/len(sub)))
    old=json.loads((P/'PARENT_CORPUS_TRANSFER_V1_RESULT.json').read_text())['groups']
    metric_replay=max(abs(g['weighted_relative_error']/next(o['weighted_relative_error'] for o in old if o['family']==g['family'] and o['method']=='query_product')-1) for g in groups if g['method']=='query_product')
    result={'pred_a':max(replay,partition)<=1e-10,
        'pred_b':all(g['weighted_alignment']>=.7 for g in groups if g['method']=='long' and g['family'] in (0,4)),
        'pred_c':all(g['weighted_closure']<=.1 for g in groups),'pred_d':metric_replay<=1e-6,
        'basis_reencoding_error':replay,'field_partition_error':partition,'prior_metric_replay_relative':metric_replay,
        'groups':groups,'cells':cells,'seconds':time.perf_counter()-start,'scope':__doc__}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='cells'},indent=2))
if __name__=='__main__':main()
