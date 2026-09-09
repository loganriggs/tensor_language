"""Necessary causal-response bound for one shared address per head/query."""
import hashlib
import json
from pathlib import Path
import signal
import time
import torch
import exact_source_edit_reference as E
import query_factor_transfer_reference as Q
import field_intervention_metrics as M

BASE=Path(__file__).resolve().parent;ROWS=BASE/'HOP_QUERY_FACTOR_TRANSFER_V1_ROWS.pt';OUT=BASE/'HOP_COMMON_ADDRESS_BOUND_V1.json'


def rank_one_error(matrix):
    u,s,v=torch.linalg.svd(matrix,full_matrices=False)
    best=(u[...,:1]*s[...,:1].unsqueeze(-2))@v[...,:1,:]
    residual=(matrix-best).square().sum((-1,-2));total=matrix.square().sum((-1,-2))
    return residual,total,s,best


def main():
    signal.alarm(180);torch.set_num_threads(2);started=time.perf_counter();assert not OUT.exists()
    data=torch.load(ROWS,map_location='cpu',weights_only=True);package=BASE/'EXACT_SOURCE_EDIT_V1_PROGRAM.pt'
    assert hashlib.sha256(package.read_bytes()).hexdigest()=='e69bff6562c48bd1b59366f118157720734d353626b0a1576626bdd437a1421a'
    program=E.load_package(torch.load(package,map_location='cpu',weights_only=True));layer=program.background.layers[-1]
    records=[];audits=[];summary={};zero_floor=1e-6
    with torch.inference_mode():
        # Live exact and non-rank-one controls for the fixed structural bound.
        a=torch.tensor([[1.,2.,3.],[2.,4.,6.]],dtype=torch.float64);b=torch.eye(3,dtype=torch.float64)
        ar,at,_,_=rank_one_error(a);br,bt,_,_=rank_one_error(b)
        assert float(ar)<1e-20 and abs(float(br/bt)-2/3)<1e-12
        for pop,block in data.items():
            totals={h:{'message_residual_sq':0.,'message_total_sq':0.,'score_residual_sq':0.,'score_total_sq':0.,'cases_above_1pct':0,'cases':0} for h in range(4)}
            tokens=block['native_tokens']
            for world in range(8):
                for query in range(24):
                    tok=tokens[world*96+query*4:world*96+query*4+4];context=program.prepare(tok)
                    audits.append(M.correspondence(context['x'][:,:48],context['x'][:1,:48].expand(4,-1,-1)))
                    p=layer.pattern(context['x'])[:,:,-1,:48].permute(1,0,2)
                    value=context['features']['v'][0,:48]
                    decoder=.5*torch.einsum('shd,chd->hsc',value,program.folded.reshape(29,4,layer.d_head))
                    messages=p[:,:,:,None]*decoder[:,None]
                    read=messages.sum((0,2));expected=Q.read(program,context,context,0)
                    audits.append(M.correspondence(read,expected))
                    # Replay every hop's registered saved native binding read.
                    indices=[next(i for i,m in enumerate(block['metadata']) if m['world']==world and m['query']==query and m['recipient_hop']==hop) for hop in range(4)]
                    audits.append(M.correspondence(read,block['binding_reads']['0'][indices]))
                    centered=messages-messages.mean(-1,keepdim=True)
                    mr,mt,ms,best=rank_one_error(centered.flatten(-2));pr,pt,ps,_=rank_one_error(p)
                    for head in range(4):
                        denom=max(float(mt[head])**.5,zero_floor);relative=float(mr[head])**.5/denom
                        totals[head]['message_residual_sq']+=float(mr[head]);totals[head]['message_total_sq']+=float(mt[head])
                        totals[head]['score_residual_sq']+=float(pr[head]);totals[head]['score_total_sq']+=float(pt[head]);totals[head]['cases']+=1
                        totals[head]['cases_above_1pct']+=int(relative>.01)
                        records.append({'population':pop,'world':world,'query':query,'head':head,
                            'message_ray_relative_lower_bound':relative,'message_total_norm':float(mt[head])**.5,
                            'score_ray_relative_lower_bound':float(pr[head])**.5/max(float(pt[head])**.5,zero_floor),
                            'message_singular_values':ms[head].tolist(),'score_singular_values':ps[head].tolist(),
                            'near_zero':float(mt[head])**.5<zero_floor})
            for head,t in totals.items():
                t['message_relative_lower_bound']=(t['message_residual_sq']/max(t['message_total_sq'],zero_floor**2))**.5
                t['score_relative_lower_bound']=(t['score_residual_sq']/max(t['score_total_sq'],zero_floor**2))**.5
                t['one_shared_address_not_ruled_out']=t['message_relative_lower_bound']<=.01
            summary[pop]=totals
    result={'scope':'opened necessary lower bound for a nativehead-specific common address acrosshops; no rank sweep or adopted approximation',
        'passed_mechanics':all(a['passed'] for a in audits),'oracle_max_abs':max(a['max_abs'] for a in audits),'oracle_max_relative_rms':max(a['relative_rms'] for a in audits),
        'controls':{'exact_rank1':True,'identity_rank3_live':True},'populations':summary,'cases':records,
        'independent_constants':program.independent_constant_count(),'native_coefficients_removed':0,
        'input_rows_sha256':hashlib.sha256(ROWS.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'wall_seconds':time.perf_counter()-started}
    OUT.write_text(json.dumps(result,indent=2)+'\n');assert result['passed_mechanics'];print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))


if __name__=='__main__':main()
