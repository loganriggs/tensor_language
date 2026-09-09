"""Four-corner lower bounds for any additively separated query-root response."""
import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import time
import torch
import field_intervention_metrics as M
BASE=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def center(x):return x-x.mean(-1,keepdim=True)
def rms(x):return float(x.square().mean().sqrt())


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    signal.alarm(180);torch.set_num_threads(2);started=time.perf_counter()
    source=BASE/'ENDPOINT_MIXED_QUERY_INPUTS_V1_ROWS.pt';fullsource=BASE/'ENDPOINT_ROUTE_QUERY_LINEAGE_V1_ROWS.pt'
    out=BASE/'QUERY_ROOT_ADDITIVITY_BOUND_V1_RESULT.json';assert not out.exists()
    assert digest(source)=='c4866f1202dcab9fed8236b983da8577c985b6f15dd7b787d5472537bfe42a6d'
    assert digest(fullsource)=='f790aa9c396b12e57d207f821be9217413255d1ac170b553999426163687b048'
    f=lambda a,b:a*b;g=lambda a,b:.5*a+.5*b-.25
    checks={'sharp_scalar_bound':max(abs(f(a,b)-g(a,b)) for a,b in itertools.product((0,1),repeat=2))==.25,
            'additive_zero_contrast':g(1,1)-g(1,0)-g(0,1)+g(0,0)==0}
    assert all(checks.values());data=torch.load(source,map_location='cpu',weights_only=True)
    previous=torch.load(fullsource,map_location='cpu',weights_only=True);results={};audits=[]
    for pop,block in data.items():
        prior=previous[pop];assert block['metadata']==prior['metadata'];meta=block['metadata'];groups={}
        for route,parts in block['mixed_query_terms'].items():
            audits.append(M.correspondence(sum(parts.values())+block['native_query_logits'],prior['routes'][route]['mixed']+prior['native_query_logits']))
            groups[route]={}
            for query in (0,1):
                groups[route][str(query)]={}
                for hop in range(4):
                    sel=torch.tensor([m['query']==query and m['hop']==hop for m in meta])
                    scale=rms(center(prior['routes'][route]['full'][sel]));den=max(scale,1e-6)
                    lh=rms(center(parts['task'][sel]));ld=rms(center(parts['document'][sel]))
                    bounds={}
                    for name,contrast in (('L_vs_HD',max(lh,ld)),('H_vs_LD',lh),('D_vs_LH',ld)):
                        relative=contrast/(4*den);slack=max(0.,contrast-4e-9)/(4*den)
                        bounds[name]={'absolute_rms_error_lower_bound':contrast/4,'relative_to_native_route_lower_bound':relative,
                            'contrast_slack_4e9_relative_bound':slack,'ruled_out_at_fixed_native_1pct':relative>.01,
                            'slack_robust_at_1pct':slack>.01}
                    groups[route][str(query)][str(hop)]={'n':int(sel.sum()),'native_route_rms':scale,'reference_floored':scale<1e-6,
                        'LH_contrast_rms':lh,'LD_contrast_rms':ld,'partitions':bounds}
        results[pop]=groups
    result={'scope':'Additive-separation lower bounds on fixed-gain binary Q-root intervention cube, not whole-model impossibility.',
        'controls':checks,'instrument_passed':all(a['passed'] for a in audits),'partition_max_abs':max(a['max_abs'] for a in audits),
        'numeric_status':'Real-arithmetic triangle inequality; FP64 evaluated RHS, not an interval certificate.',
        'populations':results,'source_sha256':digest(source),'full_source_sha256':digest(fullsource),
        'runner_sha256':digest(Path(__file__)),'prereg_sha256':digest(BASE/'QUERY_ROOT_ADDITIVITY_BOUND_V1_PREREGISTRATION.md'),
        'wall_seconds':time.perf_counter()-started}
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'instrument_passed':result['instrument_passed'],'partition_max_abs':result['partition_max_abs'],
        'wall_seconds':result['wall_seconds'],'forward_hop3':{p:{r:g['0']['3'] for r,g in gs.items()} for p,gs in results.items()}},indent=2))


if __name__=='__main__':main()
