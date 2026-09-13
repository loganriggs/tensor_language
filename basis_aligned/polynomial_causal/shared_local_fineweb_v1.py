"""Frozen weight-fit diagnostics on two output routes; historical FineWeb only."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
import torch.nn.functional as functional
from audit_fullu_output_functions_v1 import CK
from joint_quadratic_fit_v1 import product_cross

P=Path(__file__).resolve().parent
EPS=torch.finfo(torch.float32).eps


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def apply(program,states):
    result=(states@program['global_reader'].double().T)@program['global_codes'].double().T
    result+=(states@program['mean'].double())[:,None]
    for k,bank in enumerate(program['local_readers']):
        ix=program['labels']==k
        result[:,ix]+=(states@bank.double().T)@program['local_codes'][ix].double().T
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--g',type=int,default=64,choices=[64,128]);args=parser.parse_args()
    torch.set_num_threads(2);torch.set_grad_enabled(False);started=time.perf_counter()
    cache=Path('/dev/shm/bilin18_frozen_radial_fineweb_v1.pt')
    cache_sha=digest(cache);assert cache_sha=='b1107a91d55df69bc05eefade10ffb362a5c1a81c783e7dfa3c746ea2dec72de'
    data=torch.load(cache,map_location='cpu',weights_only=True);positions=[63,127]
    def port(key):return data['ports'][key][:,positions].reshape(-1,1152)
    x=port('input').double();pre=port('pre').float();output=port('native_output').float()
    targets=data['rows'][:,[p+1 for p in positions]].reshape(-1).long();assert len(targets)==128
    sd=torch.load(CK,map_location='cpu',weights_only=True,mmap=True)
    u=sd['lm_head.weight'].double()
    l,r,d=[sd[f'transformer.h.17.mlp.{key}.weight'].double() for key in ['Left','Right','Down']]
    bias=sd['transformer.h.17.mlp.Down_bias'].double();branch=output.double()-bias
    h=(pre+output).double();rho=(h.square().mean(1,keepdim=True)+EPS).sqrt()
    raw=(h/rho)@u.T
    def softcap(v):return 30*torch.tanh(v/30)
    reference=softcap(raw);log=reference.log_softmax(-1);prob=log.exp()
    ce=-log.gather(1,targets[:,None]).squeeze(1)
    native32=softcap(functional.rms_norm(h.float(),(1152,),eps=EPS)@u.float().T)
    replay=float((native32.double()-reference).norm()/reference.norm())
    ce_replay=abs(float(functional.cross_entropy(native32.double(),targets)-ce.mean()))
    exact=((x@l.T)*(x@r.T))@d.T
    branch_replay=float(((exact-branch)@u.T).norm()/(branch@u.T).norm())
    assert replay<=1e-5 and ce_replay<=1e-4 and branch_replay<=1e-5
    path=P/f'FULLU_SHARED_LOCAL_FIT_V1_G{args.g}_PROGRAM.pt'
    fit=json.loads((P/f'FULLU_SHARED_LOCAL_FIT_V1_G{args.g}_RESULT.json').read_text())
    grouped=torch.load(path,map_location='cpu',weights_only=True)
    metric=d@product_cross(l,r,l,r)@d.T;root=torch.linalg.cholesky((metric+metric.T)/2)
    mean=u.mean(0);coordinates=(u-mean)@root
    _,q=torch.linalg.eigh(coordinates.T@coordinates)
    q=q.flip(1)[:,:fit['matched_global_rank']]
    global_program=dict(mean=mean,global_codes=coordinates@q,
        global_reader=torch.linalg.solve_triangular(root.T,q,upper=True).T,local_readers=[])
    rows=[];details={}
    for name,program in [('grouped',grouped),('matched_global',global_program)]:
        arms={'quadratic_route_only':raw+(apply(program,branch)-branch@u.T)/rho,
              'whole_unembedding':apply(program,h/rho)}
        for route,candidate_raw in arms.items():
            candidate=softcap(candidate_raw).log_softmax(-1)
            candidate_ce=-candidate.gather(1,targets[:,None]).squeeze(1)
            delta=candidate_ce-ce;kl=(prob*(log-candidate)).sum(1)
            key=name+'/'+route
            rows.append(dict(name=name,route=route,ce_added=float(delta.mean()),
                mean_absolute_position_ce_change=float(delta.abs().mean()),
                maximum_absolute_ce_change=float(delta.abs().max()),mean_kl=float(kl.mean())))
            details[key]=dict(ce_added=delta.tolist(),kl=kl.tolist())
    lookup={(r['name'],r['route']):r for r in rows};routes=['quadratic_route_only','whole_unembedding']
    result={'pred_a':True,
        'pred_b':all(lookup['grouped',route]['mean_kl']<=lookup['matched_global',route]['mean_kl'] for route in routes),
        'pred_c':all(lookup['grouped',route]['mean_kl']<=.05 and lookup['grouped',route]['mean_absolute_position_ce_change']<=.05 for route in routes),
        'global_width':args.g,'rows':rows,'per_position':details,'native_mean_ce':float(ce.mean()),
        'native_logit_precision_replay':replay,'ce_precision_replay':ce_replay,'branch_replay':branch_replay,
        'cache_sha256':cache_sha,'program_sha256':digest(path),'body_forwards':0,'fit_steps':0,
        'positions':positions,'targets':targets.tolist(),'wall_seconds':time.perf_counter()-started,
        'scope':'Historical128FineWebpositions; frozen readers, two output routes. No fresh/OOD or circuit identification.'}
    with (P/f'SHARED_LOCAL_FINEWEB_V1_G{args.g}_RESULT.json').open('x') as out:
        json.dump(result,out,indent=2);out.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['per_position','targets']},indent=2))


if __name__=='__main__':main()
