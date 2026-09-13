#!/usr/bin/env python3
# BQGATE:10starts20steps;top3x5000steps120seconds;420seconds.
"""pred_a serialized balanced squared loss replay<=1e-5; monotone<=1e-10.
pred_b at least two promoted fits pass two conditional gains<=1e-8.
pred_c original coefficient error<=.1 and serialized bytes<=4896936.
Price K32/r8,4878336 nominal bytes,10starts/top3,420second cap.
Null: rebalanced geometry trades common-component fidelity for differences
without meeting the unchanged total-error or subsequent native own-effect bars.
No text fit; existing six token pairs define weights-only output metric.
"""
import json,os,sys,time,signal
from pathlib import Path
from hashlib import sha256
import torch
P=Path(__file__).resolve().parents[2]/'polynomial_causal';sys.path.insert(0,str(P))
from head17_output_block_objective_v1 import build
from run_interaction_shared_write_subspaces_v1 import assign,update

@torch.no_grad()
def main():
    files=json.loads((P/'INTERACTION_BALANCED_SUBSPACES_V1_BINDING.json').read_text())['files']
    assert all(sha256(Path(k).read_bytes()).hexdigest()==v for k,v in files.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print('K32r8;10starts20steps;top3x5000steps120seconds;420seconds');return
    assert not (P/'INTERACTION_BALANCED_SUBSPACES_V1_RESULT.json').exists()
    signal.alarm(420);torch.set_num_threads(2);start=time.perf_counter()
    t,ids=build();x=t.permute(1,2,0).reshape(-1,12).contiguous().cuda()
    plus=torch.kron(torch.eye(6,dtype=x.dtype,device=x.device),torch.ones(2,2,dtype=x.dtype,device=x.device)/2)
    minus=torch.eye(12,dtype=x.dtype,device=x.device)-plus
    ep,em=(x@plus).square().sum(),(x@minus).square().sum();gamma=(ep/em).sqrt()
    w=plus+gamma*minus;wi=plus+minus/gamma;y=x@w
    total=float(y.square().sum());outer=(y[:,:,None]*y[:,None,:]).flatten(1)
    prior=torch.load(P/'INTERACTION_SHARED_WRITE_POLISH_V1_PROGRAM.pt',weights_only=True)
    assert prior['token_ids']==ids
    screened=[];states=[]
    for seed in range(61501,61511):
        torch.manual_seed(seed)
        q=torch.linalg.qr(w@prior['output_bases'].double().cuda() if seed==61501 else torch.randn(32,12,8,dtype=y.dtype,device=y.device),mode='reduced')[0]
        for step in range(21):
            labels,codes,loss=assign(y,q)
            if step<20:q,_=update(outer,labels,q)
        screened.append(dict(seed=seed,loss=loss,initialization='transformed_prior' if seed==61501 else 'random'));states.append(q)
    promoted=[];best=None
    for index in sorted(range(10),key=lambda i:screened[i]['loss'])[:3]:
        q=states[index];tic=time.perf_counter();history=[];consecutive=0;stop='step_limit'
        for step in range(5001):
            labels,codes,loss=assign(y,q);new,gain=update(outer,labels,q);gain=max(0.,gain/total)
            if history:assert loss<=history[-1]['loss']+1e-10
            history.append(dict(step=step,loss=loss,conditional_gain=gain))
            consecutive=consecutive+1 if gain<=1e-8 else 0
            if consecutive>=2:stop='coordinate_tolerance';break
            if time.perf_counter()-tic>=120:stop='time_limit';break
            if step<5000:q=new
        promoted.append(dict(seed=screened[index]['seed'],stop=stop,history=history,coordinate_stationary=consecutive>=2,seconds=time.perf_counter()-tic))
        if best is None or loss<best['loss']:best=dict(loss=loss,q=q.clone(),labels=labels.clone(),codes=codes.clone())
        print('promotion',screened[index]['seed'],stop,loss**.5,flush=True)
    program=dict(output_bases=(wi@best['q']).cpu().float(),groups=best['labels'].cpu().to(torch.uint8),codes=best['codes'].cpu().float(),token_ids=ids,input_shape=[1152,128])
    path=P/'INTERACTION_BALANCED_SUBSPACES_V1_PROGRAM.pt';torch.save(program,path)
    fit=torch.einsum('ndr,nr->nd',program['output_bases'].double().cuda()[program['groups'].long().cuda()],program['codes'].double().cuda())
    error=fit-x;balanced=float((error@w).square().sum()/total);original=float(error.norm()/x.norm())
    result={'pred_a':abs(balanced-best['loss'])<=1e-5,'pred_b':sum(p['coordinate_stationary'] for p in promoted)>=2,
        'pred_c':original<=.1 and path.stat().st_size<=4896936,'screened':screened,'promoted':promoted,
        'relative_error':original,'balanced_relative_error':balanced**.5,'sum_relative_error':float((error@plus).norm()/ep.sqrt()),
        'difference_relative_error':float((error@minus).norm()/em.sqrt()),'gamma':float(gamma),'artifact_bytes':path.stat().st_size,
        'artifact_sha256':sha256(path.read_bytes()).hexdigest(),'seconds':time.perf_counter()-start,
        'scope':'Weights-only pair-balanced K32/r8, original error bar unchanged; decoder includes inverse metric at no extra runtime storage. Coordinate stationarity is not global recovery, behavioral fidelity or identified reuse.'}
    (P/'INTERACTION_BALANCED_SUBSPACES_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('screened','promoted')},indent=2))

if __name__=='__main__':main()
