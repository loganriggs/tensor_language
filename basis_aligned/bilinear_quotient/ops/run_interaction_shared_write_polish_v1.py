#!/usr/bin/env python3
# BQGATE:savedK32r8;5000steps;180seconds.
"""pred_a objective nonincrease<=1e-10 and savedFP32 loss replay<=1e-5.
pred_b two consecutive exact conditional gains<=1e-8.
pred_c final coefficienterror<=.1 and artifact bytes<=4896936.
Price saved32x12x8basis,5000FP64steps/180seconds, no text.
Null: exact conditional iteration remains unfinished or converges above target.
"""
import os,sys,json,time,signal
from pathlib import Path
from hashlib import sha256
import torch
P=Path(__file__).resolve().parents[2]/'polynomial_causal';sys.path.insert(0,str(P));sys.path.insert(0,str(P.parent/'bilinear_quotient/ops'))
from head17_output_block_objective_v1 import build
from run_interaction_shared_write_subspaces_v1 import assign,update


@torch.no_grad()
def main():
    files=json.loads((P/'INTERACTION_SHARED_WRITE_POLISH_V1_BINDING.json').read_text())['files']
    assert all(sha256(Path(k).read_bytes()).hexdigest()==v for k,v in files.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('savedK32r8;5000FP64steps;180seconds');return
    assert not (P/'INTERACTION_SHARED_WRITE_POLISH_V1_RESULT.json').exists()
    signal.alarm(180);torch.set_num_threads(2);start=time.perf_counter()
    t,ids=build();x=t.permute(1,2,0).reshape(-1,12).contiguous().cuda();total=float(x.square().sum())
    outer=(x[:,:,None]*x[:,None,:]).flatten(1)
    prior=torch.load(P/'INTERACTION_SHARED_WRITE_SUBSPACES_V1_PROGRAM.pt',weights_only=True,map_location='cpu')
    q=torch.linalg.qr(prior['output_bases'].double().cuda(),mode='reduced')[0]
    history=[];consecutive=0;stop='step_limit'
    for step in range(5001):
        labels,codes,loss=assign(x,q);new,gain=update(outer,labels,q);gain=max(0.,gain/total)
        if history:assert loss<=history[-1]['loss']+1e-10
        history.append(dict(step=step,loss=loss,conditional_gain=gain))
        consecutive=consecutive+1 if gain<=1e-8 else 0
        if consecutive>=2:stop='coordinate_tolerance';break
        if time.perf_counter()-start>160:stop='time_limit';break
        if step<5000:q=new
    program=dict(output_bases=q.cpu().float(),groups=labels.cpu().to(torch.uint8),codes=codes.cpu().float(),token_ids=ids,input_shape=[1152,128])
    path=P/'INTERACTION_SHARED_WRITE_POLISH_V1_PROGRAM.pt';torch.save(program,path)
    fitted=torch.einsum('ndr,nr->nd',program['output_bases'].double()[program['groups'].long()],program['codes'].double())
    saved_loss=float((x.cpu()-fitted).square().sum()/total)
    result={'pred_a':abs(saved_loss-loss)<=1e-5,'pred_b':consecutive>=2,'pred_c':saved_loss**.5<=.1 and path.stat().st_size<=4896936,
            'history':history,'stop':stop,'relative_error':saved_loss**.5,'artifact_bytes':path.stat().st_size,
            'artifact_sha256':sha256(path.read_bytes()).hexdigest(),'seconds':time.perf_counter()-start,
            'scope':'Saved best K32r8 continuation, unchanged coefficient objective. FP64coordinate tolerance is not global or multi-start convergence. Original first-runB/Cmisses retained. No nativebehavior oradoption.'}
    (P/'INTERACTION_SHARED_WRITE_POLISH_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='history'},indent=2))


if __name__=='__main__':main()
