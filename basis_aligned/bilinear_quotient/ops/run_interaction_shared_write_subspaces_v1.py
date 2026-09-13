#!/usr/bin/env python3
# BQGATE:10starts20steps;3promotions500steps240seconds;900seconds.
"""pred_a serialized coefficient/objective error<=1e-5, unitbasis error<=1e-5.
pred_b at least2promotions pass2consecutive FP64conditional gain<=1e-8.
pred_c best coefficient error<=.1 and artifact bytes<=4896936.
PriceK32r8,147456edges,4878336nominalbytes,10starts/top3,900seconds.
Null: shared output subspaces cannot preserve coefficients at this budget,
or optimizer remains unfinished; no global recovery or native circuit claim.
"""
import json,os,sys,time,signal
from pathlib import Path
from hashlib import sha256
import torch
P=Path(__file__).resolve().parents[2]/'polynomial_causal';sys.path.insert(0,str(P))
from head17_output_block_objective_v1 import build


def assign(x,q):
    k,width,r=q.shape
    score=(x@q.permute(1,0,2).reshape(width,k*r)).reshape(x.shape[0],k,r)
    labels=score.square().sum(-1).argmax(-1)
    codes=score[torch.arange(x.shape[0],device=x.device),labels]
    fit=torch.einsum('ndr,nr->nd',q[labels],codes)
    return labels,codes,float((x-fit).square().sum()/x.square().sum())


def update(outer,labels,q):
    k,width,r=q.shape
    gram=outer.new_zeros(k,width*width).index_add_(0,labels,outer).reshape(k,width,width)
    eig,vectors=torch.linalg.eigh(gram)
    gain=eig[:,-r:].sum()-torch.einsum('kdr,kde,ker->',q,gram,q)
    occupied=torch.bincount(labels,minlength=k)>0
    return torch.where(occupied[:,None,None],vectors[:,:,-r:],q),float(gain)


@torch.no_grad()
def main():
    files=json.loads((P/'INTERACTION_SHARED_WRITE_SUBSPACES_V1_BINDING.json').read_text())['files']
    assert all(sha256(Path(k).read_bytes()).hexdigest()==v for k,v in files.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print('K32r8;10x20screen;top3x500or240sec;900sec');return
    assert not (P/'INTERACTION_SHARED_WRITE_SUBSPACES_V1_RESULT.json').exists()
    signal.alarm(900);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter()
    t,ids=build();xd=t.permute(1,2,0).reshape(-1,12).contiguous().cuda();x=xd.float()
    outer=(x[:,:,None]*x[:,None,:]).flatten(1);total=float(x.square().sum())
    screened=[];states=[]
    for seed in range(61461,61471):
        torch.manual_seed(seed)
        q=torch.linalg.qr(torch.randn(32,12,8,device='cuda'),mode='reduced')[0]
        history=[]
        for step in range(21):
            labels,codes,loss=assign(x,q);history.append(loss)
            if step<20:q,_=update(outer,labels,q)
        screened.append(dict(seed=seed,loss=loss,history=history));states.append(q)
    order=sorted(range(10),key=lambda i:screened[i]['loss'])[:3]
    promoted=[];best=None;outerd=(xd[:,:,None]*xd[:,None,:]).flatten(1);totald=float(xd.square().sum())
    for index in order:
        q=states[index];tic=time.perf_counter();history=[];stop='step_limit'
        for step in range(501):
            labels,codes,loss=assign(x,q);history.append(dict(step=step,loss=loss))
            if len(history)>1:assert loss<=history[-2]['loss']+2e-6
            if len(history)>10 and abs(history[-11]['loss']-loss)<=1e-7*max(loss,1e-30):stop='plateau_for_polish';break
            if time.perf_counter()-tic>=240:stop='time_limit';break
            if step<500:q,_=update(outer,labels,q)
        q=torch.linalg.qr(q.double(),mode='reduced')[0];polish=[];consecutive=0
        for step in range(21):
            labels,codes,loss=assign(xd,q);new,gain=update(outerd,labels,q)
            gain=max(0.,gain/totald);consecutive=consecutive+1 if gain<=1e-8 else 0
            polish.append(dict(step=step,loss=loss,conditional_gain=gain))
            if consecutive>=2:break
            if step<20:q=new
        item=dict(seed=screened[index]['seed'],history=history,fp32_stop=stop,polish=polish,
                  coordinate_stationary=consecutive>=2,relative_error=loss**.5,seconds=time.perf_counter()-tic)
        promoted.append(item)
        if best is None or loss<best['loss']:best=dict(loss=loss,q=q.cpu(),labels=labels.cpu(),codes=codes.cpu())
        print('promoted',item['seed'],item['relative_error'],item['coordinate_stationary'],flush=True)
    program=dict(output_bases=best['q'].float(),groups=best['labels'].to(torch.uint8),codes=best['codes'].float(),token_ids=ids,input_shape=[1152,128])
    path=P/'INTERACTION_SHARED_WRITE_SUBSPACES_V1_PROGRAM.pt';torch.save(program,path)
    q=program['output_bases'].double();labels=program['groups'].long();codes=program['codes'].double()
    reconstructed=torch.einsum('ndr,nr->nd',q[labels],codes)
    exact_loss=float((xd.cpu()-reconstructed).square().sum()/xd.cpu().square().sum())
    orth=float((q.transpose(1,2)@q-torch.eye(8)).abs().max())
    result={'pred_a':abs(exact_loss-best['loss'])<=1e-5 and orth<=1e-5,
            'pred_b':sum(p['coordinate_stationary'] for p in promoted)>=2,
            'pred_c':exact_loss**.5<=.1 and path.stat().st_size<=4896936,
            'screening':screened,'promoted':promoted,'serialized_relative_error':exact_loss**.5,
            'serialized_objective_difference':abs(exact_loss-best['loss']),'orthogonality_error':orth,
            'artifact_bytes':path.stat().st_size,'artifact_sha256':sha256(path.read_bytes()).hexdigest(),
            'seconds':time.perf_counter()-start,'scope':'Weight-only K32r8 output subspace groups, full coordinate input products. FP64conditional gain certifies coordinate tolerance only, not global topology recovery or semantic stability. No native behavior/runtime/adoption.'}
    (P/'INTERACTION_SHARED_WRITE_SUBSPACES_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('screening','promoted')},indent=2))


if __name__=='__main__':main()
