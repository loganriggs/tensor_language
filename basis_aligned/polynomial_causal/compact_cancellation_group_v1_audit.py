"""Construct real products for a fixed cancellation group's leading output.

Uses the existing signed-eigenvalue pairing result; not a joint CP optimum.
A replays<=1e-8; B four-product groupcapture>=.95;
C full error increase<=.0005 and <=half original group floats.
"""
import hashlib,json,time
from pathlib import Path
import torch
from cancellation_group_v1_audit import load
from native_support_exchange_v1_audit import P,CK
from structured_branch_amplitudes_v1 import inner
from chunked_bilinear_coefficient_v1 import dense


def main():
    torch.set_grad_enabled(False);torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    out=P/'COMPACT_CANCELLATION_GROUP_V1_AUDIT.json';assert not out.exists();start=time.perf_counter()
    prior=json.loads((P/'CANCELLATION_GROUP_V1_AUDIT.json').read_text());ids=prior['rows'][-1]['products']
    metric=torch.load('/dev/shm/bilin18_native_product_energy_v1.pt',weights_only=True,map_location='cpu')['unembedding_gram'].double()
    wh=torch.linalg.cholesky(metric).T;current=load(prior['current_source'],wh)
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    na,nb,nd=[sd[f'transformer.h.17.mlp.{k}.weight'].double() for k in ('Left','Right','Down')]
    native=(na,nb,wh@nd);total=json.loads((P/'NATIVE_READER_METRIC_V1_AUDIT.json').read_text())['native_total']
    a,b,w=current[0][ids],current[1][ids],current[2][:,ids];group=(a,b,w)
    ib=torch.linalg.qr(torch.cat((a,b)).T,mode='reduced').Q
    ob=torch.linalg.qr(w,mode='reduced').Q;core=dense(a@ib,b@ib,ob.T@w)
    u,s,vh=torch.linalg.svd(core.reshape(16,-1),full_matrices=False)
    q=(s[0]*vh[0]).reshape(32,32);q=(q+q.T)/2
    ev,v=torch.linalg.eigh(q);pos=ev.clamp_min(0).flip(0);neg=(-ev).clamp_min(0)
    writer=ob@u[:,0];physical_writer=torch.linalg.solve_triangular(wh,writer[:,None],upper=True).flatten()
    rows=[];programs=[];worst=0.;oldfloats=32*128+16*1152;group_energy=inner(group,group)
    for k in (1,2,4,8,16):
        pp=v[:,-k:].flip(1)*pos[:k].sqrt();nn=v[:,:k]*neg[:k].sqrt()
        pl=(pp+nn).T@ib.T;pr=(pp-nn).T@ib.T;pw=writer[:,None].expand(-1,k)
        proposal=(pl,pr,pw)
        error=inner(group,group)+inner(proposal,proposal)-2*inner(group,proposal)
        capture=1-float(error/group_energy)
        curve=float((pos[:k].square().sum()+neg[:k].square().sum())/core.square().sum())
        curve_replay=abs(capture-curve)
        delta=(torch.cat((pl,a)),torch.cat((pr,b)),torch.cat((pw,-w),dim=1))
        increase=float((inner(delta,delta)+2*inner(current,delta)-2*inner(native,delta))/total)
        x=torch.randn(8,1152,generator=torch.Generator().manual_seed(892))
        compact=(((x@pl.T)*(x@pr.T)).sum(1)[:,None]*physical_writer[None,:])@wh.T
        cp=((x@pl.T)*(x@pr.T))@pw.T
        execution=float((compact-cp).norm()/cp.norm().clamp_min(1e-30))
        floatprice=2*k*1152+1152
        rows.append(dict(products=k,group_capture=capture,full_error_increase=increase,
            final_full_capture=1-prior['current_source']['loss']-increase,
            scalar_curve_replay=curve_replay,executor_replay=execution,
            group_float_coefficients=floatprice,old_group_float_coefficients=oldfloats,
            original_group_index_entries=32*128,new_group_index_entries=0))
        programs.append(dict(left=pl,right=pr,shared_writer=physical_writer))
        worst=max(worst,curve_replay,execution)
        if k==4:
            keep=[j for j in range(4608) if j not in ids]
            whole=(torch.cat((current[0][keep],pl)),torch.cat((current[1][keep],pr)),torch.cat((current[2][:,keep],pw),dim=1))
            full_loss=float((inner(whole,whole)-2*inner(native,whole)+total)/total)
            full_replay=abs(full_loss-prior['current_source']['loss']-increase);worst=max(worst,full_replay)
    four=next(row for row in rows if row['products']==4)
    cache=Path('/dev/shm/bilin18_compact_cancellation_group_v1.pt');assert not cache.exists()
    torch.save(dict(programs=programs,source=prior['current_source'],removed_products=ids),cache)
    result=dict(predictions=dict(pred_a_instrument=worst<=1e-8,pred_b_four_capture=four['group_capture']>=.95,
        pred_c_four_replacement=four['full_error_increase']<=.0005 and four['group_float_coefficients']<=oldfloats/2),
        rows=rows,whole_cp_replay=full_replay,source=prior['current_source'],removed_products=ids,
        cache=dict(path=str(cache),sha256=hashlib.sha256(cache.read_bytes()).hexdigest()),seconds=time.perf_counter()-start,
        scope='Fixed selected-group approximation, shared output and dense readers charged; other sparse '
              'program features/U/bias/background retained. No native body validation or stable semantic unit.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
