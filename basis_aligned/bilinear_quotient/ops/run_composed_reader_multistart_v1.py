#!/usr/bin/env python3
# BQGATE:0prefixes;1000evaluations;1800seconds.
"""pred_a at least one fit <=10% relative coefficient error.
pred_b all ten final unit-reader gradient RMS <=1e-7.
pred_c best FP32 normalized loss replay <=1e-6 absolute.
Price10starts,60iterations/100evaluations each,1800seconds alarm,oneGPU.
Two ranked supports times five reader perturbations; fixed seeds.
This is a local-recovery comparison, not a global guarantee or native circuit.
"""
import os,sys,json,time,signal
from pathlib import Path
from hashlib import sha256
import torch
P=Path(__file__).resolve().parents[2]/'polynomial_causal';sys.path.insert(0,str(P))
from head17_source_interface_v1 import CHECKPOINT
from fixed_writer_products_v1 import prepare_global
from symmetric_product_varpro_v1 import gram,normalized,solve

def main():
    binding=json.loads((P/'COMPOSED_READER_MULTISTART_V1_BINDING.json').read_text())
    assert all(sha256(Path(k).read_bytes()).hexdigest()==v for k,v in binding['files'].items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print('0prefixes;1000evaluations;1800seconds');return
    out=P/'COMPOSED_READER_MULTISTART_V1_RESULT.json';assert not out.exists()
    signal.alarm(1800);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    start=time.perf_counter()
    sd=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu')
    def maps(layer):return [sd[f'transformer.h.{layer}.mlp.{k}.weight'].to(device='cuda',dtype=torch.float64) for k in ('Left','Right','Down')]
    tl,tr,d=maps(9);l10,r10,d10=maps(10)
    w=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)['direction'].to(device='cuda',dtype=torch.float64)*float(sd['transformer.h.10.lambdas'][0])
    tw=(prepare_global(w,l10,r10,d10,True)['mixed_map']@d).T
    h=gram(tl,tr,tl,tr);energy=(tw*(h@tw)).sum()
    order=(tw.square().sum(1)*h.diag()).argsort(descending=True)[:4492]
    upstream_order=(d.square().sum(0)*h.diag()).argsort(descending=True)[:4492]
    del h,l10,r10,d10,d,sd
    arms=[];best_loss=float('inf');best_artifact=None
    for arm in range(10):
        order_i=order if arm<5 else upstream_order
        sigma=(0.,.01,.03,.1,.3)[arm%5]
        torch.manual_seed(7131230+arm)
        l=normalized(normalized(tl[order_i])+sigma*torch.randn_like(tl[order_i])/1152**.5).detach().requires_grad_()
        r=normalized(normalized(tr[order_i])+sigma*torch.randn_like(tr[order_i])/1152**.5).detach().requires_grad_()
        initial=float(solve(l,r,tl,tr,tw,energy)[0].detach());tic=time.perf_counter()
        optimizer=torch.optim.LBFGS([l,r],lr=1,max_iter=60,max_eval=100,
            tolerance_grad=1e-12,tolerance_change=1e-12,history_size=8,line_search_fn='strong_wolfe')
        evaluations=[]
        def closure():
            optimizer.zero_grad();loss=solve(l,r,tl,tr,tw,energy)[0];loss.backward()
            evaluations.append(float(loss.detach()));return loss
        optimizer.step(closure)
        ul=normalized(l.detach()).requires_grad_();ur=normalized(r.detach()).requires_grad_()
        loss,writers,_,_=solve(ul,ur,tl,tr,tw,energy);gl,gr=torch.autograd.grad(loss,(ul,ur))
        final=float(loss.detach());gradient=float(((gl.square().sum(1)+gr.square().sum(1)).mean()/2).sqrt())
        row=dict(arm=arm,support='composed' if arm<5 else 'upstream',sigma=sigma,seed=7131230+arm,
            initial_loss=initial,final_loss=final,relative_error=max(final,0)**.5,unit_reader_gradient_rms=gradient,
            iterations=optimizer.state[l].get('n_iter'),evaluations=len(evaluations),seconds=time.perf_counter()-tic)
        arms.append(row);print(json.dumps(row),flush=True)
        (P/'COMPOSED_READER_MULTISTART_V1_PROGRESS.json').write_text(json.dumps({'terminal':False,'arms':arms},indent=2)+'\n')
        if final<best_loss:
            best_loss=final;best_arm=arm
            best_artifact={'left':ul.detach().float().cpu(),'right':ur.detach().float().cpu(),'writers':writers.detach().float().cpu()}
        del optimizer,l,r,ul,ur,gl,gr,writers,loss
    with torch.no_grad():
        ql=best_artifact['left'].cuda().double();qr=best_artifact['right'].cuda().double();qw=best_artifact['writers'].cuda().double()
        qloss=float((energy+(qw*(gram(ql,qr,ql,qr)@qw)).sum()-2*(qw*(gram(ql,qr,tl,tr)@tw)).sum())/energy)
    path=P/'COMPOSED_READER_MULTISTART_V1_PROGRAM.pt';assert not path.exists();torch.save(best_artifact,path)
    result={'pred_a':best_loss<=.01,'pred_b':all(a['unit_reader_gradient_rms']<=1e-7 for a in arms),
        'pred_c':abs(qloss-best_loss)<=1e-6,'arms':arms,'best_arm':best_arm,'best_loss':best_loss,
        'fp32_artifact_loss':qloss,'artifact_sha256':sha256(path.read_bytes()).hexdigest(),
        'artifact_bytes':path.stat().st_size,'seconds':time.perf_counter()-start,'scope':__doc__}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='arms'},indent=2),flush=True)
if __name__=='__main__':main()
