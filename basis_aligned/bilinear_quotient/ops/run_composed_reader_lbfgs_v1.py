#!/usr/bin/env python3
# BQGATE:0prefixes;80evaluations;900seconds.
"""pred_a CPU loss replay <=1e-8 relative.
pred_b final relative coefficient error <=.1.
pred_c unit-reader gradient RMS <=1e-7.
pred_d FP32 artifact normalized loss change <=1e-6 absolute.
Price: one FP64 GPU, 40 L-BFGS iterations / 80 evaluations, 900s alarm.
Weights-only fit; no data, native suffix or circuit-adoption claim.
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
    binding=json.loads((P/'COMPOSED_READER_LBFGS_V1_BINDING.json').read_text())
    assert all(sha256(Path(k).read_bytes()).hexdigest()==v for k,v in binding['files'].items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print('0prefixes;80evaluations;900seconds');return
    out=P/'COMPOSED_READER_LBFGS_V1_RESULT.json';assert not out.exists()
    signal.alarm(900);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    start=time.perf_counter()
    sd=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu')
    def maps(layer):return [sd[f'transformer.h.{layer}.mlp.{k}.weight'].to(device='cuda',dtype=torch.float64) for k in ('Left','Right','Down')]
    tl,tr,d=maps(9);l10,r10,d10=maps(10)
    w=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)['direction'].to(device='cuda',dtype=torch.float64)*float(sd['transformer.h.10.lambdas'][0])
    tw=(prepare_global(w,l10,r10,d10,True)['mixed_map']@d).T
    h=gram(tl,tr,tl,tr);energy=(tw*(h@tw)).sum()
    order=(tw.square().sum(1)*h.diag()).argsort(descending=True)[:4492]
    del h,l10,r10,d10,d,sd
    l=normalized(tl[order]).detach().requires_grad_();r=normalized(tr[order]).detach().requires_grad_()
    initial=float(solve(l,r,tl,tr,tw,energy)[0].detach())
    cpu=json.loads((P/'COMPOSED_READER_DESCENT_V1_RESULT.json').read_text())['history'][0]['relative_squared_error']
    replay=abs(initial-cpu)/abs(cpu)
    assert replay<=1e-8,('CPU/GPU mismatch',replay)
    optimizer=torch.optim.LBFGS([l,r],lr=1,max_iter=40,max_eval=80,
        tolerance_grad=1e-12,tolerance_change=1e-12,history_size=8,line_search_fn='strong_wolfe')
    evaluations=[]
    def closure():
        tic=time.perf_counter();optimizer.zero_grad()
        loss=solve(l,r,tl,tr,tw,energy)[0];loss.backward()
        torch.cuda.synchronize()
        row=dict(evaluation=len(evaluations),loss=float(loss.detach()),seconds=time.perf_counter()-tic,
                 reader_norm_min=min(float(l.norm(dim=1).min()),float(r.norm(dim=1).min())),
                 reader_norm_max=max(float(l.norm(dim=1).max()),float(r.norm(dim=1).max())))
        evaluations.append(row)
        if len(evaluations)%5==0:print(json.dumps(row),flush=True)
        return loss
    optimizer.step(closure)
    # Re-express at unit coordinates before interpreting gradient convergence.
    ul=normalized(l.detach()).requires_grad_();ur=normalized(r.detach()).requires_grad_()
    loss,writers,_,_=solve(ul,ur,tl,tr,tw,energy)
    gl,gr=torch.autograd.grad(loss,(ul,ur));final=float(loss.detach())
    gradient=float(((gl.square().sum(1)+gr.square().sum(1)).mean()/2).sqrt())
    artifact={'left':ul.detach().float().cpu(),'right':ur.detach().float().cpu(),'writers':writers.detach().float().cpu()}
    with torch.no_grad():
        ql=artifact['left'].cuda().double();qr=artifact['right'].cuda().double();qw=artifact['writers'].cuda().double()
        qloss=float((energy+(qw*(gram(ql,qr,ql,qr)@qw)).sum()-2*(qw*(gram(ql,qr,tl,tr)@tw)).sum())/energy)
    path=P/'COMPOSED_READER_LBFGS_V1_PROGRAM.pt';assert not path.exists();torch.save(artifact,path)
    state=optimizer.state[l]
    result={'pred_a':replay<=1e-8,'pred_b':final<=.01,'pred_c':gradient<=1e-7,'pred_d':abs(qloss-final)<=1e-6,
        'initial_loss':initial,'final_loss':final,'relative_error':max(final,0)**.5,
        'cpu_initial_replay_relative_error':replay,'unit_reader_gradient_rms':gradient,
        'fp32_artifact_loss':qloss,'iterations':state.get('n_iter'),
        'function_evaluations':state.get('func_evals'),'evaluations':evaluations,
        'seconds':time.perf_counter()-start,'artifact_sha256':sha256(path.read_bytes()).hexdigest(),
        'artifact_bytes':path.stat().st_size,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),
        'scope':__doc__+' Budget termination is not convergence; unit-coordinate gradient predicate is separate.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='evaluations'},indent=2),flush=True)
if __name__=='__main__':main()
