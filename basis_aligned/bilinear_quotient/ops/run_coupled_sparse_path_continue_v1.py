#!/usr/bin/env python3
# BQGATE:0bodyforwards,0sequences; same weight objective, up to4x1200fitseconds.
"""pred_a native replays, pred_b all support/gradient convergence, pred_c joint>=1.10independent,
pred_d active-edge gradient>=1.5xspeed eachjointseed. Same hypothesis/budget; no text.
"""
import os,sys,time,json,hashlib,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from coupled_sparse_path_v1 import evaluate as full_evaluate
from coupled_sparse_path_v2 import evaluate as active_evaluate,fit_fixed
STEM='COUPLED_SPARSE_PATH_CONTINUE_V1'


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    prior=json.loads((P/'COUPLED_SPARSE_PATH_PILOT_V2_RESULT.json').read_text());assert prior['pred_a']
    artifact=P/'COUPLED_SPARSE_PATH_PILOT_V2_PROGRAMS.pt';assert digest(artifact)==prior['artifact_sha256']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,body_forwards=0,text_sequences=0,max_fit_seconds=4800,continuation=True,fit_floats=147456)));return
    out=P/(STEM+'_RESULT.json');ap=P/(STEM+'_PROGRAMS.pt');assert not out.exists() and not ap.exists();signal.alarm(5200)
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False;started=time.perf_counter()
    state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
    u=state['lm_head.weight'].double().cuda();root=torch.linalg.cholesky(u.T@u).T;del u
    l,r,d=[state['transformer.h.17.mlp.'+n+'.weight'].double().cuda() for n in ('Left','Right','Down')];writer=root@d
    o=state['transformer.h.17.attn.c_proj.weight'].double().cuda();scale=o.norm()/1152**.5
    left=torch.stack([l,l@o/scale]);right=torch.stack([r,r@o/scale]);total=torch.tensor(prior['total_coefficient_norm2'],device='cuda')
    saved=torch.load(artifact,weights_only=True,map_location='cpu')['programs'];reports=[];programs=[];errors=[];graderrors=[];ortherrors=[];selection=[]
    for program in saved:
        mode,seed=program['mode'],program['seed'];bank=program['bank'].cuda();support=program['support'].cuda()
        old=next(r for r in prior['reports'] if r['seed']==seed and r['mode']==mode)
        assert bank.numel()+program['physical_writer'].numel()==147456
        initial,detail=active_evaluate(left,right,writer,bank,mode,total,96,support)
        errors.append(abs(-float(initial)/old['capture']-1));errors.append(abs(float(scale)/program['source_scale']-1))
        expected=root@program['physical_writer'].cuda();errors.append(float((detail['selected_coefficients']-expected).norm()/expected.norm()))
        def vg(fun):
            x=bank.detach().requires_grad_();loss,_=fun(left,right,writer,x,mode,total,96,support);g=torch.autograd.grad(loss,x)[0]
            return float(loss.detach()),g.detach()
        oldloss,og=vg(full_evaluate);newloss,ng=vg(active_evaluate);ge=float((og-ng).norm()/og.norm());graderrors.append(ge);errors.append(abs(oldloss-newloss)/abs(oldloss))
        timings={}
        for name,fun in [('full',full_evaluate),('active',active_evaluate)]:
            vg(fun);torch.cuda.synchronize();tic=time.perf_counter()
            for _ in range(3):vg(fun)
            torch.cuda.synchronize();timings[name]=(time.perf_counter()-tic)/3
        assert max(errors)<=1e-8 and ge<=1e-8
        history=[];stable=0
        for cycle in range(20):
            _,allinfo=full_evaluate(left,right,writer,bank,mode,total,96);support=allinfo['support'].detach()
            bank,report=fit_fixed(left,right,writer,bank,mode,total,support,seconds=60,tolerance=1e-7)
            _,new=full_evaluate(left,right,writer,bank,mode,total,96);new_support=new['support'].detach()
            unchanged=bool(torch.equal(support.sort().values,new_support.sort().values));stable=stable+1 if unchanged else 0
            selection.append(max(0.,report['capture']-float(new['capture'])));report.update(cycle=cycle,support_unchanged=unchanged,consecutive_stable=stable,best_support_gap=float(new['capture'])-report['capture'])
            history.append(report)
            print(json.dumps(dict(seed=seed,mode=mode,**{k:v for k,v in report.items() if k!='history'})),flush=True)
            (P/(STEM+'_PROGRESS.json')).write_text(json.dumps(dict(completed=reports,current=dict(seed=seed,mode=mode,cycles=history)),indent=2)+'\n')
            if stable>=2 and report['tangent_norm']<=1e-7 and report['relative_stationarity']<=1e-4:break
        _,final=active_evaluate(left,right,writer,bank,mode,total,96,support)
        orth=float((bank.transpose(-1,-2)@bank-torch.eye(bank.shape[-1],device='cuda')).abs().max());ortherrors.append(orth)
        entry=dict(mode=mode,seed=seed,initial_capture=old['capture'],capture=float(final['capture']),
            converged=stable>=2 and report['tangent_norm']<=1e-7 and report['relative_stationarity']<=1e-4,
            best_support_gap=report['best_support_gap'],timings=timings,speedup=timings['full']/timings['active'],cycles=history,orthogonality_error=orth)
        reports.append(entry)
        physical=torch.linalg.solve_triangular(root,final['selected_coefficients'],upper=True)
        assert torch.isfinite(physical).all() and torch.isfinite(bank).all()
        programs.append(dict(seed=seed,mode=mode,bank=bank.detach().cpu(),support=support.cpu(),physical_writer=physical.detach().cpu(),source_scale=float(scale)))
        torch.save(dict(programs=programs,complete=len(programs)==4),ap)
    pairs=[{r['mode']:r for r in reports if r['seed']==seed} for seed in (11241,11242)]
    result={'pred_a':max(errors)<=1e-8 and max(graderrors)<=1e-8 and max(ortherrors)<=1e-9 and max(selection)<=1e-10,
        'pred_b':all(r['converged'] for r in reports),'pred_c':all(pair['joint']['capture']>=1.10*pair['independent']['capture'] for pair in pairs),
        'pred_d':all(r['speedup']>=1.5 for r in reports if r['mode']=='joint'),'reports':reports,'initial_replay_errors':errors,'gradient_errors':graderrors,
        'artifact_sha256':digest(ap),'artifact_bytes':ap.stat().st_size,'source_pilot_sha256':digest(P/'COUPLED_SPARSE_PATH_PILOT_V2_RESULT.json'),
        'wall_seconds':time.perf_counter()-started,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
        'scope':'Longer support/reader optimization, unchanged full-U formal path objective and budgets. Original pilot misses preserved; no standalone extraction or circuit claim.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='reports'},indent=2),flush=True);assert result['pred_a']

if __name__=='__main__':main()
