#!/usr/bin/env python3
# BQGATE: 0 body forwards, 0 sequences; weight-only4x240second fit pilot.
"""pred_a instrument, pred_b all convergence, pred_c joint>=1.10independent bothseeds,
pred_d >=8cross edges and>=4joint readers perport reused within/acrosssource.
Full preregistration COUPLED_SPARSE_PATH_PILOT_V1_PREREGISTRATION.md; no text.
"""
import os,sys,time,signal,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from coupled_sparse_path_v1 import coefficients,evaluate,fit_fixed
STEM='COUPLED_SPARSE_PATH_PILOT_V1'
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


@torch.no_grad()
def norm2(l,r,w,chunk=256):
    result=l.new_zeros(())
    for start in range(0,len(l),chunk):
        a,b=l[start:start+chunk],r[start:start+chunk]
        gram=((a@l.T)*(b@r.T)+(a@r.T)*(b@l.T))/2
        result+=((w[:,start:start+chunk].T@w)*gram).sum()
    return result


def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    controls=json.loads((P/'COUPLED_SPARSE_PATH_V1_CONTROL.json').read_text());assert controls['pred_a'] and controls['pred_b']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,body_forwards=0,sequences=0,corpus_access=False,arms=4,max_fit_seconds=960,fit_floats_per_arm=147456)));return
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_PROGRAMS.pt');assert not out.exists() and not artifact.exists();signal.alarm(1500)
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False;started=time.perf_counter()
    state=torch.load(CK,weights_only=True,mmap=True,map_location='cpu')
    u=state['lm_head.weight'].double().cuda();root=torch.linalg.cholesky(u.T@u).T;del u
    l,r,d=[state['transformer.h.17.mlp.'+n+'.weight'].double().cuda() for n in ('Left','Right','Down')]
    w=root@d;o=state['transformer.h.17.attn.c_proj.weight'].double().cuda();scale=o.norm()/1152**.5
    left=torch.stack([l,l@o/scale]);right=torch.stack([r,r@o/scale])
    original=norm2(l,r,w);bridge=abs(float(original)/99245061353.47293-1);assert bridge<=1e-8
    total=norm2(torch.cat(list(left),1),torch.cat(list(right),1),w);assert torch.isfinite(total) and total>0
    torch.manual_seed(11240);source=torch.randn(7,2,1152,device='cuda')
    direct=(source[:,0]+source[:,1]@o.T/scale)@l.T
    pulled=sum(source[:,i]@left[i].T for i in range(2));port_error=float((direct-pulled).norm()/direct.norm());assert port_error<=1e-10
    print(json.dumps(dict(setup_seconds=time.perf_counter()-started,native_bridge=bridge,port_error=port_error,source_scale=float(scale),total=float(total))),flush=True)
    reports=[];programs=[];fd_errors=[];orth_errors=[];selection_errors=[]
    for seed in (11241,11242):
        torch.manual_seed(seed)
        independent=[]
        for port in (0,0,1,1):
            sketch=left[port].T@torch.randn(4608,8,device='cuda')+right[port].T@torch.randn(4608,8,device='cuda')
            independent.append(torch.linalg.qr(sketch).Q)
        independent=torch.stack(independent)
        joint=torch.stack([torch.linalg.qr(torch.cat([independent[0],independent[1]],1)).Q,
                           torch.linalg.qr(torch.cat([independent[2],independent[3]],1)).Q])
        for mode,bank in [('joint',joint),('independent',independent)]:
            initial,info=evaluate(left,right,w,bank,mode,total,96);initial=-float(initial.detach());history=[];stable=0
            for cycle in range(4):
                _,info=evaluate(left,right,w,bank,mode,total,96);support=info['support'].detach()
                if cycle==0:
                    x=bank.detach().requires_grad_();loss,_=evaluate(left,right,w,x,mode,total,96,support);g=torch.autograd.grad(loss,x)[0]
                    delta=torch.randn_like(x);inner=x.transpose(-1,-2)@delta;delta=(delta-x@((inner+inner.transpose(-1,-2))/2)).detach();delta/=delta.norm()
                    eps=1e-4;vals=[]
                    for sign in (1,-1):
                        q,rr=torch.linalg.qr(bank+sign*eps*delta);q=q*torch.diagonal(rr,dim1=-2,dim2=-1).sign().unsqueeze(-2)
                        vals.append(evaluate(left,right,w,q,mode,total,96,support)[0].detach())
                    fd=(vals[0]-vals[1])/(2*eps);analytic=(g*delta).sum();err=float(abs(fd-analytic)/abs(analytic).clamp_min(1e-15));fd_errors.append(err);assert err<=1e-4
                    del x,g,delta,loss
                bank,report=fit_fixed(left,right,w,bank,mode,total,support,seconds=60,tolerance=1e-7)
                _,after=evaluate(left,right,w,bank,mode,total,96);new=after['support'].detach()
                unchanged=bool(torch.equal(new.sort().values,support.sort().values));stable=stable+1 if unchanged else 0
                selection_errors.append(max(0.,report['capture']-float(after['capture'])))
                report.update(cycle=cycle,support_unchanged=unchanged,consecutive_stable=stable,best_support_capture=float(after['capture']))
                history.append(report)
                print(json.dumps(dict(seed=seed,mode=mode,**{k:v for k,v in report.items() if k!='history'})),flush=True)
                (P/(STEM+'_PROGRESS.json')).write_text(json.dumps(dict(completed=reports,current=dict(seed=seed,mode=mode,cycles=history)),indent=2)+'\n')
                if stable>=2 and report['tangent_norm']<=1e-7 and report['relative_stationarity']<=1e-4:break
            # Final program uses last optimized fixed support. Report any remaining best-support gap.
            _,final=evaluate(left,right,w,bank,mode,total,96,support);capture=float(final['capture'])
            orth=float((bank.transpose(-1,-2)@bank-torch.eye(bank.shape[-1],device='cuda')).abs().max());orth_errors.append(orth)
            counts={};reuse=[]
            if mode=='joint':
                ij=torch.triu_indices(32,32,device='cuda')[:,support];self_nodes=[set(),set()];cross_nodes=[set(),set()];cross_count=0
                for i,j in ij.T.tolist():
                    if i<16<=j:cross_count+=1;cross_nodes[0].add(i);cross_nodes[1].add(j)
                    else:
                        port=int(i>=16);self_nodes[port].update([i,j])
                reuse=[len(self_nodes[i]&cross_nodes[i]) for i in range(2)];counts=dict(cross_edges=cross_count,reused_source_readers=reuse)
            entry=dict(seed=seed,mode=mode,initial_capture=initial,capture=capture,best_support_gap=float(after['capture'])-capture,
                converged=stable>=2 and report['tangent_norm']<=1e-7 and report['relative_stationarity']<=1e-4,
                orthogonality_error=orth,cycles=history,incidence=counts,fit_floats=bank.numel()+1152*96)
            reports.append(entry)
            physical=torch.linalg.solve_triangular(root,final['coefficients'][:,support],upper=True)
            programs.append(dict(seed=seed,mode=mode,bank=bank.detach().cpu(),support=support.cpu(),physical_writer=physical.detach().cpu(),source_scale=float(scale)))
    torch.save(dict(programs=programs,source_maps='[I,O17/source_scale], second source=source_scale*preOattention; input RMS/bias/direct residual/final tail native'),artifact)
    pairs=[{r['mode']:r for r in reports if r['seed']==seed} for seed in (11241,11242)]
    instrument=bridge<=1e-8 and port_error<=1e-10 and max(orth_errors)<=1e-9 and max(selection_errors)<=1e-10 and max(fd_errors)<=1e-4 and all(r['fit_floats']==147456 and torch.isfinite(torch.tensor(r['capture'])) for r in reports)
    result=dict(pred_a=instrument,pred_b=all(r['converged'] for r in reports),
        pred_c=all(pair['joint']['capture']>=1.10*pair['independent']['capture'] for pair in pairs),
        pred_d=all(r['incidence']['cross_edges']>=8 and min(r['incidence']['reused_source_readers'])>=4 for r in reports if r['mode']=='joint'),
        native_norm_bridge=bridge,port_error=port_error,finite_difference_errors=fd_errors,source_scale=float(scale),total_coefficient_norm2=float(total),
        reports=reports,artifact_sha256=digest(artifact),artifact_bytes=artifact.stat().st_size,wall_seconds=time.perf_counter()-started,
        max_gpu_memory_bytes=torch.cuda.max_memory_allocated(),scope='Weight-only formal two-source sparse-edge pilot, full unembedding metric. Native background/routing retained, no circuit/adoption or global optimization claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='reports'},indent=2),flush=True);assert instrument

if __name__=='__main__':main()
