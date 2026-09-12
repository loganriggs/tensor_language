#!/usr/bin/env python3
# BQGATE:0bodyforwards;16planes;<=496scalarevaluations;300sec.
"""pred_a interpolation<=1e-8/orth<=1e-10; pred_b perarm improvement>=1e-6;
pred_c nonlocal angle>=pi/8 improves>=1e-5. Null: no escape in bounded panel.
"""
import sys,os,json,time,signal,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from sparse_path_stability_atlas_v1 import digest
from graded_source_projection_v1 import graded_norms,balanced_loss
from graded_projection_plane_v1 import search
STEM='GRADED_NATIVE_PLANE_AUDIT_V2'
FIT_SHA='3e5a045e1f439ecc3c8b0834631725f198f08253a9b766e7fc8a4fe0cf02841a'
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('16 planes; 0 body forwards; terminal fit required for actual execution');return
    terminal=json.loads((P/'GRADED_SOURCE_LBFGS_V1_RESULT.json').read_text());fitpath=ROOT/'basis_aligned/bilinear_quotient/ops/run_graded_source_lbfgs_v1.py';assert terminal['source_shas'][str(fitpath)]==FIT_SHA
    program_path=P/'GRADED_SOURCE_LBFGS_V1_PROGRAM.pt';assert digest(program_path)==terminal['artifact_sha']
    out=P/(STEM+'_RESULT.json');ap=P/(STEM+'_FRAMES.pt');assert not out.exists() and not ap.exists();signal.alarm(300)
    tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    data=torch.load(P/'COUPLED_SOURCE_PROJECTION_NATIVE_V1_INPUTS.pt',weights_only=True,map_location='cpu');a=data['forms'].cuda();root=data['root'].cuda();root=root/root.square().sum().div(len(root)).sqrt();wg=data['writer_gram'].cuda();full=graded_norms(a,root@root.T,wg)
    frames=torch.load(program_path,weights_only=True,map_location='cpu')['frames'];reports=[];best_frames=[];counts=0
    for arm,p in enumerate(frames):
        p=p.cuda();complement=torch.linalg.qr(p,mode='complete').Q[:,p.shape[1]:]
        q=(p@p.T).detach().requires_grad_();ret=graded_norms(a,root@q@root,wg);value=(1-ret[1:]/full[1:]).mean();g=torch.autograd.grad(value,q)[0];g=(g+g.T)/2
        _,inside=torch.linalg.eigh(p.T@g@p);_,outside=torch.linalg.eigh(complement.T@g@complement)
        gen=torch.Generator(device='cuda').manual_seed(73160+arm);planes=[]
        for j in range(4):
            c=torch.randn(p.shape[1],device='cuda',dtype=p.dtype,generator=gen);c=c/c.norm();z=torch.randn(complement.shape[1],device='cuda',dtype=p.dtype,generator=gen);z=z/z.norm();planes.append(('random',p@c,complement@z))
        for j in range(4):planes.append(('gradient',p@inside[:,-1-j],complement@outside[:,j]))
        best=float(value.detach());best_frame=p.detach();original=best;plane_reports=[]
        with torch.no_grad():
            for kind,v,w in planes:
                c=p.T@v
                def rotated(theta):return p+((math.cos(theta)-1)*v+math.sin(theta)*w)[:,None]*c[None,:]
                def objective(theta):
                    nonlocal counts
                    counts+=1;assert counts<=496
                    return float(balanced_loss(a,root,rotated(theta),wg,full))
                found=search(objective);candidate=rotated(found['angle']);orth=float((candidate.T@candidate-torch.eye(p.shape[1],device='cuda')).norm())
                found.update(kind=kind,orthogonality=orth,angle_distance=min(found['angle'],math.pi-found['angle']))
                plane_reports.append(found)
                if found['value']<best:best=found['value'];best_frame=candidate.clone()
        best_frames.append(best_frame.cpu());reports.append(dict(arm=arm,original_loss=original,best_loss=best,improvement=original-best,original_status=terminal['fits'][arm]['status'],original_gradient=terminal['fits'][arm]['gradient'],planes=plane_reports))
    checks=[x for r in reports for x in r['planes']]
    result={'pred_a':all(x['interpolation_max_error']<=1e-8 and x['orthogonality']<=1e-10 and math.isfinite(x['value']) and math.isfinite(x['initial']) for x in checks),
            'pred_b':all(r['improvement']>=1e-6 for r in reports),
            'pred_c':any(x['improvement']>=1e-5 and x['angle_distance']>=math.pi/8 for x in checks)}
    torch.save(dict(frames=best_frames,original_sha=digest(program_path)),ap)
    result.update(reports=reports,evaluations=counts,artifact_sha=digest(ap),original_program_sha=digest(program_path),source_shas=binding,execution_seconds=time.perf_counter()-tic)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('reports','source_shas')}),flush=True)
if __name__=='__main__':main()
