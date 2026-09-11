#!/usr/bin/env python3
"""A exact conditional/replay; B both legal writer solves converge; C concentration gain at80%capture."""
# BQGATE: 0forwards0seq; fixed learned input products, weight-only writer red-team.
import os,sys,json,time,hashlib,signal
from pathlib import Path
RUNNER=Path(__file__).resolve();P=RUNNER.parents[3]/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from joint_quadratic_fit_v1 import product_cross
from sparse_product_dictionary_v1 import form
from legal_token_writer_admm_v1 import solve
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def metrics(p,z,g,c,total):
    capture=float((2*(z*c).sum()-((z@g)*z).sum())/total)
    writers=p@z;contrasts=writers-writers.mean(0)
    concentration=contrasts.square().topk(16,dim=1).values.sum(1)/contrasts.square().sum(1).clamp_min(1e-30)
    return dict(full_coefficient_capture=capture,median_top16_contrast_loading_energy=float(concentration.median()),
                mean_top16_contrast_loading_energy=float(concentration.mean()),
                actual_contrast_nonzero_entries=int((contrasts!=0).sum()))


def main():
    binding=json.loads((P/'PRODUCT_WRITER_REDTEAM_V1_BINDING.json').read_text());assert all(digest(p)==h for p,h in binding.items())
    assert json.loads((P/'LEGAL_TOKEN_WRITER_ADMM_V1_CONTROL.json').read_text())['instrument_passed']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,model_loaded=False,body_forwards=0,corpus_access=False,
            tokens=50304,products=512,legal_writer_penalty_multipliers=[1.,.1],seconds_per_solve=180)))
        return
    out=P/'PRODUCT_WRITER_REDTEAM_V1_RESULT.json';assert not out.exists();signal.alarm(1200)
    start=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    prior=json.loads((P/'SPARSE_PRODUCT_DICTIONARY_V1_RESULT.json').read_text())
    cache=Path(prior['cache']['path']);assert digest(cache)==prior['cache']['sha256']
    saved=torch.load(cache,weights_only=True,map_location='cpu');pl=saved['left'].cuda();pr=saved['right'].cuda();a=saved['codes'].cuda()
    scale=saved['scale'];penalty=saved['penalty'];sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    u=sd['lm_head.weight'].double().cuda();uc=u-u.mean(0)
    l,r,down=[sd[f'transformer.h.17.mlp.{key}.weight'].double().cuda() for key in ['Left','Right','Down']]
    native_cross=product_cross(l,r,pl,pr);g=product_cross(pl,pr,pl,pr)
    eigen=torch.linalg.eigvalsh(g);assert float(eigen[0])>0
    cross=(uc@down@native_cross)/scale
    center_ols=torch.linalg.solve(g,cross.T).T
    center_capture=float((center_ols*cross).sum()/len(u))
    support=a!=0;counts=support.sum(1);debias=torch.zeros_like(a);max_residual=0.;max_condition=0.
    for count in counts.unique().tolist():
        if count==0:continue
        rows=torch.where(counts==count)[0]
        for chunk in rows.split(512):
            idx=support[chunk].nonzero()[:,1].reshape(len(chunk),count)
            small=g[idx[:,:,None],idx[:,None,:]];rhs=cross[chunk[:,None],idx]
            solution=torch.linalg.solve(small,rhs[:,:,None]).squeeze(-1)
            residual=(small@solution[:,:,None]).squeeze(-1)-rhs
            max_residual=max(max_residual,float((residual.norm(dim=1)/rhs.norm(dim=1).clamp_min(1e-30)).max()))
            ev=torch.linalg.eigvalsh(small);max_condition=max(max_condition,float((ev[:,-1]/ev[:,0]).max()))
            debias[chunk[:,None],idx]=solution
    debias_capture=float((2*(debias*cross).sum()-((debias.T@debias)*g).sum())/len(u))
    root=torch.linalg.cholesky(u.T@u);p=torch.linalg.solve_triangular(root,u.T,upper=False).T
    c=root.T@down@native_cross/scale
    native_gram=product_cross(l,r,l,r)
    total=float(((down.T@(u.T@u)@down)*native_gram).sum()/scale**2)
    full_energy_error=abs(total*scale**2/99245061353.473-1)
    z0=torch.linalg.solve(g,c.T).T
    baseline=metrics(p,z0,g,c,total)
    orth_error=float((p.T@p-torch.eye(p.shape[1],dtype=p.dtype,device=p.device)).norm()/p.shape[1]**.5)
    ols_residual=float((z0@g-c).norm()/c.norm())
    arms=[];solutions={}
    for multiplier in [1.,.1]:
        z,report=solve(p,g,c,penalty*multiplier,rho=1.,max_steps=5000,tolerance=1e-5,seconds=180)
        stats=metrics(p,z,g,c,total)
        row=dict(penalty_multiplier=multiplier,penalty=penalty*multiplier,**stats,solver=report)
        arms.append(row);solutions[str(multiplier)]=z.cpu();print(json.dumps({k:v for k,v in row.items() if k!='solver'}|{'solver_final':report['final'],'converged':report['converged'],'seconds':report['seconds']}),flush=True)
    # Direct8-row full matrices verify full-U least-squares loss, separate from centered source fit.
    indices=[0,1,10,100,1000,10000,30000,50256];writers=p@z0;direct=0.;target=0.
    for i in indices:
        q=form(l,r,u[i]@down/scale);direct+=float((q-form(pl,pr,writers[i])).square().sum());target+=float(q.square().sum())
    selected=writers[indices];fullcross=u[indices]@down@native_cross/scale
    implicit=target+float(((selected.T@selected)*g).sum()-2*(selected*fullcross).sum())
    bridge=abs(direct-implicit)/max(direct,1e-30)
    valid=max(full_energy_error,orth_error,ols_residual,bridge,max_residual)<=1e-8
    candidate=arms[1]
    target_cache=Path('/dev/shm/bilin18_product_writer_redteam_v1.pt');assert not target_cache.exists()
    torch.save(dict(whitened_writers=solutions,root=root.cpu(),left=pl.cpu(),right=pr.cpu(),scale=scale,
        source_cache_sha256=prior['cache']['sha256']),target_cache)
    result=dict(predictions={'pred_a_instrument':valid,'pred_b_legal_writer_convergence':valid and all(x['solver']['converged'] for x in arms),
        'pred_c_concentration_at_retained_capture':valid and candidate['solver']['converged']
            and candidate['median_top16_contrast_loading_energy']>=1.25*baseline['median_top16_contrast_loading_energy']
            and candidate['full_coefficient_capture']>=.8*baseline['full_coefficient_capture']},
        source_product_status=prior['status'],source_penalty=penalty,
        source_centered_capture=prior['final']['captured_energy'],fixed_support_debiased_centered_capture=debias_capture,
        unrestricted_fixed_product_centered_capture=center_capture,
        fixed_support_links=int(support.sum()),maximum_support_solve_residual=max_residual,maximum_support_condition=max_condition,
        full_legal_unpenalized_baseline=baseline,legal_arms=arms,
        gram_condition=float(eigen[-1]/eigen[0]),full_energy_error=full_energy_error,
        output_whitening_error=orth_error,ols_stationarity=ols_residual,direct_full_matrix_bridge_error=bridge,
        cache=dict(path=str(target_cache),sha256=digest(target_cache),bytes=target_cache.stat().st_size,ephemeral=True),
        wall_seconds=time.perf_counter()-start,
        scope='Fixed learned products only. Centered penalty-shrinkage audit and separate full-U legal-writer solves; no joint input optimization or physical replacement. Loading concentration is not additive function energy.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='legal_arms'}),flush=True)


if __name__=='__main__':main()
