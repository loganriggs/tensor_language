#!/usr/bin/env python3
"""A exactness; B4starts converge; C>=5%same-price gain; Dfunctioncos>=.99."""
# BQGATE: 0forwards0seq; exact conditional rank16 partner plus reader optimization.
import os,sys,json,time,hashlib,signal
from pathlib import Path
RUNNER=Path(__file__).resolve();P=RUNNER.parents[3]/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from shared_reader_partner_rcg_v1 import optimize
from shared_input_factor_v1 import native_partner
from joint_quadratic_fit_v1 import product_cross
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def component_inner(a,m,b,n,ug):
    return .5*((a@b)*((ug@m)*n).sum()+(m@b)@ug@(n@a))


def main():
    binding=json.loads((P/'JOINT_SHARED_READER_RANK16_V1_BINDING.json').read_text());assert all(digest(p)==h for p,h in binding.items())
    assert json.loads((P/'SHARED_READER_PARTNER_RCG_V1_CONTROL.json').read_text())['instrument_passed']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,body_forwards=0,corpus_access=False,rank=16,starts=4,
                             steps_each=1000,seconds_each=120,tangent_stationarity=1e-8)))
        return
    out=P/'JOINT_SHARED_READER_RANK16_V1_RESULT.json';assert not out.exists();signal.alarm(900)
    start=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    prior=json.loads((P/'SHARED_INPUT_FACTOR_NATIVE_V1_RESULT.json').read_text());source=Path(prior['cache']['path'])
    assert digest(source)==prior['cache']['sha256'];saved=torch.load(source,weights_only=True,map_location='cpu')
    baseline=json.loads((P/'SHARED_INPUT_PARTNER_OBJECTIVE_V1_AUDIT.json').read_text())
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True);u=sd['lm_head.weight'].double().cuda();uc=u-u.mean(0)
    l,r,down=[sd[f'transformer.h.17.mlp.{key}.weight'].double().cuda() for key in ['Left','Right','Down']]
    ug=uc.T@uc;fullug=u.T@u;root=torch.linalg.cholesky(ug);total=92104252412.19983
    eye=torch.eye(1152,dtype=u.dtype,device=u.device);fits=[];programs=[];matrices=[];readers=[];errors=[]
    for label in ['centered','full',1163,1171]:
        if isinstance(label,str):initial=saved[label]['readers'][saved[label]['best']].cuda()
        else:
            torch.manual_seed(label);initial=torch.randn(1152,dtype=u.dtype,device=u.device)
        a,fit=optimize(initial,l,r,down,root,total,16,max_steps=1000,tolerance=1e-8,seconds=120)
        continuation=fit.pop('continuation')
        if isinstance(label,str):errors.append(abs(fit['initial_capture']-baseline['comparisons']['centered'][label]['rank16_partner_native_capture']))
        exact=native_partner(a,l,r,down);j=eye+(2**.5-1)*a[:,None]*a[None,:]
        invj=eye+(2**-.5-1)*a[:,None]*a[None,:]
        left,sv,right=torch.linalg.svd(root.T@exact@j,full_matrices=False)
        writers=torch.linalg.solve_triangular(root.T,left[:,:16]*sv[:16].sqrt(),upper=True)
        partners=(sv[:16].sqrt()[:,None]*right[:16])@invj
        m=writers@partners;aa=a.repeat(16,1)
        gram=product_cross(aa,partners,aa,partners);cross=product_cross(l,r,aa,partners)
        energy=((writers.T@ug@writers)*gram).sum();overlap=((down.T@ug@writers)*cross).sum()
        implicit=float((2*overlap-energy)/total);errors.append(abs(implicit-fit['capture']))
        errors.append(abs(float(component_inner(a,m,a,m,ug)/total)-float(energy/total)))
        torch.manual_seed(1181);x=torch.randn(64,1152,dtype=u.dtype,device=u.device)
        compact=(x@a)[:,None]*((x@partners.T)@writers.T)
        dense=(x@a)[:,None]*(x@m.T)
        execution_error=float((compact-dense).norm()/dense.norm());errors.append(execution_error)
        full_energy=float(component_inner(a,m,a,m,fullug))
        row=dict(label=label,**fit,implicit_native_capture=implicit,
                 fraction_of_exact_reader_component=float(sv[:16].square().sum()/sv.square().sum()),
                 cutoff_relative_singular_gap=float((sv[15]-sv[16])/sv[15]),
                 common_fraction_of_full_component=1-float(energy)/full_energy,
                 compact_execution_relative_error=execution_error)
        fits.append(row);matrices.append(m);readers.append(a)
        programs.append(dict(reader=a.cpu(),writers=writers.cpu(),partner_readers=partners.cpu(),continuation=continuation))
        print(json.dumps({k:v for k,v in row.items() if k!='history'}),flush=True)
    best=max(range(4),key=lambda i:fits[i]['capture']);a,m=readers[best],matrices[best]
    norm=component_inner(a,m,a,m,ug)
    cosines=[float(component_inner(a,m,b,n,ug)/(norm*component_inner(b,n,b,n,ug)).sqrt()) for b,n in zip(readers,matrices)]
    reader_cosines=[float(abs(a@b)) for b in readers]
    valid=max(errors)<=1e-8;converged=all(f['converged'] for f in fits)
    target=Path('/dev/shm/bilin18_joint_shared_reader_rank16_v1.pt');assert not target.exists();torch.save(dict(programs=programs,best=best),target)
    result=dict(predictions={'pred_a_instrument':valid,'pred_b_all_converged':valid and converged,
        'pred_c_improved_capture':valid and converged and fits[best]['capture']>=1.05*.006266262704505802,
        'pred_d_function_stability':valid and converged and min(cosines)>=.99},
        fits=fits,best=best,function_cosines=cosines,reader_cosines=reader_cosines,
        relative_capture_gain_over_best_prior=fits[best]['capture']/.006266262704505802-1,
        maximum_instrument_error=max(errors),
        cache=dict(path=str(target),sha256=digest(target),bytes=target.stat().st_size,ephemeral=True),
        price=dict(component_numbers=38016,input_projections=17,products=16,native_background_required=True,
                   body_forwards=0,corpus_access=False),wall_seconds=time.perf_counter()-start,
        scope='Joint shared-reader/rank16-partner partial component. Full reader-removal formula needs the omitted partner remainder; no semantic selectivity or model-level adoption.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='fits'}),flush=True)


if __name__=='__main__':main()
