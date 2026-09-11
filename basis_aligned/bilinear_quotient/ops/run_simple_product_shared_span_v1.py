#!/usr/bin/env python3
"""pred_a identities1e-8; pred_b all36converged; pred_c6heads gain/sharing bars.

BQGATE:0forwards0seq. All native producer weights; four starts per fixed head.
See SIMPLE_PRODUCT_SHARED_SPAN_V1_PREREGISTRATION.md for scope and exact bars.
"""
import os,sys,json,time,signal,hashlib
from pathlib import Path
RUNNER=Path(__file__).resolve();P=RUNNER.parents[3]/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from simple_product_in_span_v1 import fit,value_gradient
from producer_function_pairs_v1 import pairs,shared_private
from sparse_product_dictionary_v1 import form,unit_product
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    binding=json.loads((P/'SIMPLE_PRODUCT_SHARED_SPAN_V1_BINDING.json').read_text())
    assert all(digest(f)==h for f,h in binding.items())
    assert json.loads((P/'SIMPLE_PRODUCT_IN_SPAN_V1_CONTROL.json').read_text())['instrument_passed']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,body_forwards=0,corpus_access=False,heads=9,starts=36,seconds_per_start=15)));return
    out=P/'SIMPLE_PRODUCT_SHARED_SPAN_V1_RESULT.json';assert not out.exists()
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False;signal.alarm(1200)
    start=time.perf_counter();inputs={}
    def cached(name):
        c=json.loads((P/name).read_text())['cache'];assert digest(c['path'])==c['sha256']
        inputs[c['path']]=c['sha256'];return torch.load(c['path'],weights_only=True,map_location='cpu')
    midpoint=cached('COUPLED_PRODUCER_MIDPOINT_V1_AUDIT.json');prod=cached('MLP16_PRODUCER_OVERLAP_V1_RESULT.json')
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    l,r,d=[sd[f'transformer.h.16.mlp.{key}.weight'].double().cuda() for key in ['Left','Right','Down']]
    g=prod['metric'].cuda();rows=[];saved=[];errors=[]
    for h in range(9):
        left,right,cosine=pairs(midpoint['frames'][h].T.cuda(),prod['ov_readers'][h].cuda(),g)
        common=shared_private(left,right,cosine)['common'];coeff=common@d
        # Form only one head's17 quadratics; no vocabulary tensor or text states.
        forms=torch.matmul(l.T[None,:,:]*coeff[:,None,:],r);forms=(forms+forms.transpose(-1,-2))/2
        errors.append(float((forms.flatten(1)@forms.flatten(1).T-torch.eye(17,device='cuda')).abs().max()))
        ev=torch.linalg.eigvalsh(forms)
        base=ev[:,-1].clamp_min(0).square()+ev[:,0].clamp_max(0).square()
        s=(forms@forms).sum(0);se,sv=torch.linalg.eigh(s);se=se.flip(0)
        errors.append(float((s@sv-sv*se.flip(0)).norm()/s.norm()))
        bounds={str(k):min(1.,float(se[:2*k].sum())) for k in [1,4,16]}
        initial=[torch.eye(17)[0],torch.eye(17)[int(base.argmax())]]
        for seed in [1471+100*h,1483+100*h]:
            gen=torch.Generator().manual_seed(seed);x=torch.randn(17,generator=gen);initial.append(x/x.norm())
        fits=[];points=[]
        for x in initial:
            point,row=fit(forms,x,seconds=15);fits.append(row);points.append(point)
        best=max(range(4),key=lambda i:fits[i]['score']);point=points[best].cuda()
        q=torch.einsum('a,aij->ij',point,forms);pl,pr,magnitude=unit_product(q)
        qhat=form(pl[None,:],pr[None,:],magnitude[None])
        errors.append(abs(float(1-(q-qhat).square().sum()/q.square().sum())-fits[best]['score']))
        common_reader=point@common
        weights=point/(2*(1+cosine)).sqrt();u=weights@left;v=weights@right
        similarity=float((u@g@v)/((u@g@u)*(v@g@v)).sqrt())
        errors.append(float((common_reader-u-v).norm()/common_reader.norm()))
        errors.append(max(0.,fits[best]['score']-bounds['1']))
        frame_score=fits[best]['score'];old=float(base[0]);gain=frame_score-old
        row=dict(head=h,leading_pair_capture=old,best_basis_capture=float(base.max()),
            score=frame_score,gain=gain,ratio=frame_score/old,paired_function_cosine=similarity,
            parent_span_projection_energy=float((point.square()*(1+cosine)/2).sum()),
            whole_span_product_capture_upper_bounds=bounds,original_pair_cosines=cosine.tolist(),
            fits=fits,best=best,passes_structural_screen=frame_score>=2*old and gain>=.05 and similarity>=.5,
            minimum_extreme_gap=min(x['extreme_eigenvalue_gap'] for x in fits))
        rows.append(row);saved.append(dict(head=h,points=torch.stack(points),best=best,
            reader=common_reader.cpu(),left_reader=u.cpu(),right_reader=v.cpu(),
            product_left=pl.cpu(),product_right=pr.cpu(),product_scale=magnitude.cpu()))
        torch.save(dict(heads=saved,binding=binding,inputs=inputs),'/dev/shm/bilin18_simple_product_shared_span_v1_progress.pt')
        print(json.dumps(row),flush=True)
    valid=max(errors)<1e-8
    cache=Path('/dev/shm/bilin18_simple_product_shared_span_v1.pt');assert not cache.exists();torch.save(dict(heads=saved,binding=binding,inputs=inputs),cache)
    result=dict(predictions={'pred_a_instrument':valid,
        'pred_b_convergence':valid and all(f['converged'] for x in rows for f in x['fits']),
        'pred_c_simple_shared_function':valid and sum(x['passes_structural_screen'] for x in rows)>=6},
        heads=rows,maximum_instrument_error=max(errors),wall_seconds=time.perf_counter()-start,
        summary=dict(mean_old_capture=sum(x['leading_pair_capture'] for x in rows)/9,
            mean_new_capture=sum(x['score'] for x in rows)/9,mean_pair_cosine=sum(x['paired_function_cosine'] for x in rows)/9,
            converged_starts=sum(f['converged'] for x in rows for f in x['fits']),passing_heads=sum(x['passes_structural_screen'] for x in rows)),
        cache=dict(path=str(cache),sha256=digest(cache),bytes=cache.stat().st_size,ephemeral=True),
        body_forwards=0,corpus_access=False,binding=binding,inputs=inputs,
        scope='One-product optimized combinations within fixed17 common quadratic functions. Local convergence, not global optimum; neither shared identity nor semantic circuit. Original QK source frames remain frozen.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='heads'},indent=2),flush=True)
    assert valid

if __name__=='__main__':main()
