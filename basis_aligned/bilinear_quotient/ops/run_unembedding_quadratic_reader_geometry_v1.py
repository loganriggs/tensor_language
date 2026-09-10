#!/usr/bin/env python3
# BQGATE: weight-only reader equivalence; 518 frozen tokens, no model forward or fit.
"""pred_a Gram bridges<=1e-5, CPU controls<=1e-10; pred_b >=16 distinct
token readers with full/trace-free errors<=.10 and raw U error>.50.
pred_c CPU dense/gauge algebra controls<=1e-10. Null zero candidates.
Native weights retained; no causal promotion.
"""
import json, os, signal, sys, time
from pathlib import Path
RUNNER=Path(__file__).resolve(); ROOT=RUNNER.parents[3]
POLY=ROOT/'basis_aligned/polynomial_causal'; sys.path[:0]=[str(ROOT),str(POLY)]
import torch
from induction_context_transport_v2 import digest
from circuit_fast_screen_managed_runner import atomic_create_json
OUT=POLY/'UNEMBEDDING_QUADRATIC_READER_GEOMETRY_V1_RESULT.json'
BIND=POLY/'UNEMBEDDING_QUADRATIC_READER_GEOMETRY_V1_BINDING.json'
HIER=POLY/'UNEMBEDDING_BACKWARD_VIEWS_V1_HIERARCHY.pt'
CHECKPOINT=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')

def kernel(l,r):
    cross=l@r.T
    return .5*((l@l.T)*(r@r.T)+cross*cross.T)

def forms(c,l,r):
    q=l.T@(c[:,None]*r)
    return .5*(q+q.T)

def rel(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))
def serial(x):return x.detach().cpu().tolist()

def controls():
    torch.set_num_threads(2); gen=torch.Generator().manual_seed(9111360)
    rand=lambda *s:torch.randn(*s,generator=gen,dtype=torch.float64)
    l,r,c=rand(11,7),rand(11,7),rand(5,11)
    k=kernel(l,r); q=torch.stack([forms(v,l,r) for v in c]); expected=q.flatten(1)@q.flatten(1).T
    scales=torch.linspace(.3,2.1,11,dtype=torch.float64)
    errors={'dense_gram':rel(c@k@c.T,expected),
            'exchange':rel(kernel(r,l),k),
            'rescaling':rel(kernel(l*scales[:,None],r/scales[:,None]),k)}
    t=(l*r).sum(-1); k0=k-t[:,None]*t[None,:]/7
    q0=q-torch.diagonal(q,dim1=1,dim2=2).sum(1)[:,None,None]*torch.eye(7)/7
    errors['trace_free']=rel(c@k0@c.T,q0.flatten(1)@q0.flatten(1).T)
    assert max(errors.values())<=1e-10
    return errors

def main():
    binding=json.loads(BIND.read_text()); assert all(digest(p)==h for p,h in binding.items())
    tiny=controls()
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'gpu_accessed':False,'model_loaded':False,'body_forwards':0,'controls':tiny}));return
    assert not OUT.exists(); signal.alarm(900); tic=time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32=False
    state=torch.load(CHECKPOINT,map_location='cpu',mmap=True,weights_only=True)
    saved=torch.load(HIER,map_location='cpu',weights_only=True)
    ids=saved['token_readers']; assert len(ids)==518
    u=state['lm_head.weight'][ids].float().cuda()
    centroids=saved['raw_centroids'].float().cuda(); labels=saved['labels'][ids].long().cuda()
    l=state['transformer.h.17.mlp.Left.weight'].float().cuda()
    r=state['transformer.h.17.mlp.Right.weight'].float().cuda()
    down=state['transformer.h.17.mlp.Down.weight'].float().cuda()
    with torch.inference_mode():
        readers=torch.cat([u,centroids]); c=readers@down
        k=kernel(l,r); gram=(c@k)@c.T
        traces=c@(l*r).sum(1)
        gram0=gram-traces[:,None]*traces[None,:]/l.shape[1]
        # Independent full-width FP64 control, including the coefficient contraction.
        control_indices=[0,1,518]
        cd=readers[control_indices].double()@down.double()
        qs=torch.stack([forms(v,l.double(),r.double()) for v in cd])
        direct=qs.flatten(1)@qs.flatten(1).T
        ix=torch.tensor(control_indices,device='cuda')
        q0=qs-torch.diagonal(qs,dim1=1,dim2=2).sum(1)[:,None,None]*torch.eye(1152,device='cuda',dtype=torch.float64)/1152
        bridge={'full':rel(gram[ix][:,ix].double(),direct),
                'trace_free':rel(gram0[ix][:,ix].double(),q0.flatten(1)@q0.flatten(1).T)}
        grams={'unembedding':readers@readers.T,'quadratic':gram,'trace_free':gram0}
        min_ratios={key:float(g.diag().min()/g.diag().abs().max()) for key,g in grams.items()}
        norm=gram0.diag()[:518].clamp_min(1e-30).sqrt()
        cosine=gram0[:518,:518]/norm[:,None]/norm[None,:]
        scores=cosine.abs().clone(); scores.fill_diagonal_(-1)
        nearest=scores.argmax(1); rows=[]
        for i in range(518):
            j=int(nearest[i]); rec={'token_id':ids[i],'partner_id':ids[j],'leaf':int(labels[i]),'partner_leaf':int(labels[j])}
            for key,g in grams.items():
                alpha=g[i,j]/g[j,j].clamp_min(1e-30)
                e2=1-g[i,j].square()/(g[i,i]*g[j,j]).clamp_min(1e-30)
                rec[key]={'optimal_alpha':float(alpha),'relative_error':float(e2.clamp_min(0).sqrt())}
            rec['candidate']=all(rec[k]['relative_error']<=.10 for k in ('quadratic','trace_free')) and rec['unembedding']['relative_error']>.50
            rows.append(rec)
        centroid_errors={}
        a=torch.arange(518,device='cuda'); b=518+labels
        for key,g in grams.items():
            e2=(g[a,a]+g[b,b]-2*g[a,b])/g[a,a].clamp_min(1e-30)
            centroid_errors[key]=serial(e2.clamp_min(0).sqrt())
        finite=all(bool(torch.isfinite(g).all()) for g in grams.values())
        instrument=finite and max(bridge.values())<=1e-5 and min(min_ratios.values())>0 and float(cosine.abs().max())<=1+1e-5
        count=sum(row['candidate'] for row in rows)
        torch.cuda.synchronize()
    result={'schema':'unembedding.quadratic_reader_geometry.v1','predictions':{'pred_a_instrument':instrument,'pred_b_sharing_candidates':instrument and count>=16,'pred_c_algebra_controls':max(tiny.values())<=1e-10},
            'candidate_count':count,'rows':rows,'centroid_relative_errors':centroid_errors,
            'controls':tiny,'trained_fp64_bridges':bridge,'minimum_diagonal_relative':min_ratios,
            'runner_sha256':digest(RUNNER),'binding_sha256':digest(BIND),'wall_seconds':time.perf_counter()-tic,
            'price':{'body_forwards':0,'native_weight_saving':0,'gram_scalars':4608**2,'reader_coefficients':534*4608,'reader_gram_scalars_each':534**2,
                     'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()},
            'scope':'Weight-only nomination; full and trace-free MLP17 quadratic functions. No native causal, extraction, removal, OOD or composition evidence.'}
    atomic_create_json(OUT,result)
    print(json.dumps({k:result[k] for k in ('predictions','candidate_count','trained_fp64_bridges','wall_seconds')}))

if __name__=='__main__':main()
