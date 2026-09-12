#!/usr/bin/env python3
# BQGATE:0bodyforwards,0textsequences;1152cachedendpoints;300sec.
"""Fixed node, exact self, rank670 perpendicular partners: 14481792 floats.
pred_a identities<=1e-8 and coefficient dominance; pred_b every swap<=.1,
sign>=.9/live>=4; pred_c removal disagreement<=.02; pred_d writes<=.05.
Null: coefficient-optimal partners still fail native fidelity. No text fitting.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from sparse_path_stability_atlas_v1 import digest
from quadratic_producer_projection_v1 import atom_gram
from sparse_frame_function_inner_v1 import coefficients
from sparse_producer_graph_execute_v1 import execute,incident
from shared_input_factor_v1 import native_partner
from quartic_frozen_native_score_v2 import score
STEM='MATCHED_PARTNER_SUBSPACE_V1'

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
    assert all(digest(k)==v for k,v in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(body_forwards=0,text_sequences=0,cached_endpoints=1152,rank=670,conditional_floats=14481792)));return
    signal.alarm(300);tic=time.perf_counter();torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_PROGRAM.pt');assert not out.exists() and not artifact.exists()
    sd=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True)
    u=sd['lm_head.weight'].double().cuda();uc=u-u.mean(0);root=torch.linalg.cholesky(uc.T@uc).T;del u,uc
    l0,r0,d0,l1,r1,d1=[sd[f'transformer.h.{layer}.mlp.{name}.weight'].double().cuda() for layer in (16,17) for name in ('Left','Right','Down')]
    scale=float(sd['transformer.h.17.lambdas'][0]);h=scale**2*d0@atom_gram(l0,r0)@d0.T
    he,hv=torch.linalg.eigh(h);assert float(he.min())>0
    hs=(hv*he.sqrt())@hv.T;hi=(hv*he.rsqrt())@hv.T
    prior=json.loads((P/'AMORTIZED_SPARSE_FRAME_FIDELITY_V1_RESULT.json').read_text())
    saved=torch.load(P/'AMORTIZED_SPARSE_FRAME_NATIVE_V1_FRAMES.pt',weights_only=True)
    q=saved['frames'][prior['best_arm']].cuda();edges=saved['edges'][prior['best_arm']].cuda();node=prior['node']['index'];a=q[:,node]
    edges=edges[:,incident(edges,[node])];assert edges.shape[1]==671 and int(((edges[0]==node)&(edges[1]==node)).sum())==1
    partner_ids=torch.unique(edges);partner_ids=partner_ids[partner_ids!=node];assert len(partner_ids)==670
    l,r=l1@hs,r1@hs;old_writer=coefficients(q,l,r,d1,edges)
    m=native_partner(a,l,r,d1);self_writer=m@a;mperp=m-self_writer[:,None]*a[None,:]
    left,singular,right=torch.linalg.svd(root@mperp,full_matrices=False)
    rank=670;v=right[:rank].T;w=torch.linalg.solve_triangular(root,left[:,:rank]*singular[:rank],upper=True)
    approx_perp=w@v.T;old_perp=mperp@q[:,partner_ids]@q[:,partner_ids].T
    coefficient_error=float((root@(mperp-approx_perp)).square().sum()/2)
    old_coefficient_error=float((root@(mperp-old_perp)).square().sum()/2)
    readers=torch.cat([a[:,None],v],dim=1);writers=torch.cat([self_writer[:,None],w],dim=1)
    compiled=scale*d0.T@hi@readers
    checks=dict(unit=abs(float(a@a)-1),perpendicular=float((v.T@a).norm()),
                self_error=float((approx_perp@a).norm()/self_writer.norm()),
                svd_error=abs(coefficient_error-float(singular[rank:].square().sum()/2))/max(coefficient_error,1e-30))
    panels={}
    for name in ('development','context'):
        if name=='development':
            rows=json.loads((P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ROWS.json').read_text())['rows']
            ports=torch.load(P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True)['ports']
            x=torch.load(P/'QUARTIC_GROUP_PORTS_V1_PORTS.pt',weights_only=True)['input16'].double().cuda()
            prior_effects=prior['node']['effects']['reference_effects']
        else:
            rows=json.loads((P/'PRODUCER_METRIC_CONTEXT_HOLDOUT_V1_ROWS.json').read_text())['rows']
            cache=torch.load(P/'PRODUCER_METRIC_CONTEXT_HOLDOUT_V1_PORTS.pt',weights_only=True,mmap=True)
            assert cache['rows_sha256']==digest(P/'PRODUCER_METRIC_CONTEXT_HOLDOUT_V1_ROWS.json')
            ports=cache['ports'];x=ports['input16'].double().cuda()
            prior_effects=json.loads((P/'SPARSE_NODE_CONTEXT_V1_RESULT.json').read_text())['effects']['reference_effects']
        den=ports['pre'].double().cuda().square().mean(-1)+torch.finfo(torch.float32).eps
        hidden=(x@l0.T)*(x@r0.T);p=scale*hidden@d0.T;z=p@hi
        native=lambda pp:((pp@l1.T)*(pp@r1.T))@d1.T
        ref=(native(p)-native(p-(z@a)[:,None]*(hs@a)[None,:]))/den[:,None]
        exact=(z@a)[:,None]*(z@m.T)/den[:,None]
        old=execute(z@q,edges,old_writer,den)
        values=hidden@compiled;candidate=values[:,0,None]*(values@writers.T)/den[:,None]
        direct=(z@a)[:,None]*(z@(self_writer[:,None]*a[None,:]+approx_perp).T)/den[:,None]
        checks[name+'_exact']=float((exact-ref).norm()/ref.norm())
        checks[name+'_compiled']=float((candidate-direct).norm()/candidate.norm())
        old_direct=(z@a)[:,None]*(z@(self_writer[:,None]*a[None,:]+old_perp).T)/den[:,None]
        checks[name+'_old_feasible']=float((old-old_direct).norm()/old.norm())
        torch.set_default_dtype(torch.float32)
        effects=score([ref.cpu(),old.cpu(),candidate.cpu()],ports['pre']+ports['native_output'],rows,sd['lm_head.weight'].float(),reference_effects=prior_effects)
        torch.set_default_dtype(torch.float64)
        families=[]
        for family in sorted({row['family'] for row in rows}):
            ids=torch.tensor([i for i,row in enumerate(rows) if row['family']==family],device='cuda');ep=(2*ids[:,None]+torch.tensor([0,1],device='cuda')).flatten()
            families.append(dict(family=family,write_relative_error=float((candidate[ep]-ref[ep]).norm()/ref[ep].norm())))
        panels[name]=dict(effects=effects,families=families,pred_b=effects['pred_b'],pred_c=effects['pred_c'],pred_d=all(f['write_relative_error']<=.05 for f in families))
    torch.save(dict(compiled_producer_readers=compiled.cpu(),output_writers=writers.cpu(),node=node,
                    required_native_maps=['MLP16.Left','MLP16.Right'],execution='h16=(L16 x)*(R16 x); t=h16 C; write=t0*(t W^T)/den17',
                    conditional_floats=14481792),artifact)
    result=dict(**{'pred_a':max(checks.values())<=1e-8 and coefficient_error<=old_coefficient_error*(1+1e-10) and all(p['effects']['pred_a'] for p in panels.values())},
                **{key:all(p[key] for p in panels.values()) for key in ('pred_b','pred_c','pred_d')},
                checks=checks,panels=panels,coefficient_error=coefficient_error,old_coefficient_error=old_coefficient_error,
                relative_coefficient_error_reduction=1-coefficient_error/old_coefficient_error,rank=rank,node=node,
                conditional_floats=14481792,artifact_sha=digest(artifact),source_shas=binding,execution_seconds=time.perf_counter()-tic,
                scope='Conditional weight-optimal partner approximation of frozen node; previously inspected panels, no semantic/OOD identification.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('source_shas','panels')}),flush=True)

if __name__=='__main__':main()
