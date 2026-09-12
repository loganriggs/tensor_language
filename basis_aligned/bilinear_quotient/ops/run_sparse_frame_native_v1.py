#!/usr/bin/env python3
# BQGATE:0bodyforwards,0textsequences;128cachedendpoints,2frozencomparisons;300sec.
"""pred_a compiled/reference replay; pred_b swaps<=.1/sign>=.9/live>=4;
pred_c removalCEerror<=.02; pred_d writes<=.05. Developmental, no text fitting.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from sparse_path_stability_atlas_v1 import digest
from quadratic_producer_projection_v1 import atom_gram
from fullu_input_blocks_v1 import sandwich
from sparse_frame_function_inner_v1 import coefficients
from sparse_producer_graph_execute_v1 import execute,incident,control
from quartic_frozen_native_score_v2 import score
STEM='SPARSE_FRAME_NATIVE_V1'

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    assert control()['passed']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(body_forwards=0,text_sequences=0,cached_endpoints=128)));return
    signal.alarm(300);tic=time.perf_counter();torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_WRITES.pt');assert not out.exists() and not artifact.exists()
    sd=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True)
    fit=json.loads((P/'STREAMED_SPARSE_FRAME_NATIVE_V1_RESULT.json').read_text())
    saved=torch.load(P/'STREAMED_SPARSE_FRAME_NATIVE_V1_FRAMES.pt',weights_only=True)
    best=max(range(len(fit['reports'])),key=lambda i:fit['reports'][i]['history'][-1]['capture'])
    u=sd['lm_head.weight'].double().cuda();uc=u-u.mean(0);metric=uc.T@uc;root=torch.linalg.cholesky(metric).T;del u,uc
    l0,r0,d0,l1,r1,d1=[sd[f'transformer.h.{layer}.mlp.{name}.weight'].double().cuda() for layer in (16,17) for name in ('Left','Right','Down')]
    scale=float(sd['transformer.h.17.lambdas'][0]);h=scale**2*d0@atom_gram(l0,r0)@d0.T
    he,hv=torch.linalg.eigh(h);assert float(he.min())>0
    hs=(hv*he.sqrt())@hv.T;hi=(hv*he.rsqrt())@hv.T;l,r=l1@hs,r1@hs
    k=sandwich(l,r,d1.T@metric@d1,torch.eye(1152,device='cuda'));_,qi=torch.linalg.eigh((k+k.T)/2);qi=qi.flip(1)
    old=json.loads((P/'FULL_INPUT_SPARSE_CORE_V2_RESULT.json').read_text())['reports'][1]
    ei=torch.tensor(old['top4096_edges'],device='cuda');q=saved['frames'][best].cuda();e=saved['edges'][best].cuda()
    x=torch.load(P/'QUARTIC_GROUP_PORTS_V1_PORTS.pt',weights_only=True)['input16'].double().cuda()
    ports=torch.load(P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True)['ports']
    den=ports['pre'].double().cuda().square().mean(-1)+torch.finfo(torch.float32).eps
    hidden=(x@l0.T)*(x@r0.T);p=scale*hidden@d0.T
    native=lambda pp:((pp@l1.T)*(pp@r1.T))@d1.T
    reference=native(p)/den[:,None]
    initial=execute(p@hi@qi,ei,coefficients(qi,l,r,d1,ei),den)
    writer=coefficients(q,l,r,d1,e);z=hidden@(scale*d0.T@hi@q)
    candidate=execute(z,e,writer,den)
    errors=[float((candidate-execute(p@hi@q,e,writer,den)).norm()/candidate.norm())]
    energy=(root@writer).square().sum(0);inc=torch.zeros(1152,device='cuda')
    inc.scatter_add_(0,e[0],energy);inc.scatter_add_(0,e[1],energy*(e[0]!=e[1]))
    node=int(inc.argmax());mask=incident(e,[node]);a=q[:,node]
    reduced=p-(p@hi@a)[:,None]*(hs@a)[None,:]
    node_reference=(native(p)-native(reduced))/den[:,None]
    node_candidate=execute(z,e,writer,den,mask)
    rows=json.loads((P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ROWS.json').read_text())['rows']
    previous=json.loads((P/'FULLU_PAIRED_PRODUCER_V1_RESULT.json').read_text())['effects']['reference_effects']
    torch.set_default_dtype(torch.float32)
    state=ports['pre']+ports['native_output'];u=sd['lm_head.weight'].float()
    effects=score([reference.cpu(),initial.cpu(),candidate.cpu()],state,rows,u,reference_effects=previous)
    node_effects=score([node_reference.cpu(),node_reference.cpu(),node_candidate.cpu()],state,rows,u)
    def family_errors(ref,approx):
        result=[]
        for name in sorted({r['family'] for r in rows}):
            ids=torch.tensor([i for i,r in enumerate(rows) if r['family']==name],device='cuda');ep=(2*ids[:,None]+torch.tensor([0,1],device='cuda')).flatten()
            result.append(dict(family=name,write_relative_error=float((approx[ep]-ref[ep]).norm()/ref[ep].norm())))
        return result
    families=family_errors(reference,candidate);node_families=family_errors(node_reference,node_candidate)
    torch.save(dict(reference=reference.cpu(),initial=initial.cpu(),candidate=candidate.cpu(),node_reference=node_reference.cpu(),node_candidate=node_candidate.cpu()),artifact)
    result={'pred_a':max(errors)<=1e-8 and effects['pred_a'] and node_effects['pred_a'],
            'pred_b':effects['pred_b'],'pred_c':effects['pred_c'],'pred_d':all(v['write_relative_error']<=.05 for v in families)}
    result.update(best_arm=best,fit_converged=fit['reports'][best]['converged'],effects=effects,families=families,
                  conditional_compiled_floats=20643840,edge_indices=8192,
                  node=dict(index=node,edges=int(mask.sum()),readers=int(torch.unique(e[:,mask]).numel()),
                            conditional_compiled_floats=10616832+4608*int(torch.unique(e[:,mask]).numel())+1152*int(mask.sum()),
                            effects=node_effects,families=node_families,
                            write_bar=all(v['write_relative_error']<=.05 for v in node_families)),identity_errors=errors,
                  execution_seconds=time.perf_counter()-tic,source_shas=binding,artifact_sha=digest(artifact),
                  scope='Frozen developmental fullpath and one weight-selected local node; no semantic/OOD claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('source_shas','effects','node')}),flush=True)

if __name__=='__main__':main()
