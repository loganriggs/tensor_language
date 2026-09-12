#!/usr/bin/env python3
# BQGATE:0bodyforwards,0textsequences;128cachednativeendpoints,rank16frozenweightSVD,300sec.
"""pred_a exact coefficient/compiled/numerical replay; pred_b allfamily swaps<=.1;
pred_c removalCEerror<=.02; pred_d writes<=.05. Reused developmental panel.
"""
import os,sys,time,json,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from sparse_path_stability_atlas_v1 import digest
from quadratic_producer_projection_v1 import atom_gram
from quartic_frozen_native_score_v2 import score
STEM='COMPOSED_PARENT_NATIVE_V1'

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(body_forwards=0,text_sequences=0,cached_endpoints=128,partner_rank=16)));return
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_PROGRAM.pt');assert not out.exists() and not artifact.exists()
    signal.alarm(300);start=time.perf_counter();torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False
    sd=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True)
    original=torch.load(P/'COMPOSED_SHARED_PARENT_V1_PROGRAM.pt',weights_only=True)
    u=sd['lm_head.weight'].double().cuda();uc=u-u.mean(0);root=torch.linalg.cholesky(uc.T@uc)
    l,r,d=[sd[f'transformer.h.16.mlp.{n}.weight'].double().cuda() for n in ('Left','Right','Down')]
    scale=original['producer_scale'];h=scale**2*d@atom_gram(l,r)@d.T
    he,hv=torch.linalg.eigh(h);hs=(hv*he.sqrt())@hv.T;hi=(hv*he.rsqrt())@hv.T
    a=original['readers'][original['best']].cuda();ap=original['physical_reader'].cuda();mp=original['physical_partner'].cuda()
    eye=torch.eye(1152,device='cuda');j=eye+(2**.5-1)*a[:,None]*a[None,:];ji=eye+(2**-.5-1)*a[:,None]*a[None,:]
    weighted=root.T@mp@hs@j
    us,s,vs=torch.linalg.svd(weighted,full_matrices=False)
    writer=torch.linalg.solve_triangular(root.T,us[:,:16]*s[:16],upper=True)
    reader=vs[:16]@ji@hi
    coeff=scale*d.T@torch.cat([ap[:,None],reader.T],1)
    x=torch.load(P/'QUARTIC_GROUP_PORTS_V1_PORTS.pt',weights_only=True)['input16'].double().cuda()
    ports=torch.load(P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True)['ports']
    den=ports['pre'].double().cuda().square().mean(-1)+torch.finfo(torch.float32).eps
    hidden=(x@l.T)*(x@r.T);producer=hidden@d.T*scale
    reference=(producer@ap)[:,None]*(producer@mp.T)/den[:,None]
    candidate=(producer@ap)[:,None]*(producer@reader.T@writer.T)/den[:,None]
    compiled=hidden@coeff;compiled=compiled[:,:1]*(compiled[:,1:]@writer.T)/den[:,None]
    capture=float(s[:16].square().sum()/s.square().sum())
    old=json.loads((P/'COMPOSED_SHARED_PARENT_V1_RESULT.json').read_text())
    errors=[abs(capture-old['partner_rank16_capture']),float((candidate-compiled).norm()/candidate.norm())]
    rows=json.loads((P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ROWS.json').read_text())['rows']
    torch.set_default_dtype(torch.float32)
    effects=score([reference.cpu(),reference.cpu(),candidate.cpu()],ports['pre']+ports['native_output'],rows,sd['lm_head.weight'].float())
    baseline=effects['reports'][0]
    errors.extend(f['swap_relative_rms'] for f in baseline['families']);errors.extend(f['zero_ce_meanabs_disagreement'] for f in baseline['families'])
    family=[]
    for name in sorted({r['family'] for r in rows}):
        ids=torch.tensor([i for i,r in enumerate(rows) if r['family']==name]);ep=(2*ids[:,None]+torch.tensor([0,1])).flatten().cuda()
        swaps=torch.tensor(effects['reference_effects']['swaps'][0])[ids]
        family.append(dict(family=name,write_relative_error=float((candidate[ep]-reference[ep]).norm()/reference[ep].norm()),reference_swap_meanabs=float(swaps.abs().mean())))
    torch.save(dict(physical_parent=ap.cpu(),partner_readers=reader.cpu(),partner_writers=writer.cpu(),compiled_product_readers=coeff.cpu(),reference_write=reference.cpu(),candidate_write=candidate.cpu(),required_native_input_maps=['transformer.h.16.mlp.Left.weight','transformer.h.16.mlp.Right.weight']),artifact)
    result={'pred_a':max(errors)<=1e-8 and effects['pred_a'],'pred_b':effects['pred_b'],'pred_c':effects['pred_c'],'pred_d':all(f['write_relative_error']<=.05 for f in family)}
    result.update(effects=effects,families=family,identity_errors=errors,coefficient_capture=capture,conditional_compiled_values=l.numel()+r.numel()+coeff.numel()+writer.numel(),execution_seconds=time.perf_counter()-start,source_shas=binding,artifact_sha=digest(artifact),scope='Frozen component approximation on reused development rows, not semantic or OOD identification.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
