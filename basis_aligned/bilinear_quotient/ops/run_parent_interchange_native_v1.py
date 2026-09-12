#!/usr/bin/env python3
# BQGATE:0bodyforwards,0textsequences;128cachedendpoints,oldparentweightsfrozen,300sec.
"""New full star is reference. Old centered parent with optimal new-metric
partners is candidate; old unchanged star on same producer path is baseline.
pred_a reference write<=1e-8/effects<=1e-5 replay, finite;
pred_b allfamily swaps<=.1,sign>=.9,live>=4;
pred_c removalCEdisagreement<=.02; pred_d familywrites<=.05.
No factor/data fitting or rank cut. Reused development panel; not semantic OOD.
"""
import os,sys,time,json,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from sparse_path_stability_atlas_v1 import digest
from quadratic_producer_projection_v1 import atom_gram
from shared_input_factor_v1 import native_partner
from quartic_frozen_native_score_v2 import score
STEM='PARENT_INTERCHANGE_NATIVE_V1'

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(body_forwards=0,text_sequences=0,cached_endpoints=128,fits=0)));return
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_PROGRAM.pt');assert not out.exists() and not artifact.exists()
    signal.alarm(300);start=time.perf_counter();torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False
    sd=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True)
    old=torch.load('/dev/shm/bilin18_shared_input_factor_native_v1.pt',weights_only=True)['centered']
    new=torch.load(P/'COMPOSED_SHARED_PARENT_V1_PROGRAM.pt',weights_only=True)
    l0,r0,d0,l1,r1,d1=[sd[f'transformer.h.{layer}.mlp.{name}.weight'].double().cuda() for layer in (16,17) for name in ('Left','Right','Down')]
    scale=new['producer_scale'];h=scale**2*d0@atom_gram(l0,r0)@d0.T
    he,hv=torch.linalg.eigh(h);hs=(hv*he.sqrt())@hv.T;hi=(hv*he.rsqrt())@hv.T
    raw=old['readers'][old['best']].cuda();a=hs@raw;a/=a.norm()
    ap=hi@a;mp=native_partner(a,l1@hs,r1@hs,d1)@hi
    x=torch.load(P/'QUARTIC_GROUP_PORTS_V1_PORTS.pt',weights_only=True)['input16'].double().cuda()
    ports=torch.load(P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True)['ports']
    den=ports['pre'].double().cuda().square().mean(-1)+torch.finfo(torch.float32).eps
    producer=((x@l0.T)*(x@r0.T))@d0.T*scale
    def write(reader,partner):return (producer@reader)[:,None]*(producer@partner.T)/den[:,None]
    ref=write(new['physical_reader'].cuda(),new['physical_partner'].cuda())
    initial=write(raw,old['partner'].cuda());candidate=write(ap,mp)
    cached=torch.load(P/'COMPOSED_PARENT_NATIVE_V1_PROGRAM.pt',weights_only=True)['reference_write'].cuda()
    error=float((ref-cached).norm()/ref.norm())
    oldeffects=json.loads((P/'COMPOSED_PARENT_NATIVE_V1_RESULT.json').read_text())['effects']['reference_effects']
    rows=json.loads((P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ROWS.json').read_text())['rows'];torch.set_default_dtype(torch.float32)
    effects=score([ref.cpu(),initial.cpu(),candidate.cpu()],ports['pre']+ports['native_output'],rows,sd['lm_head.weight'].float(),oldeffects)
    families=[]
    for name in sorted({r['family'] for r in rows}):
        ix=torch.tensor([i for i,r in enumerate(rows) if r['family']==name]);ep=(2*ix[:,None]+torch.tensor([0,1])).flatten().cuda()
        families.append(dict(family=name,write_relative_error=float((candidate[ep]-ref[ep]).norm()/ref[ep].norm())))
    torch.save(dict(physical_reader=ap.cpu(),physical_partner=mp.cpu(),reference_write=ref.cpu(),initial_write=initial.cpu(),candidate_write=candidate.cpu(),producer_scale=scale),artifact)
    result={'pred_a':error<=1e-8 and effects['pred_a'],'pred_b':effects['pred_b'],'pred_c':effects['pred_c'],'pred_d':all(f['write_relative_error']<=.05 for f in families)}
    result.update(conditional_values=ap.numel()+mp.numel()+l0.numel()+r0.numel()+d0.numel(),effects=effects,families=families,write_replay_error=error,execution_seconds=time.perf_counter()-start,source_shas=binding,artifact_sha=digest(artifact),scope='Old/new frozen parent interfaces on same producer path and reused development panel; no semantic equivalence or OOD claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
