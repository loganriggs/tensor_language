#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_integrity pred_b_capacity pred_c_stability
"""Same16reader nativequartic first-slot maps; two256Gaussiantriple sketches.
Nativeautograd<1e-8/trace<1e-10; rank256floor>.10both;
rank256floordifference<.05/rank512floor>.10both. Finite-sketchbounds, nofullnormcertificate.
"""
import os,sys,time,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'

def main():
 import torch
 sys.path.insert(0,str(P));from quartic_slot_reader import first_slot,controls
 from quartic_cp import directional
 from quartic_reader_rank_bound import analyze
 from paired_root_compiler import cast
 from audit_root_matched_reader import CK
 torch.set_num_threads(2)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(controls=controls(),triples=256,seeds=[970,971],ranks=[64,128,256,512,768,1024])));return
 torch.backends.cuda.matmul.allow_tf32=False;start=time.monotonic();out=P/'QUARTIC_SLOT_CAPACITY_V1.json';assert not out.exists();s=torch.load(CK,weights_only=True,mmap=True,map_location='cpu');base=cast(torch.load(P/'EXPANDED_ROOT_EMPIRICAL_V1.pt',weights_only=True),torch.float64);scale=19054614563.464127
 def w(layer,name):return s[f'transformer.h.{layer}.mlp.{name}.weight'].cuda().double()
 with torch.no_grad():
  uv=s['lm_head.weight'].cuda().double();uw=uv@base['writer'].cuda();readers=uv.T@uw/uw.square().sum(0);del uv,uw
  teacher=[readers.T@w(17,'Down')/scale,w(17,'Left'),w(17,'Right'),w(16,'Down')*s['transformer.h.17.lambdas'][0].item(),w(16,'Left'),w(16,'Right')]
 grams=[];check=None;precision=None;sketch_hashes=[]
 for seed in [970,971]:
  vectors=torch.randn(3,256,1152,generator=torch.Generator().manual_seed(seed),dtype=torch.float64);sketch_hashes.append(hashlib.sha256(vectors.numpy().tobytes()).hexdigest());vectors=vectors.cuda();g=torch.zeros(1152,1152,device='cuda',dtype=torch.float64)
  for index in range(0,256,16):
   b,c,d=[z[index:index+16] for z in vectors]
   with torch.no_grad():maps=first_slot(teacher,b,c,d);flat=maps.reshape(-1,1152);g.add_(flat.T@flat/256)
   if check is None:
    a=torch.zeros(1,1152,device='cuda',dtype=torch.float64,requires_grad=True);fun=lambda aa:directional(*teacher,[aa,b[:1],c[:1],d[:1]])[0];ref=torch.autograd.functional.jacobian(fun,a)[:,0,:];check=float((maps[0]-ref).norm()/ref.norm())
    with torch.no_grad():fp32=first_slot([t.float() for t in teacher],b[:1].float(),c[:1].float(),d[:1].float()).double();precision=float((maps[:1]-fp32).norm()/maps[:1].norm())
  grams.append(g);print(json.dumps(dict(seed=seed,trace=float(g.trace()))),flush=True)
 with torch.no_grad():
  result,bases=analyze(grams,[64,128,256,512,768,1024]);trace_errors=[abs(float(g.trace())-r['trace'])/r['trace'] for g,r in zip(grams,result['panels'])];spaces={}
  for name,file in [('inherited','EXPANDED_ROOT_EMPIRICAL_V1.pt'),('exact_learned','EXACT_ROOT_FEATURE_LEARNED_V1.pt')]:
   program=torch.load(P/file,weights_only=True);read=torch.cat([program['U'].flatten(0,1),program['V'].flatten(0,1)],0).cuda().double().T;q=torch.linalg.qr(read).Q;spaces[name]=[float(((g.trace()-(q*(g@q)).sum())/g.trace()).clamp_min(0).sqrt()) for g in grams]
 pred=dict(pred_a_integrity=check<1e-8 and max(trace_errors)<1e-10,pred_b_capacity=all(r['floors']['256']>.1 for r in result['panels']),pred_c_stability=abs(result['panels'][0]['floors']['256']-result['panels'][1]['floors']['256'])<.05 and all(r['floors']['512']>.1 for r in result['panels']))
 result.update(predictions=pred,native_autograd_replay=check,fp32_fp64_map_difference=precision,trace_errors=trace_errors,existing_spaces=spaces,sketch_sha256=sketch_hashes,seconds=time.monotonic()-start,scope='Exactrankfloorsforfinitefirst-slotcontractionmetric; unbiasedGramestimate butnocertifiedfulltensorFrobeniusfloor. Same16readerpurequartic,independentslots,notnativebehavior.')
 torch.save(dict(grams=[g.cpu() for g in grams],seeds=[970,971],triples=256,sketch_sha256=sketch_hashes),P/'QUARTIC_SLOT_CAPACITY_GRAMS_V1.pt');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
