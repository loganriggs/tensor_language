#!/usr/bin/env python3
# BQGATE: 0forwards0seq; exact two-QK coefficient signatures and32formal probes.
import os,sys,json,time,signal,hashlib
from pathlib import Path
RUNNER=Path(__file__).resolve();P=RUNNER.parents[3]/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
import torch.nn.functional as F
from joint_router_polynomial_gram_v1 import biquadratic_inner,normalizer_inner
from folded_normalized_router_v1 import rotary,EPS
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def gram_report(gram):
    cosine=gram/gram.diag().sqrt()[:,None]/gram.diag().sqrt()[None,:]
    error=(1-cosine.square()).clamp_min(0)
    negative=max(0.,-float(torch.linalg.eigvalsh(gram).min()/gram.trace()))
    return dict(cosine=cosine.cpu().tolist(),proportional_squared_error=error.cpu().tolist(),negative_eigenvalue_fraction=negative,focus_error=float(error[2,3]),focus_cosine=float(cosine[2,3]))
def main():
    binding=json.loads((P/'JOINT_ROUTER_SIGNATURE_NATIVE_V1_BINDING.json').read_text());assert all(digest(p)==h for p,h in binding.items())
    assert json.loads((P/'JOINT_ROUTER_POLYNOMIAL_GRAM_V1_CONTROL.json').read_text())['passed']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,model_loaded=False,body_forwards=0,corpus_access=False,heads=9,query_position=8,source_positions=[8,7,0],formal_queries=32)));return
    out=P/'JOINT_ROUTER_SIGNATURE_NATIVE_V1_RESULT.json';assert not out.exists();signal.alarm(900);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    q1,k1,q2,k2=[sd[f'transformer.h.17.attn.{key}.weight'].double().cuda().reshape(9,128,1152) for key in ['c_q','c_k','c_q2','c_k2']]
    gq1=q1.transpose(1,2)@q1/128;gq2=q2.transpose(1,2)@q2/128;gk1=k1.transpose(1,2)@k1/128;gk2=k2.transpose(1,2)@k2/128
    norms={}
    for name,a,b in [('query',gq1,gq2),('key',gk1,gk2)]:
        gram=a.new_zeros(9,9)
        for h in range(9):
            for z in range(h,9):gram[h,z]=gram[z,h]=normalizer_inner(a[h],b[h],a[z],b[z],EPS,EPS)
        norms[name]=gram_report(gram)
    generator=torch.Generator().manual_seed(712091)
    probe=torch.randn(4,32,1152,generator=generator,dtype=torch.float64)
    probe=F.rms_norm(probe,(1152,),eps=EPS).cuda();query=probe[0];rt=rotary(8,128).cuda()
    positions={};scores={};bridges=[];valid=True
    for n,source in enumerate([8,7,0]):
        rs=rotary(source,128).cuda();rotation=rt.T@rs
        a=q1.transpose(1,2)@rotation@k1/128;b=q2.transpose(1,2)@rotation@k2/128
        gram=a.new_zeros(9,9)
        for h in range(9):
            for z in range(h,9):gram[h,z]=gram[z,h]=biquadratic_inner(a[h],b[h],a[z],b[z])
        positions[str(source)]=gram_report(gram);key=probe[n+1]
        folded=[];direct=[]
        for h in range(9):
            numerator=((query@a[h])*key).sum(-1)*((query@b[h])*key).sum(-1)
            denominator=(((query@gq1[h])*query).sum(-1)+EPS)*(((query@gq2[h])*query).sum(-1)+EPS)*(((key@gk1[h])*key).sum(-1)+EPS)*(((key@gk2[h])*key).sum(-1)+EPS)
            folded.append(numerator/denominator.sqrt())
            branches=[]
            for q,k in [(q1[h],k1[h]),(q2[h],k2[h])]:
                qx=F.rms_norm(query@q.T,(128,),eps=EPS)@rt.T;ky=F.rms_norm(key@k.T,(128,),eps=EPS)@rs.T
                branches.append((qx*ky).sum(-1)/128)
            direct.append(branches[0]*branches[1])
        folded=torch.stack(folded);direct=torch.stack(direct)
        bridge=float((folded-direct).norm()/direct.norm());bridges.append(bridge);scores[source]=direct
    aa=torch.stack([scores[7][2],scores[0][2]],dim=1);bb=torch.stack([scores[7][3],scores[0][3]],dim=1)
    denominator=aa.norm(dim=1)*bb.norm(dim=1);assert bool((denominator>1e-30).all())
    sine=(aa[:,0]*bb[:,1]-aa[:,1]*bb[:,0]).abs()/denominator
    negative=max(v['negative_eigenvalue_fraction'] for v in list(norms.values())+list(positions.values()))
    valid=max(bridges+[negative])<=1e-10 and bool(torch.isfinite(sine).all())
    signature_errors=[v['focus_error'] for v in list(norms.values())+list(positions.values())]
    result=dict(predictions={'pred_a_instrument':valid,'pred_b_focus_shared_signature':valid and max(signature_errors)<=.10,'pred_c_focus_proportional_routing':valid and float(sine.max())<=.10},normalizers=norms,numerators_by_source_position=positions,focus_two_source_sines=sine.cpu().tolist(),focus_two_source_sine_summary=dict(minimum=float(sine.min()),median=float(sine.median()),maximum=float(sine.max())),focus_probe_scores=dict(head2=aa.cpu().tolist(),head3=bb.cpu().tolist()),folded_direct_relative_errors=bridges,price=dict(body_forwards=0,corpus_access=False,formal_queries=32,heads=9,positions=3),wall_seconds=time.perf_counter()-tic,scope='Joint QK product and full norm-polynomial coefficient signatures at3fixedpositions. Formal continuous probes can disprove universal proportional routing, not task-specific natural agreement; no identified circuit.')
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ['normalizers','numerators_by_source_position','focus_probe_scores','focus_two_source_sines']}),flush=True)
if __name__=='__main__':main()
