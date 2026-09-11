#!/usr/bin/env python3
# BQGATE: 0forwards0seq; exact36-mode routing-contrast factorization, full centeredU.
import os,sys,json,time,signal,hashlib,math
from pathlib import Path
RUNNER=Path(__file__).resolve();P=RUNNER.parents[3]/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from routing_wedge_covariance_v1 import covariance
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    binding=json.loads((P/'ROUTING_WEDGE_COVARIANCE_V1_BINDING.json').read_text());assert all(digest(p)==h for p,h in binding.items())
    assert json.loads((P/'ROUTING_WEDGE_COVARIANCE_V1_CONTROL.json').read_text())['passed']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,model_loaded=False,body_forwards=0,corpus_access=False,output_rank_truncation=False,routing_wedge_modes=36)));return
    out=P/'ROUTING_WEDGE_COVARIANCE_V1_RESULT.json';assert not out.exists();signal.alarm(900);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True);u=sd['lm_head.weight'].double().cuda();u-=u.mean(0);metric=u.T@u
    l,r,d=[sd[f'transformer.h.17.mlp.{k}.weight'].double().cuda() for k in ['Left','Right','Down']]
    k=d.T@metric@d;o=sd['transformer.h.17.attn.c_proj.weight'].double().cuda();vc=sd['transformer.h.17.attn.c_v.weight'].double().cuda();vb=sd['transformer.h.0.attn.c_v.weight'].double().cuda();mix=float(sd['transformer.h.17.attn.lamb']);w=torch.cat(((1-mix)*vc,mix*vb),dim=1)
    gram,pairs=covariance(l,r,o,w,k,9);values,vectors=torch.linalg.eigh(gram);values=values.flip(0);vectors=vectors.flip(1);total=gram.trace()
    reference=json.loads((P/'FULL_SOURCE_QUADRATIC_ENERGY_V1_RESULT.json').read_text())['maps']['native']['antisymmetric']
    trace_error=abs(float(total)/reference-1);replay=float((gram-(vectors*values)@vectors.T).norm()/gram.norm());negative=max(0.,-float(values.min()/total))
    modes=[]
    for j in range(4):
        mat=gram.new_zeros(9,9)
        for n,(h,z) in enumerate(pairs):mat[h,z]=vectors[n,j]/2**.5;mat[z,h]=-vectors[n,j]/2**.5
        sv=torch.linalg.svdvals(mat);pair_capture=float(sv[:2].square().sum()/sv.square().sum())
        modes.append(dict(mode=j,energy_fraction=float(values[j]/total),single_alternating_pair_capture=pair_capture,head_matrix=mat.cpu().tolist()))
    valid=max(trace_error,replay,negative)<=1e-10 and bool(torch.isfinite(values).all())
    captures={str(z):float(values[:z].sum()/total) for z in [1,2,4,8,16,24,32,36]}
    result=dict(predictions={'pred_a_instrument':valid,'pred_b_four_shared_routing_contrasts':valid and captures['4']>=.5,'pred_c_simple_head_pairs':valid and all(x['single_alternating_pair_capture']>=.9 for x in modes)},capture=captures,rank90=int(torch.searchsorted(values.cumsum(0)/total,.9))+1,eigenvalues=values.cpu().tolist(),eigenvectors=vectors.cpu().tolist(),head_pairs=pairs,gram=gram.cpu().tolist(),leading_modes=modes,bridges=dict(trace=trace_error,eigen_replay=replay,negative_fraction=negative),price=dict(body_forwards=0,output_rank_truncation=False,routing_mode_dimension=36,temporary_gb_estimate=14),wall_seconds=time.perf_counter()-tic,scope='Exact optimal linear routing-wedge unfolding approximation to full centered antisymmetric coefficient tensor. Arbitrary source/output functions remain; no identified circuit or cheap full program.')
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({key:value for key,value in result.items() if key not in ['gram','eigenvectors','leading_modes']}),flush=True)
if __name__=='__main__':main()
