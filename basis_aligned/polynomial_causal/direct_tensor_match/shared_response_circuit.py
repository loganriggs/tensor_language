"""Finite joint response through four shared residual writers and explicit RMSNorm.

Coefficients t mean residual delta=t@writers.T. Removal amplitudes must first
be divided by the native MLP17 normalization denominator, with a negative sign.
"""
import json,hashlib
from pathlib import Path
import torch
P=Path(__file__).resolve().parent

def background(state,writers,eps):
 return state.square().mean(-1,keepdim=True)+eps,state@writers/state.shape[-1]

def evaluate(pre_logits,old_square,projections,coefficients,gram,vocabulary_writes):
 new_square=old_square+2*(projections*coefficients).sum(-1,keepdim=True)+((coefficients@gram)*coefficients).sum(-1,keepdim=True)
 assert bool((new_square>0).all())
 pre_new=(old_square/new_square).sqrt()*pre_logits+(coefficients@vocabulary_writes.T)/new_square.sqrt()
 return 30*torch.tanh(pre_new/30)

def main():
 torch.set_num_threads(4);torch.set_grad_enabled(False)
 source=P/'SELECTIVE_SCALAR_READOUT_V1.pt';artifact=torch.load(source,weights_only=True);writers=artifact['residual_writer'].double();d,m=writers.shape
 checkpoint='/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin'
 state=torch.load(checkpoint,weights_only=True,mmap=True);unembed=state['lm_head.weight'].double();gram=writers.T@writers/d;vocab=unembed@writers;archive=vocab.float();rng=torch.Generator().manual_seed(261013);x=torch.randn(16,d,generator=rng,dtype=torch.float64);t=torch.randn(16,m,generator=rng,dtype=torch.float64)*.1/gram.diag().sqrt();eps=torch.finfo(torch.float32).eps;square,projections=background(x,writers,eps);pre=(x/square.sqrt())@unembed.T;delta=t@writers.T;changed=x+delta;native=30*torch.tanh(((changed/(changed.square().mean(-1,keepdim=True)+eps).sqrt())@unembed.T)/30);full=evaluate(pre,square,projections,t,gram,vocab);rounded=evaluate(pre,square,projections,t,gram,archive.double());baseline=30*torch.tanh(pre/30);replay=float((full-native).norm()/native.norm());effect_replay=float((rounded-native).norm()/(native-baseline).norm());assert replay<1e-12 and effect_replay<1e-5
 zero=evaluate(pre,square,projections,torch.zeros_like(t),gram,archive.double());assert torch.equal(zero,baseline)
 t1=t.clone();t1[:,2:]=0;t2=t-t1;combined=evaluate(pre,square,projections,t1+t2,gram,archive.double());assert torch.equal(combined,rounded)
 additive=evaluate(pre,square,projections,t1,gram,archive.double())+evaluate(pre,square,projections,t2,gram,archive.double())-baseline;nonadditive=float((rounded-additive).norm()/(rounded-baseline).norm());corr=gram/gram.diag().sqrt()[:,None]/gram.diag().sqrt()[None,:]
 export=dict(residual_writers=writers.float(),writer_gram=gram,vocabulary_writes=archive,source_scalar_artifact=source.name,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),input_convention='coefficients t are residual-edit coefficients AFTER branch denominator and sign; delta=t@residual_writers.T',scope='Exact joint RMSNorm/linear-unembedding/softcap interface plus rounded cached vocabulary writes. Requires original pre-softcap logits, residual norm and writer projections. No standalone-model or semantic claim.')
 torch.save(export,P/'SHARED_RESPONSE_CIRCUIT_V1.pt')
 result=dict(writers=m,residual_width=d,vocabulary_size=len(vocab),synthetic_exact_output_replay=replay,synthetic_cached_write_effect_replay=effect_replay,synthetic_joint_nonadditivity=nonadditive,writer_gram_eigenvalues=torch.linalg.eigvalsh(gram).tolist(),writer_cosines=corr.tolist(),stored_residual_writer_scalars=writers.numel(),stored_derived_vocabulary_cache_scalars=archive.numel(),stored_derived_gram_scalars=gram.numel(),scalar_library_coefficients=13916,total_with_scalar_library=13916+writers.numel()+archive.numel()+gram.numel(),scope=export['scope'])
 (P/'SHARED_RESPONSE_CIRCUIT_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
