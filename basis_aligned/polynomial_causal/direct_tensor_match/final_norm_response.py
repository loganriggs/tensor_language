"""Exact finite RMSNorm response with linear unembedding, retaining softcap."""
import torch

def response(pre_logits,state,delta,delta_logits,eps):
 old_square=state.square().mean(-1,keepdim=True)+eps
 new_square=old_square+2*(state*delta).mean(-1,keepdim=True)+delta.square().mean(-1,keepdim=True)
 assert bool((new_square>0).all())
 old_scale=old_square.sqrt();new_scale=new_square.sqrt();ratio=old_scale/new_scale
 cap=lambda z:30*torch.tanh(z/30)
 return dict(baseline=cap(pre_logits),full=cap(ratio*pre_logits+delta_logits/new_scale),direct_fixed_normalizer=cap(pre_logits+delta_logits/old_scale),normalizer_only=cap(ratio*pre_logits),scale_ratio=ratio)

def toy_check():
 gen=torch.Generator().manual_seed(261007);rand=lambda *shape:torch.randn(*shape,generator=gen,dtype=torch.float64)
 x=rand(32,7);delta=.3*rand(32,7);U=rand(19,7);eps=torch.finfo(x.dtype).eps;pre=(x/(x.square().mean(-1,keepdim=True)+eps).sqrt())@U.T
 r=response(pre,x,delta,delta@U.T,eps);target=30*torch.tanh((((x+delta)/((x+delta).square().mean(-1,keepdim=True)+eps).sqrt())@U.T)/30);error=float((r['full']-target).norm()/target.norm());assert error<1e-12
 zero=response(pre,x,torch.zeros_like(x),torch.zeros_like(pre),eps);assert torch.equal(zero['baseline'],zero['full'])
 y=rand(32,7);y=y/y.norm(dim=1,keepdim=True)*x.norm(dim=1,keepdim=True);normpreserved=response(pre,x,y-x,(y-x)@U.T,eps);normerror=float((normpreserved['normalizer_only']-normpreserved['baseline']).abs().max());assert normerror<1e-12
 # Common shifts before softcapping need not be common after softcapping.
 z=torch.tensor([[-20.,0.,20.]],dtype=torch.float64);effect=30*(torch.tanh((z+5)/30)-torch.tanh(z/30));noncommon=float((effect-effect.mean(-1,keepdim=True)).norm());assert noncommon>.1
 return dict(finite_response_relative_error=error,norm_preserved_normalizer_effect=normerror,pre_softcap_common_shift_centered_effect_norm=noncommon)
if __name__=='__main__':
 import json
 from pathlib import Path
 r=toy_check();Path(__file__).with_name('FINAL_NORM_RESPONSE_ORACLE_V1.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
