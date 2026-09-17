"""Exact MLP8 response for the full 128-channel head8.2 writer subspace.
CPU synthetic-state identity on learned weights; not native causal evidence.
"""
from pathlib import Path
import hashlib,json,time,torch
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);torch.manual_seed(17092243);start=time.perf_counter()
 binding=json.loads((P/'PHI4_PROVENANCE_V1_BINDING.json').read_text())['files']
 checkpoint=next(k for k in binding if k.endswith('pytorch_model.bin'))
 sd=torch.load(checkpoint,weights_only=True,mmap=True,map_location='cpu')
 L,R,D=[sd['transformer.h.8.mlp.'+name+'.weight'].double() for name in ['Left','Right','Down']]
 pp=P/'TYPED_FACE_KEY_SOURCE_FRESH_V1_PROGRAM.pt'
 W=torch.load(pp,weights_only=True,map_location='cpu')['output'].double()
 LW=L@W;RW=R@W;eps=torch.finfo(torch.float32).eps
 def terms(g,c):
  delta=c@W.T;g1=g+delta;s0=g.square().mean(-1,keepdim=True)+eps;s1=g1.square().mean(-1,keepdim=True)+eps
  lg=g@L.T;rg=g@R.T;lc=c@LW.T;rc=c@RW.T;m0=(lg*rg)@D.T/s0
  return {'normalization':(s0/s1-1)*m0,'cross':(lc*rg+lg*rc)@D.T/s1,'quadratic':(lc*rc)@D.T/s1,'skip':delta}
 g=torch.randn(24,1152,dtype=torch.float64);base=torch.randn(24,128,dtype=torch.float64);checks=[]
 def block(x):return x+((x@L.T)*(x@R.T))@D.T/(x.square().mean(-1,keepdim=True)+eps)
 for scale in [.001,.1,1.]:
  c=scale*base;t=terms(g,c);actual=block(g+c@W.T)-block(g);pred=sum(t.values());error=float((pred-actual).norm()/actual.norm())
  checks.append({'channel_scale':scale,'relative_closure':error,'terms':{k:{'norm_ratio':float(v.norm()/actual.norm()),'aligned_fraction':float((v*actual).sum()/actual.square().sum())} for k,v in t.items()}})
 assert max(x['relative_closure'] for x in checks)<1e-10
 result={'pred_a':True,'checks':checks,'shapes':{'W':list(W.shape),'LW':list(LW.shape),'RW':list(RW.shape)},'derived_LW_RW_scalars':LW.numel()+RW.numel(),'native_L_R_D_scalars':L.numel()+R.numel()+D.numel(),'seconds':time.perf_counter()-start,'head8_program_sha256':hashlib.sha256(pp.read_bytes()).hexdigest(),'checkpoint_sha256_from_binding':binding[checkpoint],'identity':'delta block8 = Wc + (s0/s1-1)*M0_biasfree + D[(LWc)*(Rg)+(Lg)*(RWc)]/s1 + D[(LWc)*(RWc)]/s1; s=mean(g^2)+eps','scope':'FP64 learned-weight identity on synthetic states, FP32 epsilon. Bias cancels. Context g remains explicit. Derived adapters do not remove native L,R,D or prove storage savings. Requires native fixture and downstream factorial verification.'}
 (P/'MLP8_WRITER_RESPONSE_FOLD_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
