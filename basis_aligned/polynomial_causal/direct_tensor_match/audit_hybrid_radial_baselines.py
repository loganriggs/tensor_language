"""Native Gaussian quartic radial controls; cost includes variable products."""
import json
from pathlib import Path
import torch
from paired_root_compiler import cast
from audit_root_matched_reader import CK
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);dtype=torch.float64;s=torch.load(CK,weights_only=True,mmap=True,map_location='cpu');base=cast(torch.load(P/'EXPANDED_ROOT_EMPIRICAL_V1.pt',weights_only=True),dtype);uv=s['lm_head.weight'].double();uw=uv@base['writer'];readers=uv.T@uw/uw.square().sum(0);del uv,uw
 def w(layer,name):return s[f'transformer.h.{layer}.mlp.{name}.weight'].double()
 a,b,d=w(16,'Left'),w(16,'Right'),w(16,'Down')*s['transformer.h.17.lambdas'][0];l,r,o=w(17,'Left'),w(17,'Right'),w(17,'Down');folded=o.T@readers
 xs=[torch.randn(n,1152,generator=torch.Generator().manual_seed(seed),dtype=dtype) for n,seed in [(2048,938),(1024,939)]];ys=[]
 for x in xs:
  parts=[]
  for z in x.split(256):
   m=((z@a.T)*(z@b.T))@d.T;parts.append(((m@l.T)*(m@r.T))@folded)
  ys.append(torch.cat(parts))
 projection=torch.linalg.qr(torch.randn(1152,32,generator=torch.Generator().manual_seed(940),dtype=dtype)).Q
 results=[]
 for name,phis,products,coeff in [('full_radial',[(x.square().sum(1)/1152).square() for x in xs],1153,16),('projected_radial32',[((x@projection).square().sum(1)/32).square() for x in xs],33,1152*32+16)]:
  c=phis[0]@ys[0]/phis[0].square().sum();rows=[]
  for phi,y in zip(phis,ys):
   pred=phi[:,None]*c;rr=(pred-y).square().sum(0).sqrt()/y.square().sum(0).sqrt();rows.append(dict(aggregate_error=float((pred-y).norm()/y.norm()),root1_error=float(rr[1]),mean_root_error=float(rr.mean())))
  results.append(dict(name=name,results=rows,variable_products=products,learned_or_stored_coefficients=coeff,output_writer_common=True))
 out=dict(results=results,seeds=[938,939,940],scope='Native purequartic16reader targets; radialcoefffitonGaussiantrain, evalfresh. Fullradial1153products exceeds384budget; projected33productcontrol hasfixedrandom32dimdictionary. Common1152x16writerexcludedfrombothcoeffprices explicitly. No text/nativebehavior/semanticclaim.')
 (P/'HYBRID_RADIAL_BASELINES_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
