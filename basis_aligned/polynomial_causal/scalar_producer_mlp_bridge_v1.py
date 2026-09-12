"""Exact finite-amplitude producer->MLP8->raw layer9 read update; algebra probes only."""
import json,torch
from pathlib import Path
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);torch.manual_seed(9231745)
 out=P/'SCALAR_PRODUCER_MLP_BRIDGE_V1_RESULT.json';assert not out.exists()
 binding=json.loads((P/'STRUCTURED_PRODUCER_CACHE_V1_BINDING.json').read_text())['files']
 state=torch.load(next(k for k in binding if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
 producer=torch.load(P/'SCALAR_PRODUCERS_COMPILE_V1_PROGRAM.pt',weights_only=True,map_location='cpu')
 d=torch.load(P/'SCALAR_PRODUCER_NATIVE_LIFT_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['writers'][0].double()
 E=torch.cat([producer[k][1].double() for k in ('q1','k1','q2','k2')]+[producer['current_value_readers'][1,None].double()])
 prefix='transformer.h.8.mlp.'
 L,R,D=[state[prefix+k+'.weight'].double() for k in ('Left','Right','Down')]
 eps=torch.finfo(torch.float32).eps;scale=float(state['transformer.h.9.lambdas'][0])
 z=torch.randn(32,1152,dtype=torch.float64);ratios=torch.tensor([0,.01,.1,1.],dtype=torch.float64).repeat(8)
 a=ratios*z.norm(dim=1)/d.norm();a[1::2]*=-1
 zm=z-a[:,None]*d
 rho=z.square().mean(-1,keepdim=True)+eps;rhom=zm.square().mean(-1,keepdim=True)+eps
 zl=z@L.T;zr=z@R.T;ld=L@d;rd=R@d
 def readwrite(products):return (products@D.T)@E.T
 base=readwrite(zl*zr);cross=readwrite(zl*rd+zr*ld);square=readwrite(ld*rd)
 direct=-a[:,None]*(E@d)
 norm=(rhom.reciprocal()-rho.reciprocal())*base
 mixed=-a[:,None]/rhom*cross;quad=a[:,None].square()/rhom*square
 composed=scale*(direct+norm+mixed+quad)
 actual=scale*((zm-z)@E.T+readwrite((zm@L.T)*(zm@R.T))/rhom-base/rho)
 error=float((composed-actual).norm()/actual.norm())
 records=[]
 for value in (0.01,.1,1.):
  ix=ratios==value;den=actual[ix].norm()
  records.append(dict(perturbation_to_input_norm=value,exact_relative_error=float((composed[ix]-actual[ix]).norm()/den),omit_normalizer_term_error=float((scale*norm[ix]).norm()/den),omit_quadratic_term_error=float((scale*quad[ix]).norm()/den),direct_only_error=float((scale*direct[ix]-actual[ix]).norm()/den)))
 result={'pred_a':error<=1e-10,'exact_relative_error':error,'records':records,'read_count':E.shape[0],'direct_component8_to_value9_coefficient':float(scale*(E[-1]@d)),'scope':'Actual weight matrices, random algebra inputs and registered amplitude ratios. Exact four-term raw-read update through MLP8 with actual changed norm; Down bias cancels. Does not generate z8 or a8, preserve final input9 RMS automatically, or establish native term mediation.'}
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
