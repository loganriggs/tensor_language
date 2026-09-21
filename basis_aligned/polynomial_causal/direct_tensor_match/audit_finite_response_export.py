"""Independent endpoint replay of the exported finite-response fixed program, CPU."""
import json,time
from pathlib import Path
import torch

def main():
 torch.set_num_threads(2);start=time.monotonic();p=Path(__file__).resolve().parent
 state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True,mmap=True,map_location='cpu')
 def w(k):return state[k].double()
 data=torch.load(p/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);h,z=data['h'].double(),data['z'].double();source=w('transformer.h.17.lambdas')[0]*((z@w('transformer.h.16.mlp.Left.weight').T)*(z@w('transformer.h.16.mlp.Right.weight').T))@w('transformer.h.16.mlp.Down.weight').T
 norm=lambda x:x/(x.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt()
 x0,x1=norm(h),norm(h+source.roll(257,0)-source);mu=x0.mean(0)
 L,R,D=[w('transformer.h.17.mlp.'+k+'.weight') for k in ['Left','Right','Down']]
 program={k:v.double() for k,v in torch.load(p/'FULL_QUADRATIC_FINITE_FIXED_PROGRAM_V1.pt',weights_only=True).items()};a,b,writer=program['a'],program['b'],program['writer']
 RU=torch.load(p/'MIDPOINT_CONTINUATION_NATIVE_OPERATOR_V1.pt',weights_only=True)['R_U'].double()
 teacher=lambda x:((x@L.T)*(x@R.T))@D.T
 student_q=lambda x:((x@a.T)*(x@b.T))@writer.T
 student=lambda x:student_q(x)+x@program['linear'].T+program['bias']
 direct=(teacher(x1)-teacher(x0))-(student(x1)-student(x0))
 centered=(teacher(x1-mu)-teacher(x0-mu))-(student_q(x1-mu)-student_q(x0-mu))
 denominator=((teacher(x1-mu)-teacher(x0-mu))@RU.T).norm()
 error=float((direct@RU.T).norm()/denominator);equivalence=float(((direct-centered)@RU.T).norm()/denominator)
 result=dict(affine_repaired_direct_error=error,centered_error=float((centered@RU.T).norm()/denominator),affine_centering_equivalence_error=equivalence,passed=equivalence<1e-5 and abs(error-.04545614629384554)<1e-5,seconds=time.monotonic()-start,scope='Direct teacher/student endpoint evaluation of FP32export in FP64 arithmetic; QR metric uses original saved frame. Fixed arm only; no final normalization/softcap. Expected error from terminal fixed-arm fitting log, independent of midpoint feature helper.')
 assert result['passed'];(p/'FINITE_RESPONSE_EXPORT_AUDIT_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
