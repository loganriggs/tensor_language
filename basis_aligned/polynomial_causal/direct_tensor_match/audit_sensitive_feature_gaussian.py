"""Post-fit synthetic native polynomial test; not normalized-model OOD validation."""
import json
from pathlib import Path
import torch
from paired_root_compiler import cast
from audit_root_feature_conditions import root_features
from audit_root_matched_reader import CK
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);torch.manual_seed(926);x=torch.randn(256,1152,dtype=torch.float64)
 base=cast(torch.load(P/'EXPANDED_ROOT_EMPIRICAL_V1.pt',weights_only=True),torch.float64);s=torch.load(CK,weights_only=True,mmap=True,map_location='cpu');uv=s['lm_head.weight'].double();uw=uv@base['writer'];readers=uv.T@uw/(uw.square().sum(0));del uv,uw
 def w(layer,name):return s[f'transformer.h.{layer}.mlp.{name}.weight'].double()
 m=((x@w(16,'Left').T)*(x@w(16,'Right').T))@(s['transformer.h.17.lambdas'][0]*w(16,'Down')).T;y=((m@w(17,'Left').T)*(m@w(17,'Right').T))@(readers.T@w(17,'Down')).T
 reference=root_features(base,x);rows=[]
 for name,file in [('inherited','EXPANDED_ROOT_EMPIRICAL_V1.pt'),('fixed_sensitive','SENSITIVE_ROOT_SENSITIVE_V2.pt'),('learned_uniform','SENSITIVE_ROOT_FEATURE_UNIFORM_V1.pt'),('learned_sensitive','SENSITIVE_ROOT_FEATURE_SENSITIVE_V1.pt')]:
  program=cast(torch.load(P/file,weights_only=True),torch.float64);h=root_features(program,x);r=(h-y).square().sum(0).sqrt()/y.square().sum(0).sqrt();rows.append(dict(program=name,native_aggregate_error=float((h-y).norm()/y.norm()),native_root1_error=float(r[1]),mean_native_root_error=float(r.mean()),inherited_disagreement=float((h-reference).norm()/reference.norm())))
 result=dict(seed=926,probes=256,distribution='N(0,I1152), eachslot samevector, no RMS/modeltext',rows=rows,scope='Nativepurequartic16fixedreader target, postfit synthetic diagnostic. Not coefficientFrobeniusnorm or native text/OODbehavior. No model selection on probes.')
 (P/'SENSITIVE_FEATURE_GAUSSIAN_AUDIT_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
