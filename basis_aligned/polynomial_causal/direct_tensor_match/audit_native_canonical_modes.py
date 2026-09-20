"""Compare frozen canonical scalar predictions to the actual isolated native target."""
import json
from pathlib import Path
import torch
from frozen_program_evaluation import quartic
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(4);torch.set_grad_enabled(False)
 view=torch.load(P/'CANONICAL_ROOT_FEATURES_V1.pt',weights_only=True)
 U=view['output_directions'].double();mu=view['output_mean'].double()
 s={k:v.double() for k,v in torch.load(P/'FUSED_ROOT_PROGRAM_V1.pt',weights_only=True)['programs'][4].items()}
 panels=torch.load(P/'FROZEN_FRESH_VALIDATION_V1.pt',weights_only=True)['panels'];records=[]
 for panel in panels:
  x=panel['rows'].double();native=panel['targets'].double();pred=quartic(s,x)
  # Fixed calibration mean for both branches, not test-panel centering.
  a=(native-mu)@U;b=(pred-mu)@U
  for g in range(4):
   n=a[:,g];v=b[:,g];nc=n-n.mean();vc=v-v.mean()
   records.append(dict(context=panel['context'],mode=g,relative_error=float((v-n).norm()/n.norm()),centered_error=float((vc-nc).norm()/nc.norm()),correlation=float(nc@vc/(nc.norm()*vc.norm())),native_mean=float(n.mean()),predicted_mean=float(v.mean()),native_second_moment=float(n.square().mean())))
 result=dict(records=records,scope='Post-hoc scalar diagnostic on reused panels; no fitted parameters or causal claim. Fixed output directions and calibration mean from canonical archive.')
 (P/'NATIVE_CANONICAL_MODE_DIAGNOSTIC_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
