"""Exploratory position confound screen for frozen output-shared features."""
import json
from pathlib import Path
import torch
from frozen_program_evaluation import quartic
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(3);torch.set_grad_enabled(False)
 panel=next(p for p in torch.load(P/'FROZEN_FRESH_VALIDATION_V1.pt',weights_only=True)['panels'] if p['context']==256);view=torch.load(P/'CANONICAL_ROOT_FEATURES_V1.pt',weights_only=True);U=view['output_directions'].double();mu=view['output_mean'].double();s={k:t.double() for k,t in torch.load(P/'SKIP_METRIC_BLEND_PROGRAM_V1.pt',weights_only=True)['programs'][.5].items()};x=panel['rows'].double();ys={'native':panel['targets'].double(),'primary':quartic(s,x)};records=[];rng=torch.Generator().manual_seed(260935)
 for name,y in ys.items():
  a=((y-mu)@U).reshape(32,256,4);center=a-a.mean(1,keepdim=True);total=center.square().mean((0,1));stat=center.mean(0).square().mean(0)/total;null=[]
  for _ in range(500):
   shifts=torch.randint(256,(32,),generator=rng);ix=(torch.arange(256)[None,:]+shifts[:,None])%256;shifted=center[torch.arange(32)[:,None],ix];null.append(shifted.mean(0).square().mean(0)/total)
  null=torch.stack(null);z=torch.log1p(torch.arange(256,dtype=torch.float64));z=z-z.mean();corr=(center*z[None,:,None]).mean((0,1))/(total*z.square().mean()).sqrt();first=(center[:,:8].square().sum((0,1))/center.square().sum((0,1)))
  records.append(dict(source=name,position_mean_variance_fraction=stat.tolist(),circular_shift_null_mean=null.mean(0).tolist(),circular_shift_null_95=torch.quantile(null,.95,dim=0).tolist(),log_position_correlation=corr.tolist(),first8_positions_fraction_of_centered_energy=first.tolist(),scope='Within-document centering. Circular shifts preserve each trajectory but randomize alignment; exploratory reference, not a definitive null for nonstationary language.'))
 result=dict(records=records,documents=32,context=256,seed=260935,permutations=500,scope='Reused panel, frozen canonical basis and primary blend. Position association is descriptive and does not establish semantic meaning or causal position dependence.')
 (P/'MODE_POSITION_AUDIT_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
