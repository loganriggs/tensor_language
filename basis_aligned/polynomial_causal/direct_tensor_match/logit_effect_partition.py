"""Exact common-vocabulary and centered decomposition after native logit operations."""
import torch

def partition(native,predicted):
 # Last axis is vocabulary; rows may include batch/position dimensions.
 a=native.double();b=predicted.double();am=a.mean(-1,keepdim=True);bm=b.mean(-1,keepdim=True);ac=a-am;bc=b-bm;v=a.shape[-1]
 return dict(native_centered_effect_energy=float(ac.square().sum()),predicted_centered_effect_energy=float(bc.square().sum()),centered_effect_dot=float((ac*bc).sum()),centered_effect_error_energy=float((ac-bc).square().sum()),native_common_effect_energy=float(v*am.square().sum()),predicted_common_effect_energy=float(v*bm.square().sum()),common_effect_error_energy=float(v*(am-bm).square().sum()))

def position_partition(native,predicted,positions):
 bins={}
 for lo,hi in [(0,64),(64,128),(128,256)]:
  mask=(positions>=lo)&(positions<hi)
  if bool(mask.any()):bins[f'{lo}:{hi}']=dict(sites=int(mask.sum()),**partition(native[mask],predicted[mask]))
 assert sum(v['sites'] for v in bins.values())==len(positions)
 return bins

def toy_check():
 g=torch.Generator().manual_seed(261003);a=torch.randn(7,11,generator=g,dtype=torch.float64);b=torch.randn(7,11,generator=g,dtype=torch.float64);r=partition(a,b);total=float((a-b).square().sum());replay=abs(r['centered_effect_error_energy']+r['common_effect_error_energy']-total)/total;assert replay<1e-12
 assert partition(torch.ones(3,5),2*torch.ones(3,5))['native_centered_effect_energy']==0
 centered=torch.tensor([[-1.,0.,1.]]);assert partition(centered,2*centered)['native_common_effect_energy']==0
 # Row-specific common shifts preserve log-softmax exactly in real arithmetic.
 shifted=a+torch.randn(7,1,generator=g,dtype=torch.float64);softmax=float((a.log_softmax(-1)-shifted.log_softmax(-1)).abs().max());assert softmax<1e-12
 return dict(energy_replay=replay,common_shift_logsoftmax_replay=softmax)
if __name__=='__main__':
 import json
 from pathlib import Path
 r=toy_check();Path(__file__).with_name('LOGIT_EFFECT_PARTITION_ORACLE_V1.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
