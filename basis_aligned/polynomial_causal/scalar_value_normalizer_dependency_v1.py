"""Audit supplied denominator dependence in the compact value-change result.
A exact split<=1e-5; B oracle numerator/frozen norm and normpiece<=10%;
C rank4 anchored numerator/frozen norm<=5%. No new causal model forwards.
"""
from pathlib import Path
import json,torch,time
import torch.nn.functional as F
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);tic=time.perf_counter();out=P/'SCALAR_VALUE_NORMALIZER_DEPENDENCY_V1_RESULT.json';assert not out.exists();program=torch.load(P/'SCALAR_VALUE_GENERATOR_MLP8_V1_PROGRAM.pt',weights_only=True);cache=torch.load(P/'SCALAR_PRODUCER_MLP_BRIDGE_CACHE_V1_ARTIFACT.pt',weights_only=True);rows=json.loads((P/'STRUCTURED_PRODUCER_CONFIRMATION_V1_ROWS.json').read_text())['rows'];d=torch.load(P/'SCALAR_PRODUCER_NATIVE_LIFT_V1_ARTIFACT.pt',weights_only=True)['writers'][0].double();r=program['reader'];lam=program['lambdas'][0];eps=torch.finfo(torch.float32).eps
 raw=torch.stack([torch.cat([cache['r9'][s,i,:len(row['ids'])] for i,row in enumerate(rows)]) for s in [0,1]]);den=(raw.square().mean(-1)+eps).sqrt().double();inv=den.reciprocal();num=raw.double()@r;y=F.rms_norm(raw,(1152,),eps=eps).double()@r;delta=y[1]-y[0];dn=num[1]-num[0];di=inv[1]-inv[0];numerator=dn*(inv[1]+inv[0])/2;normalizer=(num[1]+num[0])*di/2
 z0=torch.cat([cache['z8'][i,:len(row['ids'])] for i,row in enumerate(rows)]).double();amp=torch.cat([cache['a8'][i,:len(row['ids'])] for i,row in enumerate(rows)]);z=torch.stack((z0,z0-amp[:,None]*d));rho=z.square().mean(-1)+eps;ev=program['eigenvalues'];vec=program['eigenvectors'];project=z@vec;terms=project.square()*ev;records=[]
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30));aligned=lambda a:float((a@delta)/delta.square().sum())
 for rank in [0,1,4,16,64,1152]:
  nhat=lam*((z@r)+terms[...,:rank].sum(-1)/rho);dn_hat=nhat[1]-nhat[0];frozen=dn_hat*inv[0];with_oracle_norm=dn_hat*inv[1]+num[0]*di
  records.append(dict(rank=rank,numerator_change_error=rel(dn_hat,dn),frozen_norm_change_error=rel(frozen,delta),oracle_changed_norm_anchored_error=rel(with_oracle_norm,delta)))
 split=rel(numerator+normalizer,delta);oracle_frozen=rel(dn*inv[0],delta);normratio=float(normalizer.norm()/delta.norm());rank4=next(x for x in records if x['rank']==4)
 result=dict(pred_a=split<=1e-5,pred_b=oracle_frozen<=.1 and normratio<=.1,pred_c=rank4['frozen_norm_change_error']<=.05,split_replay_error=split,oracle_numerator_frozen_norm_error=oracle_frozen,numerator_aligned_fraction=aligned(numerator),normalizer_aligned_fraction=aligned(normalizer),numerator_norm_ratio=float(numerator.norm()/delta.norm()),normalizer_norm_ratio=normratio,normalizer_numerator_cosine=float((normalizer@numerator)/(normalizer.norm()*numerator.norm())),records=records,seconds=time.perf_counter()-tic,scope='Existing broaderhead8removal cache, actual native numerator and9RMSreciprocal. Symmetric product allocation avoids arbitrary ordering; aligned fractions can be signed or exceedone. Anchored lowrank rows use nativebaseline and, whenlabelled, actualchangednorm. These are scalar dependency diagnostics, not selectivecomponent/logit or autonomous extraction results.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
