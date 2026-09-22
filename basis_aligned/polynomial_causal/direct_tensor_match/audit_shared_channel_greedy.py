"""Native selected-output quadratic baseline with shared channel selection."""
import json,time
import torch
from audit_conditional_residual_accounting import P,SCALE
from audit_root_matched_reader import CK
from shared_channel_greedy import gram,greedy,refit

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic()
 state=torch.load(CK,weights_only=True,mmap=True,map_location='cpu');W=torch.load(P/'EXPANDED_ROOT_EMPIRICAL_V1.pt',weights_only=True)['writer'].double();U=state['lm_head.weight'].double();UW=U@W;readers=U.T@UW/UW.square().sum(0);del U,UW
 get=lambda n:state[f'transformer.h.17.mlp.{n}.weight'].double()
 L,R,D=get('Left'),get('Right'),get('Down');C=readers.T@D/SCALE
 K=gram(L,R);norm=K.diagonal().sqrt();K/=norm[:,None];K/=norm[None,:];C*=norm
 B=C@K;energy=(B*C).sum(1);rows=[];checks=[]
 print('Native Gram ready',time.monotonic()-start,flush=True)
 for metric in ['natural','equal_output']:
  weight=torch.ones_like(energy) if metric=='natural' else energy.rsqrt();weighted_B=B*weight[:,None]
  ids,gains=greedy(K,weighted_B,512);ranking=(C*weight[:,None]).square().sum(0).argsort(descending=True).tolist()
  for strategy,indices in [('greedy',ids),('individual_energy',ranking)]:
   for width in [64,128,256,512]:
    take=indices[:width];fit,gain,normal=refit(K,B,take);residual=(energy-gain).clamp_min(0)
    full=C.clone();full[:,take]-=fit;explicit=(full*(full@K)).sum(1)
    identity=float((explicit-residual).abs().max()/energy.max());assert max(identity,normal)<1e-8
    if strategy=='greedy':
     gap=float(((gain*weight.square())-gains[width-1]).abs().max()/(energy*weight.square()).max());assert gap<1e-8
    else:gap=None
    row=dict(metric=metric,strategy=strategy,products=width,stored_floats=width*(2*1152+16),natural_centered_error=float((residual.sum()/energy.sum()).sqrt()),equal_output_centered_error=float((residual/energy).mean().sqrt()),per_output_centered_errors=(residual/energy).sqrt().tolist(),selected_channels=take,normal_residual=normal,explicit_residual_identity=identity,greedy_gain_identity=gap)
    rows.append(row);print(metric,strategy,width,row['natural_centered_error'],row['equal_output_centered_error'],flush=True)
  field='natural_centered_error' if metric=='natural' else 'equal_output_centered_error';best=next(r for r in rows if r['metric']==metric and r['strategy']=='greedy' and r['products']==512);base=next(r for r in rows if r['metric']==metric and r['strategy']=='individual_energy' and r['products']==512);checks.append(dict(metric=metric,relative_error_ratio=best[field]/base[field],pred_improvement=best[field]<=.8*base[field]))
 result=dict(rows=rows,predictions=checks,seconds=time.monotonic()-start,native_baseline=dict(products=4608,stored_floats=10690560,error=0),scope='Selected16-output nativeMLP17 quadratic. Shared teacher channels retained; exact coefficient readout fit; greedy subset not globally optimal. All errors centered isotropic Gaussian/coefficient, not text fidelity.')
 (P/'SHARED_CHANNEL_GREEDY_V1.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
