"""Read-only, CPU diagnostics on previously opened evaluation states; no fitting."""
import csv, hashlib, json, math, time
from pathlib import Path
import torch, tiktoken
from sparse_quartic_bank import features
from paired_root_compiler import cast
from audit_root_feature_conditions import root_features
P=Path(__file__).resolve().parent
SCALE=19054614563.464127

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.perf_counter()
 panel=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][1]
 labels=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)
 ids=torch.load(P.parents[1]/'bilinear_quotient/.rowcache/fineweb_n192_skip7000.pt',weights_only=True)[:32,:65]
 digest=lambda x:hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()
 assert digest(ids)==panel['token_sha256']
 assert digest(ids[:,:64])==labels['token_sha256']['evaluation']
 x=panel['rows'].double();y=labels['panels'][1]['target'].double()/SCALE
 assert x.shape==(2048,1152) and y.shape==(2048,16)
 enc=tiktoken.get_encoding('gpt2');newline=torch.tensor([b'\n' in enc.decode_single_token_bytes(int(i)) for i in ids[:,:64].flatten()])
 programs=[('original384','EXPANDED_ROOT_EMPIRICAL_V1.pt','original')]+[(f'mixed_cp_{s}',f'MIXED_CP_FEATURES_SEED{s}_V1.pt','cp') for s in [1001,1002]]+[(f'shared_support_{s}',f'GAUSSIAN_SUPPORT_EXCHANGE_SEED{s}_V1.pt','shared') for s in [1101,1102]]
 rows=[];csvrows=[];heatmaps=[]
 def fraction_top(a,f):return float(a.topk(max(1,math.ceil(len(a)*f))).values.sum()/a.sum())
 for name,file,kind in programs:
  a=torch.load(P/file,weights_only=True);t=time.perf_counter()
  if kind=='original':pred=root_features(cast(a,torch.float64),x)/SCALE
  elif kind=='cp':
   phi=torch.ones(len(x),a['factors'][0].shape[0],dtype=torch.float64)
   for f in a['factors']:phi*=x@f.double().T
   pred=phi@(a['coefficients'].double()/SCALE).T
  else:pred=features(x,*[f.double() for f in a['factors']],a['pairs'])@(a['coefficients'].double()/SCALE).T
  seconds=time.perf_counter()-t
  residual=pred-y;err=residual.square();e=err.sum(1);energy=y.square().sum(1);total=e.sum();target=energy.sum();common=(target/len(x)).sqrt()
  relative=e.sqrt()/energy.sqrt().clamp_min(1e-30)
  quant=lambda z:{str(q):float(torch.quantile(z,q)) for q in [.5,.9,.95,.99,1.]}
  prefix_e=e.reshape(32,64).sum(1);prefix_y=energy.reshape(32,64).sum(1)
  mean=residual.mean(0);centered=residual-mean;cov=centered.T@centered/len(x);ev,vec=torch.linalg.eigh(cov);ev=ev.flip(0).clamp_min(0);vec=vec.flip(1)
  feature_rows=[]
  for g in range(16):
   feature_rows.append(dict(feature=g,relative_error=float((err[:,g].sum()/y[:,g].square().sum()).sqrt()),residual_energy_share=float(err[:,g].sum()/total),target_energy_share=float(y[:,g].square().sum()/target),bias=float(mean[g]),target_rms=float(y[:,g].square().mean().sqrt())))
  groups=[]
  for group,mask in [('newline',newline),('other',~newline)]+[(f'positions_{i}_{i+15}',(torch.arange(2048)%64>=i)&(torch.arange(2048)%64<i+16)) for i in [0,16,32,48]]:
   groups.append(dict(group=group,count=int(mask.sum()),relative_error=float((e[mask].sum()/energy[mask].sum()).sqrt()),residual_energy_share=float(e[mask].sum()/total),target_energy_share=float(energy[mask].sum()/target)))
  top=[]
  for idx in torch.argsort(e,descending=True)[:12].tolist():
   p,pos=divmod(idx,64);g=int(err[idx].argmax());top.append(dict(flat_index=idx,prefix=p,position=pos,context=enc.decode(ids[p,max(0,pos-23):pos+1].tolist()),current_token=enc.decode([int(ids[p,pos])]),relative_error=float(relative[idx]),error_over_common_target_rms=float(e[idx].sqrt()/common),squared_error_share=float(e[idx]/total),largest_error_feature=g))
  r=dict(name=name,artifact=file,artifact_sha256=hashlib.sha256((P/file).read_bytes()).hexdigest(),prediction_seconds=seconds,pooled_relative_error=float((total/target).sqrt()),row_error_over_common_target_rms=quant(e.sqrt()/common),row_relative_error=quant(relative),row_target_norm_over_common_rms=quant(energy.sqrt()/common),error_concentration={str(f):fraction_top(e,f) for f in [.01,.05,.1]},target_energy_concentration={str(f):fraction_top(energy,f) for f in [.01,.05,.1]},target_energy_on_worst_error_rows={str(f):float(energy[e.topk(math.ceil(len(e)*f)).indices].sum()/target) for f in [.01,.05,.1]},prefix_relative_errors=quant((prefix_e/prefix_y).sqrt()),prefixes=[dict(prefix=i,relative_error=float((prefix_e[i]/prefix_y[i]).sqrt()),residual_energy_share=float(prefix_e[i]/total)) for i in range(32)],features=feature_rows,groups=groups,bias_energy_share=float(len(x)*mean.square().sum()/total),centered_residual_pc_energy_shares=(ev/ev.sum()).tolist(),leading_residual_direction=vec[:,0].tolist(),top_error_contexts=top)
  assert abs(sum(f['residual_energy_share'] for f in feature_rows)-1)<1e-12
  assert abs(float(ev.sum()+mean.square().sum()-total/len(x)))<1e-10*max(1,float(total/len(x)))
  rows.append(r);heatmaps.append((err.reshape(32,64,16).sum(1)/y.square().reshape(32,64,16).sum(1)).sqrt().numpy())
  for i in range(len(x)):csvrows.append(dict(candidate=name,prefix=i//64,position=i%64,residual_norm=float(e[i].sqrt()),target_norm=float(energy[i].sqrt()),relative_error=float(relative[i]),largest_error_feature=int(err[i].argmax())))
  print(name,round(r['pooled_relative_error']*100,3),r['error_concentration'],flush=True)
 result=dict(scope='Previously opened evaluation panel, no training or model forward. 2048 correlated token states in 32 prefixes; 16 scalar output coordinates of the selected MLP16-to-MLP17 pure quartic component, not full-model outputs.',states=2048,prefixes=32,states_per_prefix=64,outputs=16,normalizing_scale=SCALE,rows=rows,analysis_seconds=time.perf_counter()-start)
 (P/'CANDIDATE_RESIDUAL_DISTRIBUTION_V1.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 with (P/'CANDIDATE_RESIDUAL_ROWS_V1.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,len(rows),figsize=(16,7),sharey=True,layout='constrained')
 for ax,r,h in zip(axes,rows,heatmaps):
  im=ax.imshow(h*100,aspect='auto',vmin=0,vmax=60,cmap='magma');ax.set_title(r['name']);ax.set_xlabel('Output feature');ax.set_xticks([0,5,10,15])
 axes[0].set_ylabel('Text prefix (64 token states each)');fig.colorbar(im,ax=axes,label='Relative RMS error (%) — color saturates at 60%')
 fig.suptitle('Residual error by prefix and fixed output feature\nPreviously inspected evaluation panel; individual cells normalized by their own target energy')
 fig.savefig(P/'CANDIDATE_RESIDUAL_HEATMAP_V1.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
