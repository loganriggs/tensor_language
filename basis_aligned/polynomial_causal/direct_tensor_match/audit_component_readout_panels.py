"""Recompute native component values and disclose calibration normalization by panel."""
from pathlib import Path
import json,torch
from compact_source_graph import component_scalars
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
def main():
 d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);extra=torch.load(P/'EXPANDED_COMPONENT_INPUTS_V1.pt',weights_only=True);cache=torch.load(P/'EXPANDED_COVARIANCE_STATES_V1.pt',weights_only=True)
 meta=json.loads((P/'COMPONENT_PAIR_READOUT_V1.json').read_text());programs=torch.load(P/'COMPONENT_PAIR_READOUT_PROGRAMS_V1.pt',weights_only=True)
 oldtruth=torch.stack([p['truth'] for p in d['pairs']],1)
 ids=d['indices'];panels={'old1536':(d['z'][:1536],d['h'][:1536],oldtruth[:1536]),'additional14848':(cache['z'].flatten(0,1).double(),extra['h'].flatten(0,1).double(),extra['native_phi'].flatten(0,1)),'opened448':(d['z'][ids],d['h'][ids],oldtruth[ids])}
 train=torch.cat([panels['old1536'][2],panels['additional14848'][2]]);trainstd=train.std(0,correction=0);stats={}
 for name,(z,h,y) in panels.items():
  centered=(y-train.mean(0)).square();energy=centered.sum(0);count=max(1,len(y)//100)
  stats[name]=dict(n=len(y),mean=y.mean(0).tolist(),std=y.std(0,correction=0).tolist(),std_relative_to_full_calibration=(y.std(0,correction=0)/trainstd).tolist(),top_one_percent_target_energy_fraction=(centered.topk(count,dim=0).values.sum(0)/energy).tolist())
 rows=[];max_replay=0
 for key in meta['winners'].values():
  p=programs[key];rec=next(r for r in meta['records'] if r['key']==key);sums=[]
  for name,(z,h,y) in panels.items():
   pred=component_scalars(z,h,p);E=(pred-y).square();rmse=E.mean(0).sqrt();values=(rmse/y.std(0,correction=0)).tolist()
   if name=='opened448':max_replay=max(max_replay,max(abs(a-b) for a,b in zip(values,rec['per_mode_errors'])))
   else:sums.append(E.sum(0))
   count=max(1,len(y)//100)
   rows.append(dict(key=key,panel=name,relative_component_errors=values,absolute_rmse=rmse.tolist(),error_normalized_by_full_calibration_std=(rmse/trainstd).tolist(),top_one_percent_error_energy_fraction=(E.topk(count,dim=0).values.sum(0)/E.sum(0)).tolist()))
  full=(sum(sums)/len(train)).sqrt()/trainstd;max_replay=max(max_replay,max(abs(a-b) for a,b in zip(full.tolist(),rec['calibration_component_errors'])))
 assert max_replay<1e-8
 out=dict(panel_statistics=stats,rows=rows,maximum_execution_replay=max_replay,scope='All panels use original fixed component coordinates. Historical calibration diagnostic, no document independence or fresh validation. Normalization differences are disclosed, not used to revise original gates.')
 (P/'COMPONENT_READOUT_PANEL_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
