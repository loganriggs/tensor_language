"""Joint eight-to-six product edits, four disjoint multi-output neighborhoods.

pred_a: exact reduced-core/ambient metric replay <1e-10 and planted rank3
recovery error<1e-3. pred_b: at least2/4 groups achieve <=.75 squared error
of exhaustive best six-input-product subset with output refitting.
pred_c: at least2 groups have two independent random starts within5% squared
error of winner (same candidate class; includes optimizer/rate alternatives).
Fixed metric, no new data/model evaluation; adopted edits must pass pred_b
individually. This does not prove semantic identity or native transfer.
"""
from pathlib import Path
import json,itertools,time,torch
from joint_product_refactor import reconstruct,best_subset,fit
p=Path(__file__).resolve().parent;torch.set_num_threads(2)
out=p/'MIDPOINT_JOINT_PRODUCT_REFACTOR_V1.json';assert not out.exists();start=time.perf_counter()
# Planted dense CP core: oracle replay and actual recovery, not just zero-loss evaluation.
gen=torch.Generator().manual_seed(261314)
f=[torch.randn(3,8,dtype=torch.float64,generator=gen) for _ in range(3)]
toy=torch.einsum('ri,rj,rk->ijk',*f);oracle,_=reconstruct(toy,f[0],f[1]);toyfit=fit(toy,3,'adam',.05,88,600)
control=dict(oracle_relative_squared_error=float(oracle/toy.square().sum()),random_recovery_relative_error=toyfit[0]**.5)
print('CONTROL',control,flush=True)
e=torch.load(p/'MIDPOINT_PRIVATE_FROZEN_V1.pt',weights_only=True)['private512'];rows=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True)
n=rows['n'].flatten(0,1).double();m=rows['m'].flatten(0,1).double();nb=n.mean(0);mb=m.mean(0);n-=nb;m-=mb
S=torch.load(p/'MIDPOINT_CENTERED_V1.pt',weights_only=True)['metric_sqrt'].double();A=e['A'].double();B=e['B'].double();W=e['reduced_writers'].double()
factors=[n@A/len(n)**.5,m@B/len(m)**.5,S@W];grams=[v.T@v for v in factors];scales=[g.diag().clamp_min(1e-30).sqrt() for g in grams];corr=[g/s[:,None]/s[None,:] for g,s in zip(grams,scales)];amp=scales[0]*scales[1]*scales[2]
# Choose four disjoint neighborhoods: strongest available seed, closest input products.
unused=set(range(512));groups=[]
for _ in range(4):
 seed=max(unused,key=lambda j:float(amp[j]));score=(corr[0][seed]*corr[1][seed]).abs();order=score.argsort(descending=True).tolist();ids=[j for j in order if j in unused][:8];groups.append(ids);unused.difference_update(ids)
records=[];exports=[];replays=[]
for group,ids in enumerate(groups):
 ix=torch.tensor(ids)
 roots=[]
 for g in grams:
  ev,U=torch.linalg.eigh(g[ix[:,None],ix[None,:]])
  roots.append(ev.clamp_min(0).sqrt()[:,None]*U.T)
 core=torch.einsum('ik,jk,lk->ijl',*roots)
 exact_energy=(grams[0][ix[:,None],ix[None,:]]*grams[1][ix[:,None],ix[None,:]]*grams[2][ix[:,None],ix[None,:]]).sum()
 replay=float(abs(core.square().sum()-exact_energy)/exact_energy);assert replay<1e-10;replays.append(replay)
 base=best_subset(core,roots[0].T,roots[1].T,6)
 baseline=base[0]/float(exact_energy)
 bounds=[]
 for mode in range(3):bounds.append(float(torch.linalg.svdvals(core.movedim(mode,0).reshape(8,64))[6:].square().sum()/exact_energy))
 runs=[];winner=None
 for optimizer,lr,init in itertools.product(['adam','muon'],[.01,.05],['subset','random0','random1']):
  seed=261315+(1 if init=='random1' else 0)
  fitted=fit(core,6,optimizer,lr,seed,400,initial=(base[2],base[3]) if init=='subset' else None)
  row=dict(optimizer=optimizer,lr=lr,initialization=init,relative_squared_error=fitted[0],error_vs_subset=fitted[0]/baseline,best_step=fitted[4]);runs.append(row)
  if winner is None or fitted[0]<winner[0]:winner=fitted
  print('FIT',group,row,flush=True)
 accept=winner[0]<=.75*baseline
 # Independent random initializations must both reproduce the winner within5%.
 best_random=[min(v['relative_squared_error'] for v in runs if v['initialization']==f'random{k}') for k in [0,1]]
 stable=all(v<=1.05*winner[0] for v in best_random)
 rec=dict(group=group,products=ids,baseline_subset_ids=[ids[j] for j in base[1]],baseline_relative_squared_error=baseline,unfolding_lower_bound=max(bounds),winner_relative_squared_error=winner[0],winner_error_vs_subset=winner[0]/baseline,accepted=accept,random_restart_near_winner=stable,runs=runs);records.append(rec)
 if accept:
  aa=A[:,ix]@torch.linalg.pinv(roots[0])@winner[1].T
  bb=B[:,ix]@torch.linalg.pinv(roots[1])@winner[2].T
  ww=W[:,ix]@torch.linalg.pinv(roots[2])@winner[3]
  # Direct physical Gram error verifies inverse coordinate mapping.
  ar=torch.cat([A[:,ix],aa],1);br=torch.cat([B[:,ix],bb],1);wr=torch.cat([W[:,ix],-ww],1)
  gg=[(n@ar).T@(n@ar)/len(n),(m@br).T@(m@br)/len(m),(S@wr).T@(S@wr)]
  physical=float((gg[0]*gg[1]*gg[2]).sum()/exact_energy)
  assert abs(physical-winner[0])<1e-9
  rec['ambient_residual_replay']=abs(physical-winner[0]);exports.append((ids,aa,bb,ww))
result=dict(control=control,groups=records,predictions=dict(pred_a_instrument=max(replays)<1e-10 and control['random_recovery_relative_error']<1e-3,pred_b_two_useful_edits=sum(r['accepted'] for r in records)>=2,pred_c_two_repeatable_fits=sum(r['random_restart_near_winner'] for r in records)>=2),seconds=time.perf_counter()-start,scope='Joint local edits of a frozen approximation under fixed independent calibration-role covariance metric. Exhaustive subset comparator refits outputs. Lower bounds are necessary, not attained certificates. No semantic identity or native intervention claim.')
if exports:
 keep=sorted(set(range(512))-set(j for ids,*_ in exports for j in ids))
 aa=torch.cat([A[:,keep]]+[v[1] for v in exports],1);bb=torch.cat([B[:,keep]]+[v[2] for v in exports],1);ww=torch.cat([W[:,keep]]+[v[3] for v in exports],1)
 new={k:v for k,v in e.items()};new.update(A=aa,B=bb,reduced_writers=ww,base_left_mean=nb@aa,base_right_mean=mb@bb,product_mean=torch.zeros(aa.shape[1],dtype=aa.dtype))
 torch.save({'original512':e,'joint_refactor':new},p/'MIDPOINT_JOINT_PRODUCT_REFACTOR_GRAPHS_V1.pt')
 result['exported_products']=aa.shape[1]
out.write_text(json.dumps(result,indent=2)+'\n');print('RESULT',result['predictions'],flush=True)
