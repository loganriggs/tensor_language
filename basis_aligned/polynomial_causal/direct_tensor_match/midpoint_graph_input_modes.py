"""Exact input-mode Grams of implicit weighted bilinear graph tensor."""
from pathlib import Path
import torch,json,time
from weighted_bilinear_svd import roots
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.perf_counter()
e=torch.load(p/'MIDPOINT_GRAPH_FROZEN_V1.pt',weights_only=True)['rank8'];graph=torch.load(p/'MIDPOINT_OUTPUT_CORRECTION_GRAPHS_V1.pt',weights_only=True)['rank8'];rows=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True);n=rows['n'].flatten(0,1).double();m=rows['m'].flatten(0,1).double();y=rows['y'].flatten(0,1).double()-e['full_mean'];S=torch.load(p/'MIDPOINT_CENTERED_V1.pt',weights_only=True)['metric_sqrt'].double();A=e['A'].double();B=e['B'].double();C=e['reduced_writers'].double();Gc=(S@C).T@(S@C);norm=(y@S).norm();original=(((n@A)*(m@B))-e['product_mean'])@C.T;records=[];programs={};graphs={};replays=[]
# Toy: exact dense unfoldings match Hadamard-Gram construction.
gen=torch.Generator().manual_seed(261223);aa=torch.randn(5,7,generator=gen,dtype=torch.float64);bb=torch.randn(4,7,generator=gen,dtype=torch.float64);cc=torch.randn(3,7,generator=gen,dtype=torch.float64);tensor=torch.einsum('vk,ik,jk->vij',cc,aa,bb);unfold=tensor.permute(1,0,2).reshape(5,-1);toy=float((aa@((bb.T@bb)*(cc.T@cc))@aa.T-unfold@unfold.T).norm()/(unfold@unfold.T).norm());assert toy<1e-12
for metric in ['isotropic','separable_moment']:
 eye=torch.eye(1152,dtype=torch.float64);Sn,In=(eye,eye) if metric=='isotropic' else roots(n.T@n/len(n));Sm,Im=(eye,eye) if metric=='isotropic' else roots(m.T@m/len(m));Aw=Sn@A;Bw=Sm@B;Ga=Aw.T@Aw;Gb=Bw.T@Bw;total=(Ga*Gb*Gc).sum();Hn=Aw@(Gb*Gc)@Aw.T;Hm=Bw@(Ga*Gc)@Bw.T;_,Un=torch.linalg.eigh((Hn+Hn.T)/2);_,Um=torch.linalg.eigh((Hm+Hm.T)/2);Un=Un.flip(1);Um=Um.flip(1)
 for rank in [32,64,128,256,512,1024]:
  Tn=Un[:,:rank].T@Aw;Tm=Um[:,:rank].T@Bw;Pn=In@Un[:,:rank];Pm=Im@Um[:,:rank];Ah=Pn@Tn;Bh=Pm@Tm;products=((n@Pn)@Tn)*((m@Pm)@Tm);mu=products.mean(0);pred=(products-mu)@C.T;originalmean=(products-e['product_mean'])@C.T;kept=((Tn.T@Tn)*(Tm.T@Tm)*Gc).sum();coefficient_error=float(((total-kept).clamp_min(0)/total).sqrt());err=float(((pred-y)@S).norm()/norm);referr=float(((pred-original)@S).norm()/norm)
  records.append(dict(metric=metric,input_rank=rank,products=1024,weight_coefficients=2*rank*(1152+1024)+312320,full_calibration_variation_error=err,original_mean_full_error=float(((originalmean-y)@S).norm()/norm),change_over_full_variation=referr,relative_coefficient_error_to_frozen=coefficient_error))
  if rank==1024:replays.append(float((originalmean-original).norm()/original.norm()))
  if metric=='separable_moment' and rank in [64,128,256]:
   key=f'rank{rank}';programs[key]=dict(A=Ah.float(),B=Bh.float(),product_mean=mu,full_mean=e['full_mean'],reduced_writers=C);graphs[key]=dict(Pn=Pn.float(),Pm=Pm.float(),Tn=Tn.float(),Tm=Tm.float(),product_mean=mu,full_mean=e['full_mean'],group_writers=graph['group_writers'],correction_left=graph['correction_left'],correction_writers=graph['correction_writers'])
  print(metric,rank,err,flush=True)
assert max(replays)<1e-8
out=p/'MIDPOINT_GRAPH_INPUT_MODES_V1.json';assert not out.exists();out.write_text(json.dumps(dict(records=records,toy_gram_replay=toy,full_span_replay=max(replays),seconds=time.perf_counter()-start,scope='Shared input subspaces from exact modeGrams of frozen approximate graph, with fixed output metric. Both isotropic and separable input metrics. Means recentered calibration only, original-mean error also reported. Fullnative target calibrationerror separate from coefficienterror relative tofrozen approximation. Noheldfitting.'),indent=2)+'\n');torch.save(programs,p/'MIDPOINT_GRAPH_INPUT_MODES_V1.pt');torch.save(graphs,p/'MIDPOINT_GRAPH_INPUT_MODE_GRAPHS_V1.pt')
