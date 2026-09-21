from pathlib import Path
import torch,json,time
from weighted_bilinear_svd import roots
from joint_product_selection import select,refit,toy_check
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.perf_counter();toy=toy_check()
e=torch.load(p/'MIDPOINT_GRAPH_INPUT_MODE_GRAPHS_V1.pt',weights_only=True)['rank256'];e={k:v.double() for k,v in e.items()};A=e['Pn']@e['Tn'];B=e['Pm']@e['Tm'];rows=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True);n=rows['n'].flatten(0,1).double();m=rows['m'].flatten(0,1).double();y=rows['y'].flatten(0,1).double()-e['full_mean'];S=torch.load(p/'MIDPOINT_CENTERED_V1.pt',weights_only=True)['metric_sqrt'].double();Sn,_=roots(n.T@n/len(n));Sm,_=roots(m.T@m/len(m));Aw=Sn@A;Bw=Sm@B;G=(Aw.T@Aw)*(Bw.T@Bw);scale=G.diag().sqrt();K=G/(scale[:,None]*scale[None,:]);H=torch.zeros(256,1024,dtype=torch.float64);H[torch.arange(1024)//4,torch.arange(1024)]=1;Q,R=torch.linalg.qr(torch.cat([e['group_writers'],e['correction_writers']],1),mode='reduced');C=R@torch.cat([H,e['correction_left'].T],0);Cn=C*scale;metric_map=S@Q;Cw=metric_map@Cn;energy=((Cw.T@Cw)*K).sum();orders={'joint_greedy':select(K,Cw@K,768),'individual_energy':Cw.square().sum(0).argsort(descending=True)};records=[];graphs={};norm=(y@S).norm();allphi=(n@A)*(m@B)
for policy,order in orders.items():
 for width in [128,256,512,768]:
  ids=order[:width];writer,normal,rank=refit(K,Cn,ids);# Exact implicit residual norm, including output cross terms.
  W=metric_map@writer;error=energy+((W.T@W)*K[ids][:,ids]).sum()-2*(W*(Cw@K[:,ids])).sum();coefficient=float((error.clamp_min(0)/energy).sqrt());core=writer/scale[ids];phi=allphi[:,ids];mean=phi.mean(0);prediction=(phi-mean)@core.T@Q.T;cal=float(((prediction-y)@S).norm()/norm)
  direct_cost=2*256*(1152+width)+1152*width;factored_cost=2*256*(1152+width)+264*(1152+width);usefactored=factored_cost<direct_cost
  program=dict(Pn=e['Pn'].float(),Pm=e['Pm'].float(),Tn=e['Tn'][:,ids].float(),Tm=e['Tm'][:,ids].float(),product_mean=mean,full_mean=e['full_mean'])
  if usefactored:program.update(output_basis=Q,output_core=core)
  else:program['reduced_writers']=Q@core
  records.append(dict(policy=policy,products=width,coefficient_error_to_shared_graph=coefficient,full_calibration_variation_error=cal,weight_coefficients=min(direct_cost,factored_cost),output_factored=usefactored,normal_equation_error=normal,retained_rank=rank))
  if policy=='joint_greedy' and width in [256,512]:graphs[f'products{width}']=program
  print(policy,width,coefficient,cal,flush=True)
out=p/'MIDPOINT_GRAPH_PRODUCT_PRUNE_V1.json';assert not out.exists();out.write_text(json.dumps(dict(toy=toy,records=records,seconds=time.perf_counter()-start,scope='Greedy deletion/reselection of existing learnedproducts followedby exact conditional output refit under separableweighted coefficientmetric. Frozen sharedinputgraph is teacher, originalnativefullcalibration error separatelyreported. Exactcompactoutputframe priced; meansrecentered calibrationonly. Noheldfit or nativeclaims.'),indent=2)+'\n');torch.save(graphs,p/'MIDPOINT_GRAPH_PRODUCT_PRUNE_V1.pt')
